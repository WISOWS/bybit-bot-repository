"""Portfolio backtest of regime_switch_hybrid over combinations of pairs.

- One shared $100k account, leverage 5x.
- Risk 0.75%/trade (sized off current realized portfolio balance).
- Global cap: max 2 concurrent open positions across the whole portfolio.
- Taker fee 0.055%/side, slippage 0.05%/side, funding -0.01%/8h on shorts.
- Candidates: top-10 pairs by PF (history>1.5y, trades>80, excl NEAR/SOL/LINK/ENA).
- Tests every 3- and 4-pair combination; reports top-5 by annual % with DD<8%,
  weighting low cross-pair correlation as a diversification tie-breaker.
"""

import argparse
import csv
import itertools
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from backtest import (
    BACKTEST_TAKER_FEE_RATE,
    apply_adverse_slippage,
    load_ohlcv_csv,
)
from main import LEVERAGE, calc_qty_by_margin, calc_qty_by_risk
from multi_strategy_backtest import (
    FUNDING_RATE_PER_8H,
    MS_PER_YEAR,
    count_funding_periods,
    daily_sharpe,
    gen_hybrid,
    generate_trades,
    max_drawdown_pct,
    profit_factor,
)

TOP10 = ["ONDOUSDT", "ETHUSDT", "BCHUSDT", "TONUSDT", "XLMUSDT",
         "ZECUSDT", "DOGEUSDT", "1000PEPEUSDT", "SUIUSDT", "ADAUSDT"]

RISK = 0.0075
MAX_CONCURRENT = 2
INITIAL = 100_000.0


def daily_returns(k1: List[List[Any]]) -> Dict[str, float]:
    """date(YYYY-MM-DD) -> close-to-close return, from the day's last 1H close."""
    last_close: Dict[str, float] = {}
    for row in k1:
        d = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        last_close[d] = float(row[4])
    days = sorted(last_close)
    rets: Dict[str, float] = {}
    for i in range(1, len(days)):
        prev = last_close[days[i - 1]]
        if prev > 0:
            rets[days[i]] = (last_close[days[i]] - prev) / prev
    return rets


def pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / math.sqrt(va * vb)


def pair_corr(ra: Dict[str, float], rb: Dict[str, float]) -> float:
    common = sorted(set(ra) & set(rb))
    if len(common) < 5:
        return 0.0
    return pearson([ra[d] for d in common], [rb[d] for d in common])


def avg_portfolio_corr(symbols: Tuple[str, ...], corr_mat: Dict[Tuple[str, str], float]) -> float:
    pairs = list(itertools.combinations(symbols, 2))
    if not pairs:
        return 0.0
    return sum(corr_mat[tuple(sorted(p))] for p in pairs) / len(pairs)


def simulate_portfolio(symbols: Tuple[str, ...],
                       raw_by_symbol: Dict[str, List[Tuple]]) -> Dict[str, Any]:
    # merge candidate entries from all pairs, chronological by entry time
    candidates: List[Tuple] = []
    for sym in symbols:
        for (entry_ms, exit_ms, side, entry, stop, exit_price) in raw_by_symbol[sym]:
            candidates.append((entry_ms, exit_ms, side, entry, stop, exit_price, sym))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        return {}

    balance = INITIAL
    active: List[Dict[str, Any]] = []  # open positions
    pnls: List[float] = []
    curve = [INITIAL]
    events: List[Tuple[int, float]] = []
    monthly: Dict[str, float] = {}
    longs = 0
    taken = 0

    def close_position(pos: Dict[str, Any]) -> None:
        nonlocal balance, longs
        eff_exit = apply_adverse_slippage(pos["exit_price"], pos["side"], "exit")
        pnl_unit = (eff_exit - pos["eff_entry"]) if pos["side"] == "Buy" else (pos["eff_entry"] - eff_exit)
        fees = pos["qty"] * pos["eff_entry"] * BACKTEST_TAKER_FEE_RATE + pos["qty"] * eff_exit * BACKTEST_TAKER_FEE_RATE
        funding = 0.0
        if pos["side"] == "Sell":
            funding = pos["qty"] * pos["eff_entry"] * FUNDING_RATE_PER_8H * count_funding_periods(pos["entry_ms"], pos["exit_ms"])
        pnl = pos["qty"] * pnl_unit - fees - funding
        balance += pnl
        pnls.append(pnl)
        curve.append(balance)
        events.append((pos["exit_ms"], balance))
        if pos["side"] == "Buy":
            longs += 1
        mkey = datetime.fromtimestamp(pos["entry_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m")
        monthly[mkey] = monthly.get(mkey, 0.0) + pnl

    def flush_until(now_ms: int) -> None:
        # close all active positions whose exit happened on/before now_ms, in exit order
        active.sort(key=lambda p: p["exit_ms"])
        while active and active[0]["exit_ms"] <= now_ms:
            close_position(active.pop(0))

    for entry_ms, exit_ms, side, entry, stop, exit_price, sym in candidates:
        flush_until(entry_ms)
        if len(active) >= MAX_CONCURRENT:
            continue  # global concurrency cap -> skip signal
        qty = min(
            calc_qty_by_risk(balance, RISK, entry, stop),
            calc_qty_by_margin(balance, LEVERAGE, entry),
        )
        if qty <= 0:
            continue
        active.append({
            "exit_ms": exit_ms, "entry_ms": entry_ms, "symbol": sym, "side": side,
            "eff_entry": apply_adverse_slippage(entry, side, "entry"),
            "exit_price": exit_price, "qty": qty,
        })
        taken += 1

    # close any remaining open positions
    active.sort(key=lambda p: p["exit_ms"])
    for pos in active:
        close_position(pos)

    n = len(pnls)
    if n == 0:
        return {}
    # order equity events chronologically for DD / sharpe
    events.sort(key=lambda e: e[0])
    eq_curve = [INITIAL]
    for _t, bal in events:
        eq_curve.append(bal)

    first_entry = candidates[0][0]
    last_exit = max(e[0] for e in events)
    years = (last_exit - first_entry) / MS_PER_YEAR
    total_return = balance / INITIAL
    annual = (total_return ** (1.0 / years) - 1.0) * 100.0 if (years > 0 and total_return > 0) else -100.0
    wins = sum(1 for p in pnls if p > 0)
    pf = profit_factor(pnls)
    best = max(monthly.items(), key=lambda kv: kv[1])
    worst = min(monthly.items(), key=lambda kv: kv[1])
    return {
        "trades": n,
        "signals_skipped": len(candidates) - taken,
        "annual_return_pct": round(annual, 2),
        "max_dd_pct": round(max_drawdown_pct(eq_curve), 2),
        "winrate_pct": round(wins / n * 100.0, 2),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else float("inf"),
        "sharpe": round(daily_sharpe(events, first_entry, last_exit, INITIAL), 3),
        "years": round(years, 2),
        "final_balance": round(balance, 2),
        "long_pct": round(longs / n * 100.0, 1),
        "short_pct": round((n - longs) / n * 100.0, 1),
        "best_month": best[0], "best_month_pnl": round(best[1], 2),
        "worst_month": worst[0], "worst_month_pnl": round(worst[1], 2),
    }


CSV_FIELDS = [
    "pairs", "n_pairs", "annual_return_pct", "max_dd_pct", "profit_factor", "sharpe",
    "winrate_pct", "trades", "signals_skipped", "avg_corr", "final_balance",
    "long_pct", "short_pct", "best_month", "best_month_pnl", "worst_month", "worst_month_pnl",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--ohlcv-dir", default=os.path.join(base, "ohlcv"))
    parser.add_argument("--out", default=os.path.join(base, "portfolio_backtest_results.csv"))
    parser.add_argument("--max-dd", type=float, default=8.0)
    args = parser.parse_args()

    raw_by_symbol: Dict[str, List[Tuple]] = {}
    rets_by_symbol: Dict[str, Dict[str, float]] = {}
    for sym in TOP10:
        k1 = load_ohlcv_csv(os.path.join(args.ohlcv_dir, f"{sym}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(args.ohlcv_dir, f"{sym}_240.csv"))
        raw_by_symbol[sym] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        rets_by_symbol[sym] = daily_returns(k1)
        print(f"loaded {sym}: {len(raw_by_symbol[sym])} hybrid trades")

    # correlation matrix
    corr_mat: Dict[Tuple[str, str], float] = {}
    for a, b in itertools.combinations(TOP10, 2):
        corr_mat[tuple(sorted((a, b)))] = round(pair_corr(rets_by_symbol[a], rets_by_symbol[b]), 4)

    combos: List[Tuple[str, ...]] = []
    for r in (3, 4):
        combos.extend(itertools.combinations(TOP10, r))

    results: List[Dict[str, Any]] = []
    for combo in combos:
        m = simulate_portfolio(combo, raw_by_symbol)
        if not m:
            continue
        m["pairs"] = "+".join(combo)
        m["n_pairs"] = len(combo)
        m["avg_corr"] = round(avg_portfolio_corr(combo, corr_mat), 4)
        results.append(m)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            if isinstance(row["profit_factor"], float) and math.isinf(row["profit_factor"]):
                row["profit_factor"] = "inf"
            w.writerow(row)

    # correlation matrix print
    print("\n=== Daily-return correlation matrix (top-10 candidates) ===")
    hdr = "        " + "".join(f"{s[:6]:>8}" for s in TOP10)
    print(hdr)
    for a in TOP10:
        line = f"{a[:7]:<8}"
        for b in TOP10:
            if a == b:
                line += f"{'1.00':>8}"
            else:
                line += f"{corr_mat[tuple(sorted((a, b)))]:>8.2f}"
        print(line)

    def show(rows: List[Dict[str, Any]], title: str) -> None:
        print(f"\n=== {title} ===")
        h = (f"{'pairs':<46}{'annual%':>9}{'DD%':>7}{'PF':>7}{'Sharpe':>8}"
             f"{'win%':>7}{'trades':>7}{'corr':>7}")
        print(h)
        print("-" * len(h))
        for r in rows:
            pf = "inf" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
            print(f"{r['pairs']:<46}{r['annual_return_pct']:>9.2f}{r['max_dd_pct']:>7.2f}"
                  f"{pf:>7}{r['sharpe']:>8.2f}{r['winrate_pct']:>7.1f}{r['trades']:>7}{r['avg_corr']:>7.2f}")

    within = [r for r in results if r["max_dd_pct"] < args.max_dd]
    within.sort(key=lambda x: x["annual_return_pct"], reverse=True)
    print(f"\nPortfolios tested: {len(results)} | with DD<{args.max_dd}%: {len(within)}")
    show(within[:5], f"TOP-5 by annual % (DD < {args.max_dd}%)")

    # diversification-aware view: among DD<8% with annual in top tier, lowest correlation
    if within:
        top_tier = within[:15]
        by_div = sorted(top_tier, key=lambda x: x["avg_corr"])
        show(by_div[:5], f"Best-diversified among DD<{args.max_dd}% top-15 (lowest avg correlation)")

    print(f"\nFull results saved to {args.out} ({len(results)} portfolios)")


if __name__ == "__main__":
    main()
