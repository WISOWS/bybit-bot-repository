"""Backtest the live hybrid regime-switch strategy across all OHLCV pairs.

Reuses the bot's own strategy logic (research_strategies.regime_switch_hybrid),
exit simulation, slippage, taker fees and position sizing. Adds:
  - funding cost (-0.01% per 8h on short positions)
  - annualized return (CAGR) over each pair's actual data span
  - annualized Sharpe (from a daily equity curve)

Parameters (per user spec):
  leverage 5x, risk 0.25%/trade, initial $100,000,
  taker fee 0.055% per side, funding -0.01% / 8h on shorts.
"""

import argparse
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backtest import (
    BACKTEST_SLIPPAGE_RATE,
    BACKTEST_TAKER_FEE_RATE,
    apply_adverse_slippage,
    load_ohlcv_csv,
    simulate_exit,
)
from main import LEVERAGE, RISK_PER_TRADE, calc_qty_by_margin, calc_qty_by_risk
from research_strategies import regime_switch_hybrid

FUNDING_RATE_PER_8H = 0.0001  # 0.01% per 8h, charged to short positions
FUNDING_INTERVAL_MS = 8 * 3600 * 1000
MS_PER_YEAR = 365.0 * 24 * 3600 * 1000


def count_funding_periods(entry_ms: int, exit_ms: int) -> int:
    """Number of 8h funding stamps (00:00/08:00/16:00 UTC) strictly after entry, up to and incl. exit."""
    if exit_ms <= entry_ms:
        return 0
    first = (entry_ms // FUNDING_INTERVAL_MS) + 1
    last = exit_ms // FUNDING_INTERVAL_MS
    return max(0, int(last - first + 1))


def profit_factor(pnls: List[float]) -> float:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown_pct(balance_curve: List[float]) -> float:
    peak = balance_curve[0] if balance_curve else 0.0
    dd = 0.0
    for v in balance_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = min(dd, (v - peak) / peak)
    return abs(dd) * 100.0


def daily_sharpe(equity_events: List[Tuple[int, float]], start_ms: int, end_ms: int,
                 initial_balance: float) -> float:
    """Annualized Sharpe from a daily-sampled equity curve (rf=0).

    equity_events: list of (exit_time_ms, balance_after) sorted by time.
    """
    if end_ms <= start_ms or not equity_events:
        return 0.0
    day_ms = 24 * 3600 * 1000
    n_days = int((end_ms - start_ms) // day_ms)
    if n_days < 2:
        return 0.0
    # balance as of the end of each day boundary
    balances: List[float] = []
    ei = 0
    cur = initial_balance
    for d in range(n_days + 1):
        t = start_ms + d * day_ms
        while ei < len(equity_events) and equity_events[ei][0] <= t:
            cur = equity_events[ei][1]
            ei += 1
        balances.append(cur)
    returns: List[float] = []
    for i in range(1, len(balances)):
        prev = balances[i - 1]
        if prev > 0:
            returns.append((balances[i] - prev) / prev)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std <= 0:
        return 0.0
    return (mean / std) * math.sqrt(365.0)


def backtest_symbol(symbol: str, file_1h: str, file_4h: str,
                    initial_balance: float, risk_per_trade: float) -> Optional[Dict[str, Any]]:
    klines_1h = load_ohlcv_csv(file_1h)
    klines_4h = load_ohlcv_csv(file_4h)
    if len(klines_1h) < 50 or len(klines_4h) < 50:
        return None

    balance = initial_balance
    pnls: List[float] = []
    balance_curve: List[float] = [initial_balance]
    equity_events: List[Tuple[int, float]] = []
    trade_log: List[Dict[str, Any]] = []

    times_4h = [int(r[0]) for r in klines_4h]
    idx_4h = -1
    idx_1h = 1

    while idx_1h < len(klines_1h) - 1:
        current_time_ms = int(klines_1h[idx_1h][0])
        while idx_4h + 1 < len(times_4h) and times_4h[idx_4h + 1] <= current_time_ms:
            idx_4h += 1
        if idx_4h < 0:
            idx_1h += 1
            continue

        setup = regime_switch_hybrid(klines_1h, idx_1h, klines_4h, idx_4h)
        if setup is None:
            idx_1h += 1
            continue

        risk_per_unit = abs(setup.entry - setup.stop)
        if risk_per_unit <= 0:
            idx_1h += 1
            continue

        qty = min(
            calc_qty_by_risk(balance, risk_per_trade, setup.entry, setup.stop),
            calc_qty_by_margin(balance, LEVERAGE, setup.entry),
        )
        if qty <= 0:
            idx_1h += 1
            continue

        eff_entry = apply_adverse_slippage(setup.entry, setup.side, "entry")
        exit_price, _reason, exit_index = simulate_exit(
            klines_1h, idx_1h, setup.side, setup.stop, setup.tp
        )
        eff_exit = apply_adverse_slippage(exit_price, setup.side, "exit")
        exit_time_ms = int(klines_1h[exit_index][0])

        pnl_per_unit = (eff_exit - eff_entry) if setup.side == "Buy" else (eff_entry - eff_exit)
        entry_fee = qty * eff_entry * BACKTEST_TAKER_FEE_RATE
        exit_fee = qty * eff_exit * BACKTEST_TAKER_FEE_RATE

        funding_cost = 0.0
        if setup.side == "Sell":
            periods = count_funding_periods(current_time_ms, exit_time_ms)
            funding_cost = qty * eff_entry * FUNDING_RATE_PER_8H * periods

        realized_pnl = qty * pnl_per_unit - entry_fee - exit_fee - funding_cost
        balance += realized_pnl
        pnls.append(realized_pnl)
        balance_curve.append(balance)
        equity_events.append((exit_time_ms, balance))
        trade_log.append({"side": setup.side, "entry_ms": current_time_ms, "pnl": realized_pnl})

        idx_1h = exit_index + 1

    n = len(pnls)
    if n == 0:
        return {
            "symbol": symbol, "trades": 0, "annual_return_pct": 0.0,
            "max_dd_pct": 0.0, "winrate_pct": 0.0, "profit_factor": 0.0,
            "sharpe": 0.0, "final_balance": round(balance, 2), "years": 0.0,
            "trade_log": [],
        }

    start_ms = int(klines_1h[0][0])
    end_ms = int(klines_1h[-1][0])
    years = (end_ms - start_ms) / MS_PER_YEAR
    total_return = balance / initial_balance
    if years > 0 and total_return > 0:
        annual_return_pct = (total_return ** (1.0 / years) - 1.0) * 100.0
    else:
        annual_return_pct = 0.0

    wins = sum(1 for p in pnls if p > 0)
    pf = profit_factor(pnls)

    return {
        "symbol": symbol,
        "trades": n,
        "annual_return_pct": round(annual_return_pct, 2),
        "max_dd_pct": round(max_drawdown_pct(balance_curve), 2),
        "winrate_pct": round(wins / n * 100.0, 2),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else float("inf"),
        "sharpe": round(daily_sharpe(equity_events, start_ms, end_ms, initial_balance), 3),
        "final_balance": round(balance, 2),
        "years": round(years, 2),
        "trade_log": trade_log,
    }


def trade_breakdown(res: Dict[str, Any]) -> Dict[str, Any]:
    log = res["trade_log"]
    n = len(log)
    longs = sum(1 for t in log if t["side"] == "Buy")
    shorts = n - longs
    months = max(res["years"] * 12.0, 1e-9)
    monthly: Dict[str, float] = {}
    for t in log:
        key = datetime.fromtimestamp(t["entry_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0.0) + t["pnl"]
    best = max(monthly.items(), key=lambda kv: kv[1]) if monthly else ("-", 0.0)
    worst = min(monthly.items(), key=lambda kv: kv[1]) if monthly else ("-", 0.0)
    return {
        "trades_per_month": round(n / months, 2),
        "long_pct": round(longs / n * 100.0, 1) if n else 0.0,
        "short_pct": round(shorts / n * 100.0, 1) if n else 0.0,
        "best_month": best,
        "worst_month": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ohlcv"))
    parser.add_argument("--initial-balance", type=float, default=100_000.0)
    parser.add_argument("--risk", type=float, default=RISK_PER_TRADE)
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--min-annual-return", type=float, default=50.0)
    parser.add_argument("--max-dd", type=float, default=15.0)
    parser.add_argument("--min-trades", type=int, default=0)
    parser.add_argument("--min-years", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if args.symbols == "all":
        files = os.listdir(args.ohlcv_dir)
        symbols = sorted({f[:-7] for f in files if f.endswith("_60.csv")})
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    results: List[Dict[str, Any]] = []
    for symbol in symbols:
        f1 = os.path.join(args.ohlcv_dir, f"{symbol}_60.csv")
        f4 = os.path.join(args.ohlcv_dir, f"{symbol}_240.csv")
        if not (os.path.exists(f1) and os.path.exists(f4)):
            print(f"SKIP {symbol}: missing ohlcv")
            continue
        try:
            res = backtest_symbol(symbol, f1, f4, args.initial_balance, args.risk)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {symbol}: {exc}")
            continue
        if res is None or res["trades"] == 0:
            continue
        results.append(res)

    header = f"{'symbol':<16}{'trades':>7}{'annual%':>10}{'maxDD%':>9}{'win%':>8}{'PF':>9}{'Sharpe':>9}{'years':>7}"
    print(f"\n=== SELECTED PAIRS (regime_switch_hybrid, 1H+4H, risk={args.risk*100:.2f}%/trade) ===")
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: (x["profit_factor"] if math.isfinite(x["profit_factor"]) else 1e9), reverse=True):
        pf = "inf" if not math.isfinite(r["profit_factor"]) else f"{r['profit_factor']:.3f}"
        print(f"{r['symbol']:<16}{r['trades']:>7}{r['annual_return_pct']:>10.2f}{r['max_dd_pct']:>9.2f}"
              f"{r['winrate_pct']:>8.2f}{pf:>9}{r['sharpe']:>9.3f}{r['years']:>7.2f}")

    filtered = [
        r for r in results
        if r["annual_return_pct"] > args.min_annual_return
        and r["max_dd_pct"] < args.max_dd
        and r["trades"] > args.min_trades
        and r["years"] > args.min_years
    ]
    filtered.sort(key=lambda x: (x["profit_factor"] if math.isfinite(x["profit_factor"]) else 1e9), reverse=True)
    top = filtered[: args.top]

    print(f"\n=== TOP {args.top} by Profit Factor ===")
    print(f"filters: annual>{args.min_annual_return}%, maxDD<{args.max_dd}%, "
          f"trades>{args.min_trades}, span>{args.min_years}y  -> matched {len(filtered)}/{len(results)}")
    print(header)
    print("-" * len(header))
    for r in top:
        pf = "inf" if not math.isfinite(r["profit_factor"]) else f"{r['profit_factor']:.3f}"
        print(f"{r['symbol']:<16}{r['trades']:>7}{r['annual_return_pct']:>10.2f}{r['max_dd_pct']:>9.2f}"
              f"{r['winrate_pct']:>8.2f}{pf:>9}{r['sharpe']:>9.3f}{r['years']:>7.2f}")

    if top:
        print("\n=== TOP-5 TRADE BREAKDOWN ===")
        for r in top:
            b = trade_breakdown(r)
            bm, bv = b["best_month"]
            wm, wv = b["worst_month"]
            print(f"\n{r['symbol']}  ({r['trades']} trades, {r['years']}y)")
            print(f"  trades/month (avg): {b['trades_per_month']}")
            print(f"  Long vs Short:      {b['long_pct']}% / {b['short_pct']}%")
            print(f"  best month:         {bm}  {bv:+,.2f} USDT")
            print(f"  worst month:        {wm}  {wv:+,.2f} USDT")


if __name__ == "__main__":
    main()
