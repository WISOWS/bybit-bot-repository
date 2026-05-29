"""Mark-to-market drawdown + walk-forward for the ONDO+ZEC+SUI hybrid portfolio.

MTM equity is evaluated on every 1H bar: realized cash + unrealized PnL of open
positions (marked at the bar close, net of entry/exit fees and accrued short
funding, so the curve is continuous with the realized balance at each close).

Account: $100k, leverage 5x, risk 0.75%/trade, max 2 concurrent positions,
taker fee 0.055%/side, slippage 0.05%/side, funding -0.01%/8h on shorts.
"""

import csv
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backtest import BACKTEST_TAKER_FEE_RATE, apply_adverse_slippage, load_ohlcv_csv
from main import LEVERAGE, calc_qty_by_margin, calc_qty_by_risk
from multi_strategy_backtest import (
    FUNDING_RATE_PER_8H,
    MS_PER_YEAR,
    count_funding_periods,
    daily_sharpe,
    gen_hybrid,
    generate_trades,
    profit_factor,
)

SYMBOLS = ["ONDOUSDT", "ZECUSDT", "SUIUSDT"]
RISK = 0.0075
MAX_CONCURRENT = 2
INITIAL = 100_000.0
HOUR_MS = 3600 * 1000
BASE = os.path.dirname(os.path.abspath(__file__))


def to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def run_portfolio(raw_by_symbol: Dict[str, List[Tuple]],
                  window: Optional[Tuple[int, int]] = None,
                  max_concurrent: int = MAX_CONCURRENT) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates: List[Tuple] = []
    for sym in SYMBOLS:
        for (entry_ms, exit_ms, side, entry, stop, exit_price) in raw_by_symbol[sym]:
            if window and not (window[0] <= entry_ms < window[1]):
                continue
            candidates.append((entry_ms, exit_ms, side, entry, stop, exit_price, sym))
    candidates.sort(key=lambda x: x[0])

    balance = INITIAL
    active: List[Dict[str, Any]] = []
    executed: List[Dict[str, Any]] = []
    pnls: List[float] = []
    realized_curve = [INITIAL]
    events: List[Tuple[int, float]] = []
    limit_skips = 0
    qty_skips = 0

    def close_pos(pos: Dict[str, Any]) -> None:
        nonlocal balance
        eff_exit = apply_adverse_slippage(pos["exit_price"], pos["side"], "exit")
        unit = (eff_exit - pos["eff_entry"]) if pos["side"] == "Buy" else (pos["eff_entry"] - eff_exit)
        fees = pos["qty"] * pos["eff_entry"] * BACKTEST_TAKER_FEE_RATE + pos["qty"] * eff_exit * BACKTEST_TAKER_FEE_RATE
        funding = pos["qty"] * pos["eff_entry"] * FUNDING_RATE_PER_8H * count_funding_periods(pos["entry_ms"], pos["exit_ms"]) if pos["side"] == "Sell" else 0.0
        pnl = pos["qty"] * unit - fees - funding
        balance += pnl
        pnls.append(pnl)
        realized_curve.append(balance)
        events.append((pos["exit_ms"], balance))
        pos["pnl"] = pnl
        executed.append(pos)

    def flush(now_ms: int) -> None:
        active.sort(key=lambda p: p["exit_ms"])
        while active and active[0]["exit_ms"] <= now_ms:
            close_pos(active.pop(0))

    for entry_ms, exit_ms, side, entry, stop, exit_price, sym in candidates:
        flush(entry_ms)
        if len(active) >= max_concurrent:
            limit_skips += 1
            continue
        qty = min(calc_qty_by_risk(balance, RISK, entry, stop), calc_qty_by_margin(balance, LEVERAGE, entry))
        if qty <= 0:
            qty_skips += 1
            continue
        active.append({"entry_ms": entry_ms, "exit_ms": exit_ms, "symbol": sym, "side": side,
                       "eff_entry": apply_adverse_slippage(entry, side, "entry"),
                       "exit_price": exit_price, "qty": qty})
    active.sort(key=lambda p: p["exit_ms"])
    for pos in active:
        close_pos(pos)

    n = len(pnls)
    if n == 0:
        return [], {}
    events.sort(key=lambda e: e[0])
    first_entry = min(p["entry_ms"] for p in executed)
    last_exit = max(p["exit_ms"] for p in executed)
    years = (last_exit - first_entry) / MS_PER_YEAR
    tot = balance / INITIAL
    annual = (tot ** (1.0 / years) - 1.0) * 100.0 if (years > 0 and tot > 0) else -100.0
    rc_peak, rdd = realized_curve[0], 0.0
    for v in realized_curve:
        rc_peak = max(rc_peak, v)
        rdd = min(rdd, (v - rc_peak) / rc_peak)
    pf = profit_factor(pnls)
    metrics = {
        "trades": n, "annual_return_pct": round(annual, 2),
        "realized_dd_pct": round(abs(rdd) * 100, 2),
        "winrate_pct": round(sum(1 for p in pnls if p > 0) / n * 100, 2),
        "profit_factor": round(pf, 3) if math.isfinite(pf) else float("inf"),
        "sharpe": round(daily_sharpe(events, first_entry, last_exit, INITIAL), 3),
        "years": round(years, 2), "final_balance": round(balance, 2),
        "total_signals": len(candidates), "limit_skips": limit_skips, "qty_skips": qty_skips,
        "skip_pct": round(limit_skips / len(candidates) * 100, 2) if candidates else 0.0,
    }
    return executed, metrics


def build_mtm_curve(executed: List[Dict[str, Any]],
                    close_maps: Dict[str, Dict[int, float]],
                    ts_grid: List[int]) -> List[Tuple[int, float]]:
    exits_sorted = sorted(executed, key=lambda p: p["exit_ms"])
    realized = INITIAL
    ei = 0
    last_close = {s: None for s in SYMBOLS}
    curve: List[Tuple[int, float]] = []
    for t in ts_grid:
        for s in SYMBOLS:
            c = close_maps[s].get(t)
            if c is not None:
                last_close[s] = c
        while ei < len(exits_sorted) and exits_sorted[ei]["exit_ms"] <= t:
            realized += exits_sorted[ei]["pnl"]
            ei += 1
        unreal = 0.0
        for pos in executed:
            if pos["entry_ms"] <= t < pos["exit_ms"]:
                mark = last_close[pos["symbol"]]
                if mark is None:
                    continue
                eff_mark = apply_adverse_slippage(mark, pos["side"], "exit")
                gross = pos["qty"] * (eff_mark - pos["eff_entry"]) if pos["side"] == "Buy" else pos["qty"] * (pos["eff_entry"] - eff_mark)
                fees = pos["qty"] * pos["eff_entry"] * BACKTEST_TAKER_FEE_RATE + pos["qty"] * eff_mark * BACKTEST_TAKER_FEE_RATE
                funding = pos["qty"] * pos["eff_entry"] * FUNDING_RATE_PER_8H * count_funding_periods(pos["entry_ms"], t) if pos["side"] == "Sell" else 0.0
                unreal += gross - fees - funding
        curve.append((t, realized + unreal))
    return curve


def dd_stats(curve: List[Tuple[int, float]]) -> Dict[str, Any]:
    peak = curve[0][1]
    peak_t = curve[0][0]
    max_dd = 0.0
    dd_sum = 0.0
    dd_sum_active = 0.0
    active_bars = 0
    longest_ms = 0
    crossings = {5.0: 0, 8.0: 0, 10.0: 0}
    armed = {5.0: True, 8.0: True, 10.0: True}
    for t, eq in curve:
        if eq >= peak:
            peak = eq
            longest_ms = max(longest_ms, t - peak_t)
            peak_t = t
        dd = (eq - peak) / peak if peak > 0 else 0.0
        depth = -dd * 100.0
        max_dd = max(max_dd, depth)
        dd_sum += depth
        if depth > 1e-9:
            dd_sum_active += depth
            active_bars += 1
        for thr in crossings:
            if depth >= thr and armed[thr]:
                crossings[thr] += 1
                armed[thr] = False
            elif depth < thr:
                armed[thr] = True
    longest_ms = max(longest_ms, curve[-1][0] - peak_t)  # still underwater at end
    n = len(curve)
    return {
        "max_dd_pct": round(max_dd, 2),
        "avg_dd_all_pct": round(dd_sum / n, 3),
        "avg_dd_active_pct": round(dd_sum_active / active_bars, 3) if active_bars else 0.0,
        "pct_time_underwater": round(active_bars / n * 100, 1),
        "cross_5": crossings[5.0], "cross_8": crossings[8.0], "cross_10": crossings[10.0],
        "longest_dd_days": round(longest_ms / (24 * HOUR_MS), 1),
    }


def ascii_plot(curve: List[Tuple[int, float]], width: int = 74, height: int = 16) -> str:
    if not curve:
        return "(empty)"
    vals = [v for _, v in curve]
    step = max(1, len(vals) // width)
    sampled = vals[::step][:width]
    lo, hi = min(sampled), max(sampled)
    rng = hi - lo or 1.0
    rows = [[" "] * len(sampled) for _ in range(height)]
    for x, v in enumerate(sampled):
        y = int((v - lo) / rng * (height - 1))
        rows[height - 1 - y][x] = "*"
    out = []
    for r, line in enumerate(rows):
        val = hi - (hi - lo) * r / (height - 1)
        out.append(f"{val/1000:7.0f}k |" + "".join(line))
    out.append(" " * 9 + "+" + "-" * len(sampled))
    t0 = datetime.fromtimestamp(curve[0][0] / 1000, tz=timezone.utc).strftime("%Y-%m")
    t1 = datetime.fromtimestamp(curve[-1][0] / 1000, tz=timezone.utc).strftime("%Y-%m")
    out.append(" " * 10 + t0 + " " * (len(sampled) - len(t0) - len(t1)) + t1)
    return "\n".join(out)


def fmt(m: Dict[str, Any]) -> str:
    pf = "inf" if (isinstance(m["profit_factor"], float) and math.isinf(m["profit_factor"])) else f"{m['profit_factor']:.3f}"
    return (f"trades={m['trades']} annual={m['annual_return_pct']:.2f}% "
            f"realizedDD={m['realized_dd_pct']:.2f}% PF={pf} Sharpe={m['sharpe']:.2f} "
            f"win={m['winrate_pct']:.1f}% span={m['years']}y final=${m['final_balance']:,.0f}")


def main() -> None:
    import argparse
    global RISK
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk", type=float, default=RISK)
    args = parser.parse_args()
    RISK = args.risk

    raw_by_symbol: Dict[str, List[Tuple]] = {}
    close_maps: Dict[str, Dict[int, float]] = {}
    ts_set = set()
    for s in SYMBOLS:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw_by_symbol[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1}
        ts_set.update(close_maps[s].keys())

    # ---- full period ----
    executed, metrics = run_portfolio(raw_by_symbol)
    fe = min(p["entry_ms"] for p in executed)
    lx = max(p["exit_ms"] for p in executed)
    grid = sorted(t for t in ts_set if fe <= t <= lx)
    curve = build_mtm_curve(executed, close_maps, grid)
    stats = dd_stats(curve)

    print("=" * 78)
    print(f"PORTFOLIO: ONDO + ZEC + SUI  |  hybrid, risk {RISK*100:.2f}%, max {MAX_CONCURRENT} concurrent, ${INITIAL:,.0f}")
    print("=" * 78)
    monthly: Dict[str, float] = {}
    for p in executed:
        mk = datetime.fromtimestamp(p["entry_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m")
        monthly[mk] = monthly.get(mk, 0.0) + p["pnl"]
    bm = max(monthly.items(), key=lambda kv: kv[1])
    wm = min(monthly.items(), key=lambda kv: kv[1])
    tpm = metrics["trades"] / (metrics["years"] * 12)
    print("\nFULL PERIOD (realized):", fmt(metrics))
    print(f"  trades/month={tpm:.2f}  best {bm[0]}: +${bm[1]:,.0f}  worst {wm[0]}: ${wm[1]:,.0f}")
    print("\n--- MARK-TO-MARKET DRAWDOWN (per 1H bar, incl. unrealized) ---")
    print(f"  1. Real MAX drawdown (MTM):     {stats['max_dd_pct']:.2f}%   "
          f"(vs realized-only {metrics['realized_dd_pct']:.2f}%)")
    print(f"  2. Real AVG drawdown:           {stats['avg_dd_all_pct']:.3f}%  (over all bars)   |  "
          f"{stats['avg_dd_active_pct']:.3f}% (while underwater)")
    print(f"  3. Times DD exceeded threshold: >5%: {stats['cross_5']}   >8%: {stats['cross_8']}   >10%: {stats['cross_10']}  (distinct episodes)")
    print(f"  4. Longest drawdown duration:   {stats['longest_dd_days']:.1f} days")
    print(f"     (% of time underwater:       {stats['pct_time_underwater']:.1f}%)")
    print("\n  5. Equity curve (MTM):")
    print(ascii_plot(curve))

    # save curve for external plotting
    out_csv = os.path.join(BASE, "portfolio_ondo_zec_sui_equity.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["datetime_utc", "ts_ms", "mtm_equity"])
        for t, v in curve:
            w.writerow([datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"), t, round(v, 2)])
    print(f"\n  Equity points saved -> {out_csv} ({len(curve)} rows)")

    # ---- walk-forward ----
    print("\n" + "=" * 78)
    print("WALK-FORWARD  (regime_switch_hybrid has FIXED params -> split-sample robustness check,")
    print("              not parameter optimization; each window resets to $100k)")
    print("=" * 78)
    windows = [
        ("IN-SAMPLE  2023-01..2024-12", to_ms("2023-01-01"), to_ms("2025-01-01")),
        ("OUT-OF-SAMPLE 2025-01..2026-05", to_ms("2025-01-01"), to_ms("2026-06-01")),
    ]
    for label, ws, we in windows:
        ex, m = run_portfolio(raw_by_symbol, window=(ws, we))
        if not ex:
            print(f"\n{label}: no trades")
            continue
        fe2 = min(p["entry_ms"] for p in ex)
        lx2 = max(p["exit_ms"] for p in ex)
        g2 = sorted(t for t in ts_set if fe2 <= t <= lx2)
        c2 = build_mtm_curve(ex, close_maps, g2)
        s2 = dd_stats(c2)
        print(f"\n{label}")
        print(f"  {fmt(m)}")
        print(f"  MTM maxDD={s2['max_dd_pct']:.2f}%  avgDD={s2['avg_dd_all_pct']:.3f}%  "
              f">5%:{s2['cross_5']} >8%:{s2['cross_8']} >10%:{s2['cross_10']}  "
              f"longestDD={s2['longest_dd_days']:.0f}d  underwater={s2['pct_time_underwater']:.0f}%")


if __name__ == "__main__":
    main()
