"""How much does the max_concurrent limit cost the ONDO+ZEC+SUI portfolio?

Compares limit=2, limit=3, and unlimited; reports skip counts and MTM metrics.
Risk 0.70%/trade (the bot #2 setting).
"""

import math
import os
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from multi_strategy_backtest import gen_hybrid, generate_trades

P.RISK = 0.007
BASE = os.path.dirname(os.path.abspath(__file__))


def max_overlap(executed: List[Dict[str, Any]]) -> int:
    pts = []
    for p in executed:
        pts.append((p["entry_ms"], 1))
        pts.append((p["exit_ms"], -1))
    pts.sort(key=lambda x: (x[0], x[1]))  # exits (-1) before entries at same ts
    cur = best = 0
    for _t, d in pts:
        cur += d
        best = max(best, cur)
    return best


def main() -> None:
    raw: Dict[str, List[Tuple]] = {}
    close_maps: Dict[str, Dict[int, float]] = {}
    ts_set = set()
    for s in P.SYMBOLS:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1}
        ts_set.update(close_maps[s].keys())

    configs = [("limit=2", 2), ("limit=3", 3), ("unlimited", 10**9)]
    rows = []
    for label, lim in configs:
        ex, m = P.run_portfolio(raw, max_concurrent=lim)
        fe = min(p["entry_ms"] for p in ex)
        lx = max(p["exit_ms"] for p in ex)
        grid = sorted(t for t in ts_set if fe <= t <= lx)
        stats = P.dd_stats(P.build_mtm_curve(ex, close_maps, grid))
        rows.append((label, lim, m, stats, max_overlap(ex)))

    base = rows[0][2]  # limit=2 reference
    print("=" * 92)
    print("max_concurrent impact | ONDO+ZEC+SUI | hybrid, risk 0.70%, $100k")
    print("=" * 92)
    tot = base["total_signals"]
    print(f"\nTotal signals generated across the 3 pairs (chronological): {tot}")
    print(f"  (max physically possible concurrent positions for 3 non-overlapping-per-pair series = {rows[2][4]})")

    print("\n--- SKIP ANALYSIS ---")
    for label, lim, m, _stats, _ov in rows:
        print(f"  {label:<11}: wanted 3rd+ position but blocked = {m['limit_skips']:>4}  "
              f"({m['skip_pct']:.2f}% of {m['total_signals']} signals)   taken={m['trades']}")

    print("\n--- METRICS BY LIMIT ---")
    h = f"{'config':<11}{'taken':>6}{'skipped':>8}{'annual%':>10}{'MTM_DD%':>9}{'realDD%':>9}{'PF':>7}{'Sharpe':>8}{'final$':>13}{'maxConc':>8}"
    print(h)
    print("-" * len(h))
    for label, lim, m, stats, ov in rows:
        pf = "inf" if (isinstance(m["profit_factor"], float) and math.isinf(m["profit_factor"])) else f"{m['profit_factor']:.3f}"
        print(f"{label:<11}{m['trades']:>6}{m['limit_skips']:>8}{m['annual_return_pct']:>10.2f}"
              f"{stats['max_dd_pct']:>9.2f}{m['realized_dd_pct']:>9.2f}{pf:>7}{m['sharpe']:>8.2f}"
              f"{m['final_balance']:>13,.0f}{ov:>8}")

    # deltas vs limit=2
    unl = rows[2][2]
    unl_dd = rows[2][3]["max_dd_pct"]
    print("\n--- IMPACT OF REMOVING THE LIMIT (unlimited vs limit=2) ---")
    print(f"  annual return : {base['annual_return_pct']:.2f}%  ->  {unl['annual_return_pct']:.2f}%  "
          f"({unl['annual_return_pct']-base['annual_return_pct']:+.2f} pp)")
    print(f"  MTM max DD    : {rows[0][3]['max_dd_pct']:.2f}%  ->  {unl_dd:.2f}%  "
          f"({unl_dd-rows[0][3]['max_dd_pct']:+.2f} pp)")
    print(f"  profit factor : {base['profit_factor']}  ->  {unl['profit_factor']}")
    print(f"  final balance : ${base['final_balance']:,.0f}  ->  ${unl['final_balance']:,.0f}")
    print(f"  extra trades  : {unl['trades']-base['trades']:+d}")


if __name__ == "__main__":
    main()
