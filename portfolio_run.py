"""Generic portfolio backtest runner (reuses portfolio_mtm engine).

Usage: python3 portfolio_run.py --symbols NEARUSDT,SOLUSDT,LINKUSDT,ENAUSDT --risk 0.0025 --max-concurrent 2
"""

import argparse
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))


def monthly_breakdown(executed: List[Dict[str, Any]]) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    monthly: Dict[str, float] = {}
    for p in executed:
        mk = datetime.fromtimestamp(p["entry_ms"] / 1000, tz=timezone.utc).strftime("%Y-%m")
        monthly[mk] = monthly.get(mk, 0.0) + p["pnl"]
    return (max(monthly.items(), key=lambda kv: kv[1]), min(monthly.items(), key=lambda kv: kv[1]))


def run(symbols: List[str], risk: float, max_concurrent: int) -> None:
    P.SYMBOLS = symbols
    P.RISK = risk

    raw: Dict[str, List[Tuple]] = {}
    close_maps: Dict[str, Dict[int, float]] = {}
    ts_set = set()
    for s in symbols:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1}
        ts_set.update(close_maps[s].keys())
        print(f"  {s}: {len(raw[s])} hybrid trades")

    def mtm_for(executed):
        fe = min(p["entry_ms"] for p in executed)
        lx = max(p["exit_ms"] for p in executed)
        grid = sorted(t for t in ts_set if fe <= t <= lx)
        return P.dd_stats(P.build_mtm_curve(executed, close_maps, grid))

    print("\n" + "=" * 80)
    print(f"PORTFOLIO: {' + '.join(symbols)}")
    print(f"hybrid | risk {risk*100:.2f}% | max_concurrent {max_concurrent} | $100k | "
          f"fee 0.055% | slip 0.05% | funding -0.01%/8h short")
    print("=" * 80)

    executed, m = P.run_portfolio(raw, max_concurrent=max_concurrent)
    stats = mtm_for(executed)
    bm, wm = monthly_breakdown(executed)
    tpm = m["trades"] / (m["years"] * 12)
    pf = "inf" if (isinstance(m["profit_factor"], float) and math.isinf(m["profit_factor"])) else f"{m['profit_factor']:.3f}"
    print("\n--- FULL PERIOD ---")
    print(f"  Annual return (CAGR):  {m['annual_return_pct']:.2f}%")
    print(f"  MTM Max DD:            {stats['max_dd_pct']:.2f}%   (realized-only {m['realized_dd_pct']:.2f}%)")
    print(f"  Profit factor:         {pf}")
    print(f"  Win rate:              {m['winrate_pct']:.2f}%")
    print(f"  Trades total / month:  {m['trades']} / {tpm:.2f}")
    print(f"  Sharpe:                {m['sharpe']:.2f}")
    print(f"  Best month:            {bm[0]}  +${bm[1]:,.0f}")
    print(f"  Worst month:           {wm[0]}  ${wm[1]:,.0f}")
    print(f"  Span / final:          {m['years']}y / ${m['final_balance']:,.0f}")
    print(f"  Signals / limit-skips: {m['total_signals']} / {m['limit_skips']} ({m['skip_pct']:.2f}%)")

    print("\n--- WALK-FORWARD (each window resets to $100k) ---")
    windows = [
        ("IN-SAMPLE  (... -> 2024-12)", P.to_ms("2020-01-01"), P.to_ms("2025-01-01")),
        ("OUT-OF-SAMPLE (2025-01 -> now)", P.to_ms("2025-01-01"), P.to_ms("2026-06-01")),
    ]
    for label, ws, we in windows:
        ex, mm = P.run_portfolio(raw, window=(ws, we), max_concurrent=max_concurrent)
        if not ex:
            print(f"  {label}: НЕТ сделок (нет данных в этом окне)")
            continue
        s2 = mtm_for(ex)
        pf2 = "inf" if (isinstance(mm["profit_factor"], float) and math.isinf(mm["profit_factor"])) else f"{mm['profit_factor']:.3f}"
        print(f"  {label}:")
        print(f"      annual={mm['annual_return_pct']:.2f}% MTM_DD={s2['max_dd_pct']:.2f}% "
              f"PF={pf2} win={mm['winrate_pct']:.1f}% trades={mm['trades']} span={mm['years']}y "
              f"final=${mm['final_balance']:,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--risk", type=float, default=0.0025)
    ap.add_argument("--max-concurrent", type=int, default=2)
    args = ap.parse_args()
    run([s.strip().upper() for s in args.symbols.split(",") if s.strip()], args.risk, args.max_concurrent)


if __name__ == "__main__":
    main()
