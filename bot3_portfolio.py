"""Bot #3 search: ema_crossover portfolios (3-4 pairs) with MTM drawdown.

Candidates: top-10 ema_crossover pairs by PF (excluding bot #1/#2 pairs).
Tests every 3- and 4-pair combo at risk 0.75% and 1.0%, max_concurrent 3.
Goal: annual > 100%, MTM DD < 10%, span > 1y.

MTM is computed per 1H bar with an event-driven open-position tracker
(O(bars + trades)) so all ~660 portfolios are feasible.
"""

import csv
import itertools
import math
import os
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import BACKTEST_TAKER_FEE_RATE, apply_adverse_slippage, load_ohlcv_csv
from multi_strategy_backtest import (
    FUNDING_RATE_PER_8H,
    count_funding_periods,
    generate_trades,
    make_ema_crossover,
)

BASE = os.path.dirname(os.path.abspath(__file__))
TOP10 = ["1000PEPEUSDT", "MUSDT", "TONUSDT", "DYDXUSDT", "RUNEUSDT",
         "XRPUSDT", "BOMEUSDT", "ENSUSDT", "WLDUSDT", "XLMUSDT"]
RISKS = [0.0075, 0.01]
MAX_CONCURRENT = 3
INITIAL = P.INITIAL


def mtm_dd_stats(executed: List[Dict[str, Any]], close_maps: Dict[str, Dict[int, float]],
                 symbols: Tuple[str, ...]) -> Dict[str, Any]:
    """Event-driven MTM equity curve -> dd_stats. O(bars + trades)."""
    if not executed:
        return {"max_dd_pct": 0.0}
    ts = set()
    for s in symbols:
        ts.update(close_maps[s].keys())
    fe = min(p["entry_ms"] for p in executed)
    lx = max(p["exit_ms"] for p in executed)
    grid = sorted(t for t in ts if fe <= t <= lx)

    for i, p in enumerate(executed):
        p["_k"] = i
    by_entry = sorted(executed, key=lambda p: p["entry_ms"])
    by_exit = sorted(executed, key=lambda p: p["exit_ms"])
    ei = xi = 0
    realized = INITIAL
    active: Dict[int, Dict[str, Any]] = {}
    last_close = {s: None for s in symbols}
    curve: List[Tuple[int, float]] = []

    for t in grid:
        for s in symbols:
            c = close_maps[s].get(t)
            if c is not None:
                last_close[s] = c
        while ei < len(by_entry) and by_entry[ei]["entry_ms"] <= t:
            p = by_entry[ei]
            active[p["_k"]] = p
            ei += 1
        while xi < len(by_exit) and by_exit[xi]["exit_ms"] <= t:
            p = by_exit[xi]
            realized += p["pnl"]
            active.pop(p["_k"], None)
            xi += 1
        unreal = 0.0
        for p in active.values():
            mark = last_close[p["symbol"]]
            if mark is None:
                continue
            eff_mark = apply_adverse_slippage(mark, p["side"], "exit")
            gross = p["qty"] * (eff_mark - p["eff_entry"]) if p["side"] == "Buy" else p["qty"] * (p["eff_entry"] - eff_mark)
            fees = p["qty"] * p["eff_entry"] * BACKTEST_TAKER_FEE_RATE + p["qty"] * eff_mark * BACKTEST_TAKER_FEE_RATE
            funding = p["qty"] * p["eff_entry"] * FUNDING_RATE_PER_8H * count_funding_periods(p["entry_ms"], t) if p["side"] == "Sell" else 0.0
            unreal += gross - fees - funding
        curve.append((t, realized + unreal))
    return P.dd_stats(curve)


def main() -> None:
    raw_by_symbol: Dict[str, List[Tuple]] = {}
    close_maps: Dict[str, Dict[int, float]] = {}
    ema = make_ema_crossover()
    for s in TOP10:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw_by_symbol[s] = generate_trades(ema, k1, k4, min_4h=50)
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1}
        print(f"  {s}: {len(raw_by_symbol[s])} ema_crossover trades")

    combos: List[Tuple[str, ...]] = []
    for r in (3, 4):
        combos.extend(itertools.combinations(TOP10, r))

    results: List[Dict[str, Any]] = []
    for combo in combos:
        for risk in RISKS:
            P.SYMBOLS = list(combo)
            P.RISK = risk
            executed, m = P.run_portfolio(raw_by_symbol, max_concurrent=MAX_CONCURRENT)
            if not m:
                continue
            dd = mtm_dd_stats(executed, close_maps, combo)
            results.append({
                "pairs": "+".join(combo), "n_pairs": len(combo), "risk_pct": round(risk * 100, 2),
                "annual_return_pct": m["annual_return_pct"], "mtm_dd_pct": dd["max_dd_pct"],
                "realized_dd_pct": m["realized_dd_pct"], "profit_factor": m["profit_factor"],
                "sharpe": m["sharpe"], "winrate_pct": m["winrate_pct"], "trades": m["trades"],
                "years": m["years"], "final_balance": m["final_balance"],
            })

    fields = ["pairs", "n_pairs", "risk_pct", "annual_return_pct", "mtm_dd_pct", "realized_dd_pct",
              "profit_factor", "sharpe", "winrate_pct", "trades", "years", "final_balance"]
    out = os.path.join(BASE, "bot3_portfolio_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            if isinstance(row["profit_factor"], float) and math.isinf(row["profit_factor"]):
                row["profit_factor"] = "inf"
            w.writerow(row)

    def pfx(r):
        return "inf" if (isinstance(r["profit_factor"], float) and math.isinf(r["profit_factor"])) else f"{r['profit_factor']:.2f}"

    passing = [r for r in results if r["annual_return_pct"] > 100 and r["mtm_dd_pct"] < 10 and r["years"] > 1]
    passing.sort(key=lambda r: r["annual_return_pct"], reverse=True)

    hdr = f"{'pairs':<48}{'risk':>5}{'annual%':>9}{'mtmDD%':>8}{'realDD%':>8}{'PF':>6}{'Shp':>6}{'win%':>6}{'tr':>5}{'yrs':>5}"
    print(f"\n{'='*len(hdr)}\nBOT #3 — ema_crossover portfolios | max_concurrent {MAX_CONCURRENT} | {len(results)} portfolios tested")
    print("=" * len(hdr))

    def show(rows):
        print(hdr); print("-" * len(hdr))
        for r in rows:
            print(f"{r['pairs']:<48}{r['risk_pct']:>5.2f}{r['annual_return_pct']:>9.1f}{r['mtm_dd_pct']:>8.2f}"
                  f"{r['realized_dd_pct']:>8.2f}{pfx(r):>6}{r['sharpe']:>6.2f}{r['winrate_pct']:>6.1f}{r['trades']:>5}{r['years']:>5.2f}")

    if passing:
        print(f"\n>>> ЦЕЛЬ ДОСТИГНУТА: {len(passing)} портфелей с annual>100% И MTM DD<10% И span>1y")
        show(passing[:10])
        best = passing[0]
    else:
        print("\n>>> Цель (annual>100% при MTM DD<10%) НЕ достигнута.")
        near = [r for r in results if r["mtm_dd_pct"] < 10 and r["years"] > 1]
        near.sort(key=lambda r: r["annual_return_pct"], reverse=True)
        print(f"Топ-5 по annual% при MTM DD<10%, span>1y (из {len(near)} удовлетворяющих DD<10%):")
        show(near[:5])
        best = near[0] if near else None

    # walk-forward for best
    if best:
        combo = tuple(best["pairs"].split("+"))
        risk = best["risk_pct"] / 100.0
        print(f"\n{'='*len(hdr)}\nWALK-FORWARD лучшего портфеля: {best['pairs']} @ risk {best['risk_pct']}%")
        print("=" * len(hdr))
        windows = [("IN-SAMPLE (->2024-12)", P.to_ms("2020-01-01"), P.to_ms("2025-01-01")),
                   ("OUT-OF-SAMPLE (2025-01->)", P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))]
        for label, ws, we in windows:
            P.SYMBOLS = list(combo)
            P.RISK = risk
            ex, mm = P.run_portfolio(raw_by_symbol, window=(ws, we), max_concurrent=MAX_CONCURRENT)
            if not ex:
                print(f"  {label}: нет сделок")
                continue
            dd = mtm_dd_stats(ex, close_maps, combo)
            print(f"  {label}: annual={mm['annual_return_pct']:.1f}% MTM_DD={dd['max_dd_pct']:.2f}% "
                  f"PF={pfx(mm)} win={mm['winrate_pct']:.1f}% trades={mm['trades']} span={mm['years']}y "
                  f"final=${mm['final_balance']:,.0f}")

    print(f"\nРезультаты сохранены -> {out} ({len(results)} строк)")


if __name__ == "__main__":
    main()
