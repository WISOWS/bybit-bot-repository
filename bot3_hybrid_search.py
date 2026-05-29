"""Bot #3 search: regime_switch_hybrid portfolios from unused pairs.

Candidates (excl. NEAR/SOL/LINK/ENA/ONDO/ZEC/SUI — SUI is taken by bot #2):
  ENS, SNX, ARB, DOT, ATOM, AVAX, APT, INJ, OP, TIA
Tests all 3-pair combos at risk 0.70%, max_concurrent 3, with MTM drawdown.
Filter: annual>80%, MTM DD<10%, span>1y, trades>50. Saves best -> config_bot3.json.
"""

import csv
import itertools
import json
import math
import os
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats  # optimized event-driven MTM
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = ["ENSUSDT", "SNXUSDT", "ARBUSDT", "DOTUSDT", "ATOMUSDT",
              "AVAXUSDT", "APTUSDT", "INJUSDT", "OPUSDT", "TIAUSDT"]
RISK = 0.007
MAX_CONCURRENT = 3


def pfx(r):
    return "inf" if (isinstance(r["profit_factor"], float) and math.isinf(r["profit_factor"])) else f"{r['profit_factor']:.2f}"


def main() -> None:
    raw_by_symbol: Dict[str, List[Tuple]] = {}
    close_maps: Dict[str, Dict[int, float]] = {}
    for s in CANDIDATES:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw_by_symbol[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1}
        print(f"  {s}: {len(raw_by_symbol[s])} hybrid trades")

    results: List[Dict[str, Any]] = []
    for combo in itertools.combinations(CANDIDATES, 3):
        P.SYMBOLS = list(combo)
        P.RISK = RISK
        executed, m = P.run_portfolio(raw_by_symbol, max_concurrent=MAX_CONCURRENT)
        if not m:
            continue
        dd = mtm_dd_stats(executed, close_maps, combo)
        results.append({
            "pairs": "+".join(combo), "risk_pct": round(RISK * 100, 2),
            "annual_return_pct": m["annual_return_pct"], "mtm_dd_pct": dd["max_dd_pct"],
            "realized_dd_pct": m["realized_dd_pct"], "profit_factor": m["profit_factor"],
            "sharpe": m["sharpe"], "winrate_pct": m["winrate_pct"], "trades": m["trades"],
            "years": m["years"], "final_balance": m["final_balance"],
        })

    out = os.path.join(BASE, "bot3_hybrid_results.csv")
    fields = ["pairs", "risk_pct", "annual_return_pct", "mtm_dd_pct", "realized_dd_pct",
              "profit_factor", "sharpe", "winrate_pct", "trades", "years", "final_balance"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            if isinstance(row["profit_factor"], float) and math.isinf(row["profit_factor"]):
                row["profit_factor"] = "inf"
            w.writerow(row)

    passing = [r for r in results
               if r["annual_return_pct"] > 80 and r["mtm_dd_pct"] < 10
               and r["years"] > 1 and r["trades"] > 50]
    passing.sort(key=lambda r: r["annual_return_pct"], reverse=True)

    hdr = f"{'pairs':<40}{'annual%':>9}{'mtmDD%':>8}{'realDD%':>8}{'PF':>6}{'Shp':>6}{'win%':>6}{'tr':>5}{'yrs':>6}"
    print(f"\n{'='*len(hdr)}\nBOT #3 — hybrid portfolios | risk 0.70% | conc 3 | {len(results)} portfolios tested")
    print(f"Filter: annual>80%, MTM DD<10%, span>1y, trades>50  -> matched {len(passing)}")
    print("=" * len(hdr))
    print(hdr); print("-" * len(hdr))

    show = passing if passing else sorted([r for r in results if r["mtm_dd_pct"] < 10], key=lambda r: r["annual_return_pct"], reverse=True)
    for r in show[:5]:
        print(f"{r['pairs']:<40}{r['annual_return_pct']:>9.1f}{r['mtm_dd_pct']:>8.2f}{r['realized_dd_pct']:>8.2f}"
              f"{pfx(r):>6}{r['sharpe']:>6.2f}{r['winrate_pct']:>6.1f}{r['trades']:>5}{r['years']:>6.2f}")
    if not passing:
        print("\n(никто не прошёл annual>80% при DD<10% — показаны лучшие по annual среди DD<10%)")

    best = (passing or show)[0] if (passing or show) else None
    if not best:
        print("\nНет подходящих портфелей."); return

    # walk-forward
    combo = tuple(best["pairs"].split("+"))
    print(f"\n{'='*len(hdr)}\nWALK-FORWARD топ-1: {best['pairs']} @ risk 0.70%")
    print("=" * len(hdr))
    for label, ws, we in [("IN-SAMPLE (->2024-12)", P.to_ms("2020-01-01"), P.to_ms("2025-01-01")),
                          ("OUT-OF-SAMPLE (2025-01->)", P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))]:
        P.SYMBOLS = list(combo)
        P.RISK = RISK
        ex, mm = P.run_portfolio(raw_by_symbol, window=(ws, we), max_concurrent=MAX_CONCURRENT)
        if not ex:
            print(f"  {label}: нет сделок (нет данных в окне)"); continue
        dd = mtm_dd_stats(ex, close_maps, combo)
        print(f"  {label}: annual={mm['annual_return_pct']:.1f}% MTM_DD={dd['max_dd_pct']:.2f}% "
              f"PF={pfx(mm)} win={mm['winrate_pct']:.1f}% trades={mm['trades']} span={mm['years']}y "
              f"final=${mm['final_balance']:,.0f}")

    # save config_bot3.json
    cfg = json.load(open(os.path.join(BASE, "config.json")))
    cfg["risk_per_trade"] = RISK
    cfg["leverage"] = 5
    cfg.pop("symbols", None)
    cfg["max_concurrent"] = MAX_CONCURRENT
    cfg["symbols"] = list(combo)
    json.dump(cfg, open(os.path.join(BASE, "config_bot3.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nЛучший портфель сохранён -> config_bot3.json: {best['pairs']} (risk 0.70%, conc 3)")
    print(f"Все результаты -> {out} ({len(results)} строк)")


if __name__ == "__main__":
    main()
