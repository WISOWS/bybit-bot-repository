"""Bot #3 — hybrid portfolios with REAL walk-forward on 3y data.

Candidates: ENS/SNX/ARB/DOT/ATOM/AVAX/APT/INJ/OP/TIA (now re-downloaded from 2023).
For every 3-pair combo: run IS (2023-06..2024-12) and OOS (2025-01..now) separately,
each with MTM drawdown. Filter on OOS: annual>80%, MTM DD<10%, trades>50.
Rank by OOS annual %. Save top-1 -> config_bot3.json.
Risk 0.70%, max_concurrent 3.
"""

import csv
import itertools
import json
import math
import os
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = ["ENSUSDT", "SNXUSDT", "ARBUSDT", "DOTUSDT", "ATOMUSDT",
              "AVAXUSDT", "APTUSDT", "INJUSDT", "OPUSDT", "TIAUSDT"]
RISK = 0.007
MAX_CONCURRENT = 3
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def window_run(raw, combo, win, close_maps):
    P.SYMBOLS = list(combo)
    P.RISK = RISK
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=MAX_CONCURRENT)
    if not m:
        return None
    dd = mtm_dd_stats(ex, close_maps, combo)
    return {"annual": m["annual_return_pct"], "mtm_dd": dd["max_dd_pct"],
            "real_dd": m["realized_dd_pct"], "pf": m["profit_factor"],
            "sharpe": m["sharpe"], "win": m["winrate_pct"], "trades": m["trades"],
            "years": m["years"], "final": m["final_balance"]}


def main() -> None:
    raw_by_symbol: Dict[str, List[Tuple]] = {}
    close_maps: Dict[str, Dict[int, float]] = {}
    for s in CANDIDATES:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw_by_symbol[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1}
        span = f"{len(k1)} bars"
        print(f"  {s}: {len(raw_by_symbol[s])} hybrid trades ({span})")

    rows: List[Dict[str, Any]] = []
    for combo in itertools.combinations(CANDIDATES, 3):
        oos = window_run(raw_by_symbol, combo, OOS_WIN, close_maps)
        ins = window_run(raw_by_symbol, combo, IS_WIN, close_maps)
        if oos is None:
            continue
        rows.append({"pairs": "+".join(combo), "oos": oos, "is": ins})

    # CSV
    out = os.path.join(BASE, "bot3_walkforward_results.csv")
    fields = ["pairs", "oos_annual", "oos_mtm_dd", "oos_pf", "oos_trades", "oos_years",
              "is_annual", "is_mtm_dd", "is_pf", "is_trades", "is_years"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            o = r["oos"]; i = r["is"]
            w.writerow({
                "pairs": r["pairs"],
                "oos_annual": o["annual"], "oos_mtm_dd": o["mtm_dd"],
                "oos_pf": "inf" if math.isinf(o["pf"]) else round(o["pf"], 4),
                "oos_trades": o["trades"], "oos_years": o["years"],
                "is_annual": i["annual"] if i else "", "is_mtm_dd": i["mtm_dd"] if i else "",
                "is_pf": ("inf" if (i and math.isinf(i["pf"])) else (round(i["pf"], 4) if i else "")),
                "is_trades": i["trades"] if i else 0, "is_years": i["years"] if i else 0,
            })

    passing = [r for r in rows
               if r["oos"]["annual"] > 80 and r["oos"]["mtm_dd"] < 10 and r["oos"]["trades"] > 50]
    passing.sort(key=lambda r: r["oos"]["annual"], reverse=True)

    hdr = (f"{'pairs':<40}{'OOS_an%':>9}{'OOS_DD%':>9}{'OOS_PF':>7}{'OOS_tr':>7}"
           f"{'IS_an%':>9}{'IS_DD%':>8}{'IS_PF':>7}{'IS_tr':>6}")
    print(f"\n{'='*len(hdr)}")
    print(f"BOT #3 WALK-FORWARD | hybrid | risk 0.70% | conc 3 | {len(rows)} combos")
    print(f"Filter: OOS annual>80%, OOS MTM DD<10%, OOS trades>50 -> matched {len(passing)}")
    print("=" * len(hdr))
    print(hdr); print("-" * len(hdr))

    def line(r):
        o = r["oos"]; i = r["is"]
        is_an = f"{i['annual']:.1f}" if i else "n/a"
        is_dd = f"{i['mtm_dd']:.2f}" if i else "n/a"
        is_pf = pfx(i["pf"]) if i else "n/a"
        is_tr = str(i["trades"]) if i else "0"
        print(f"{r['pairs']:<40}{o['annual']:>9.1f}{o['mtm_dd']:>9.2f}{pfx(o['pf']):>7}{o['trades']:>7}"
              f"{is_an:>9}{is_dd:>8}{is_pf:>7}{is_tr:>6}")

    show = passing if passing else sorted(rows, key=lambda r: r["oos"]["annual"], reverse=True)
    for r in show[:5]:
        line(r)
    if not passing:
        print("\n(никто не прошёл OOS-фильтр — показаны лучшие по OOS annual)")

    best = (passing or show)[0]
    o = best["oos"]; i = best["is"]
    print(f"\n{'='*len(hdr)}\nWALK-FORWARD топ-1: {best['pairs']} @ risk 0.70%, conc 3")
    print("=" * len(hdr))
    if i:
        print(f"  IN-SAMPLE  (2023-06..2024-12): annual={i['annual']:.1f}% MTM_DD={i['mtm_dd']:.2f}% "
              f"PF={pfx(i['pf'])} win={i['win']:.1f}% trades={i['trades']} span={i['years']}y final=${i['final']:,.0f}")
    else:
        print("  IN-SAMPLE: нет сделок")
    print(f"  OUT-OF-SAMPLE (2025-01..now): annual={o['annual']:.1f}% MTM_DD={o['mtm_dd']:.2f}% "
          f"PF={pfx(o['pf'])} win={o['win']:.1f}% trades={o['trades']} span={o['years']}y final=${o['final']:,.0f}")
    if i and i["annual"] != 0:
        degr = (o["annual"] - i["annual"]) / abs(i["annual"]) * 100
        print(f"  OOS vs IS annual: {i['annual']:.1f}% -> {o['annual']:.1f}%  ({degr:+.0f}%)")

    # save config_bot3.json
    combo = tuple(best["pairs"].split("+"))
    cfg = json.load(open(os.path.join(BASE, "config.json")))
    cfg["risk_per_trade"] = RISK
    cfg["leverage"] = 5
    cfg.pop("symbols", None)
    cfg["max_concurrent"] = MAX_CONCURRENT
    cfg["symbols"] = list(combo)
    json.dump(cfg, open(os.path.join(BASE, "config_bot3.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\nconfig_bot3.json обновлён -> {best['pairs']} (risk 0.70%, conc 3)")
    print(f"Все результаты -> {out} ({len(rows)} строк)")


if __name__ == "__main__":
    main()
