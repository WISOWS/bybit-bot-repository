"""Bot #5 search: hybrid portfolios robust on BOTH IS and OOS, from remaining pairs.

Pool = unoccupied long-history pairs (excl. NEAR/SOL/LINK/ENA/ONDO/ZEC/SUI/ATOM/INJ/TIA/APT).
All 3-pair combos, risk 0.70%, conc 3. IS 2023-06..2024-12, OOS 2025-01..now, MTM DD.
Robust filter: MTM DD<10% on BOTH windows, both profitable, trades>50, OOS annual>80.
Saves top-1 by OOS annual -> config_bot5.json.
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
POOL = ["OPUSDT", "SNXUSDT", "ENSUSDT", "DOTUSDT", "AVAXUSDT", "ARBUSDT",
        "XRPUSDT", "XLMUSDT", "HBARUSDT", "ETHUSDT", "DOGEUSDT", "BTCUSDT",
        "BCHUSDT", "ADAUSDT", "1000PEPEUSDT", "WLDUSDT", "TONUSDT"]
RISK = 0.007
MAX_CONCURRENT = 3
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def run(raw, combo, win, cmaps):
    P.SYMBOLS = list(combo); P.RISK = RISK
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=MAX_CONCURRENT)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, combo)
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"], "pf": m["profit_factor"],
            "win": m["winrate_pct"], "trades": m["trades"], "years": m["years"], "final": m["final_balance"]}


def main() -> None:
    raw: Dict[str, List[Tuple]] = {}
    cmaps: Dict[str, Dict[int, float]] = {}
    for s in POOL:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    print(f"loaded {len(POOL)} pairs")

    rows = []
    for combo in itertools.combinations(POOL, 3):
        o = run(raw, combo, OOS_WIN, cmaps)
        i = run(raw, combo, IS_WIN, cmaps)
        if o is None:
            continue
        rows.append({"pairs": "+".join(combo), "o": o, "i": i})

    out = os.path.join(BASE, "bot5_search_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pairs", "oos_annual", "oos_dd", "oos_pf", "oos_trades",
                    "is_annual", "is_dd", "is_pf", "is_trades"])
        for r in rows:
            o = r["o"]; i = r["i"]
            w.writerow([r["pairs"], o["annual"], o["dd"],
                        "inf" if math.isinf(o["pf"]) else round(o["pf"], 4), o["trades"],
                        i["annual"] if i else "", i["dd"] if i else "",
                        ("inf" if (i and math.isinf(i["pf"])) else (round(i["pf"], 4) if i else "")),
                        i["trades"] if i else 0])

    robust = [r for r in rows if r["i"] and r["o"]["dd"] < 10 and r["i"]["dd"] < 10
              and r["o"]["annual"] > 80 and r["i"]["annual"] > 0
              and r["o"]["trades"] > 50 and r["i"]["trades"] > 50]
    robust.sort(key=lambda r: r["o"]["annual"], reverse=True)

    hdr = f"{'pairs':<40}{'OOS_an%':>9}{'OOS_DD%':>8}{'IS_an%':>8}{'IS_DD%':>8}{'OOS_PF':>7}{'IS_PF':>7}"
    print(f"\nBOT #5 — robust (DD<10% on BOTH), OOS annual>80: matched {len(robust)} of {len(rows)}")
    print(hdr); print("-" * len(hdr))
    for r in robust[:10]:
        o = r["o"]; i = r["i"]
        print(f"{r['pairs']:<40}{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>8.1f}{i['dd']:>8.2f}{pfx(o['pf']):>7}{pfx(i['pf']):>7}")

    if not robust:
        print("Нет устойчивых портфелей с OOS annual>80. Топ-5 устойчивых по OOS annual (любой annual):")
        r2 = [r for r in rows if r["i"] and r["o"]["dd"] < 10 and r["i"]["dd"] < 10 and r["o"]["trades"] > 50 and r["i"]["trades"] > 50 and r["i"]["annual"] > 0]
        r2.sort(key=lambda r: r["o"]["annual"], reverse=True)
        for r in r2[:5]:
            o = r["o"]; i = r["i"]
            print(f"{r['pairs']:<40}{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>8.1f}{i['dd']:>8.2f}{pfx(o['pf']):>7}{pfx(i['pf']):>7}")
        robust = r2

    if robust:
        best = robust[0]
        combo = tuple(best["pairs"].split("+"))
        cfg = json.load(open(os.path.join(BASE, "config.json")))
        cfg["risk_per_trade"] = RISK; cfg["leverage"] = 5
        cfg.pop("symbols", None); cfg["max_concurrent"] = MAX_CONCURRENT; cfg["symbols"] = list(combo)
        json.dump(cfg, open(os.path.join(BASE, "config_bot5.json"), "w"), indent=2, ensure_ascii=False)
        print(f"\nconfig_bot5.json -> {best['pairs']} (risk 0.70%, conc 3)")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
