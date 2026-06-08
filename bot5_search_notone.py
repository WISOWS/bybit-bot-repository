"""Bot #5 search — принципиально БЕЗ TON.

Универсум = 8 доступных пар (не занятых другими ботами):
XRP, ADA, BNB, LTC, ETC, XLM, VET, ALGO. Перебор всех C(8,3)=56 комбинаций
на regime_switch_hybrid. Risk 0.60%, conc 3. IS до 2024-12, OOS 2025-01+.
Фильтр: MTM DD < 8% на ОБОИХ периодах И OOS Annual > 60%.
Топ-1 по OOS annual -> config_bot5.json.
"""

import csv
import itertools
import json
import math
import os
from typing import Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = ["XRPUSDT", "ADAUSDT", "BNBUSDT", "LTCUSDT",
        "ETCUSDT", "XLMUSDT", "VETUSDT", "ALGOUSDT"]
RISK = 0.0060
MAX_CONCURRENT = 3
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))   # до 2024-12
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))  # 2025-01+

DD_MAX = 8.0
OOS_ANNUAL_MIN = 60.0


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def run(raw, combo, win, cmaps):
    P.SYMBOLS = list(combo)
    P.RISK = RISK
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=MAX_CONCURRENT)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, combo)
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"],
            "pf": m["profit_factor"], "win": m["winrate_pct"], "trades": m["trades"]}


def main() -> None:
    raw: Dict[str, List[Tuple]] = {}
    cmaps: Dict[str, Dict[int, float]] = {}
    for s in POOL:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    combos = list(itertools.combinations(POOL, 3))
    print(f"loaded {len(POOL)} pairs, {len(combos)} combos")

    rows = []
    for combo in combos:
        o = run(raw, combo, OOS_WIN, cmaps)
        i = run(raw, combo, IS_WIN, cmaps)
        if o is None or i is None:
            continue
        rows.append({"pairs": "+".join(s.replace("USDT", "") for s in combo),
                     "combo": combo, "o": o, "i": i})

    out = os.path.join(BASE, "bot5_search_notone_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pairs", "oos_annual", "oos_dd", "oos_pf", "oos_trades",
                    "is_annual", "is_dd", "is_pf", "is_trades", "passes"])
        for r in rows:
            o, i = r["o"], r["i"]
            ok = (o["dd"] < DD_MAX and i["dd"] < DD_MAX and o["annual"] > OOS_ANNUAL_MIN)
            w.writerow([r["pairs"], round(o["annual"], 1), round(o["dd"], 2),
                        pfx(o["pf"]), o["trades"], round(i["annual"], 1), round(i["dd"], 2),
                        pfx(i["pf"]), i["trades"], "YES" if ok else "no"])

    passing = [r for r in rows if r["o"]["dd"] < DD_MAX and r["i"]["dd"] < DD_MAX
               and r["o"]["annual"] > OOS_ANNUAL_MIN]
    passing.sort(key=lambda r: r["o"]["annual"], reverse=True)

    hdr = (f"{'pairs':<22}{'OOS_an%':>9}{'OOS_DD':>8}{'IS_an%':>8}{'IS_DD':>7}"
           f"{'OOS_PF':>7}{'OOS_win':>8}{'OOS_tr':>7}")
    print(f"\nBOT #5 (без TON) — прошли фильтр (DD<8% оба + OOS an>60): "
          f"{len(passing)} из {len(rows)}")
    print(hdr)
    print("-" * len(hdr))
    for r in passing[:10]:
        o, i = r["o"], r["i"]
        print(f"{r['pairs']:<22}{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>8.1f}"
              f"{i['dd']:>7.2f}{pfx(o['pf']):>7}{o['win']:>8.1f}{o['trades']:>7}")

    if not passing:
        print("\nНи один портфель не прошёл фильтр. Топ-10 по OOS annual (для справки):")
        rows.sort(key=lambda r: r["o"]["annual"], reverse=True)
        for r in rows[:10]:
            o, i = r["o"], r["i"]
            print(f"{r['pairs']:<22}{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>8.1f}"
                  f"{i['dd']:>7.2f}{pfx(o['pf']):>7}{o['win']:>8.1f}{o['trades']:>7}")
        print(f"\nresults -> {out}")
        return

    best = passing[0]
    cfg_path = os.path.join(BASE, "config_bot5.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else json.load(open(os.path.join(BASE, "config.json")))
    cfg["risk_per_trade"] = RISK
    cfg["max_concurrent"] = MAX_CONCURRENT
    cfg.pop("symbols", None)
    cfg["symbols"] = list(best["combo"])
    json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nБЕСТ -> config_bot5.json: {best['pairs']} "
          f"(OOS {best['o']['annual']:.1f}% an / DD {best['o']['dd']:.2f}%, "
          f"IS DD {best['i']['dd']:.2f}%, risk 0.60%, conc 3)")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
