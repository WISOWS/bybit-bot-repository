"""Bot #6 — FULL risk sweep across all C(16,3) portfolios.

For every 3-pair combo and every risk in {0.45..0.75%}, compute IS+OOS annual
and MTM DD on both windows. Lowering risk lowers DD ~linearly and annual more
slowly, so this maps the (combo, risk) frontier directly.

Gate: MTM DD < 8% on BOTH IS and OOS, OOS annual > 100%.
Output: ranked passing configs + frontier CSV. Best -> config_bot6.json.
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
POOL = ["1000PEPEUSDT", "ADAUSDT", "APTUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT",
        "DOGEUSDT", "ENSUSDT", "ETHUSDT", "HBARUSDT", "LTCUSDT", "SNXUSDT",
        "TONUSDT", "VETUSDT", "WLDUSDT", "XRPUSDT"]
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))
DD_MAX = 8.0
OOS_ANNUAL_MIN = 100.0
MAX_CONCURRENT = 3
RISKS = [0.0045, 0.0050, 0.0055, 0.0060, 0.0065, 0.0070, 0.0075]


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def run(raw, cmaps, combo, risk, win):
    P.SYMBOLS = list(combo)
    P.RISK = risk
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=MAX_CONCURRENT)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, tuple(combo))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"],
            "pf": m["profit_factor"], "win": m["winrate_pct"], "trades": m["trades"],
            "sharpe": m["sharpe"]}


def main():
    raw, cmaps = {}, {}
    for s in POOL:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    combos = list(itertools.combinations(POOL, 3))
    print(f"loaded {len(POOL)} pairs, {len(combos)} combos x {len(RISKS)} risks", flush=True)

    all_rows = []   # every (combo, risk)
    for k, combo in enumerate(combos):
        pairs = "+".join(s.replace("USDT", "") for s in combo)
        # OOS at lowest risk first; if even there annual<100 we can skip higher? No: higher risk -> higher annual.
        # IS DD at lowest risk; if >8 there, no higher risk passes -> still record for frontier.
        for risk in RISKS:
            o = run(raw, cmaps, combo, risk, OOS_WIN)
            i = run(raw, cmaps, combo, risk, IS_WIN)
            if o is None or i is None:
                continue
            ok = o["dd"] < DD_MAX and i["dd"] < DD_MAX and o["annual"] > OOS_ANNUAL_MIN
            all_rows.append({"combo": combo, "pairs": pairs, "risk": risk,
                             "o": o, "i": i, "ok": ok})
        if (k + 1) % 80 == 0:
            print(f"  ...{k+1}/{len(combos)}", flush=True)

    with open(os.path.join(BASE, "bot6_risk_full_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pairs", "risk_pct", "oos_annual", "oos_dd", "oos_pf", "oos_sharpe",
                    "oos_trades", "is_annual", "is_dd", "is_pf", "passes"])
        for r in all_rows:
            o, i = r["o"], r["i"]
            w.writerow([r["pairs"], round(r["risk"]*100, 2), round(o["annual"], 1), round(o["dd"], 2),
                        pfx(o["pf"]), round(o["sharpe"], 2), o["trades"], round(i["annual"], 1),
                        round(i["dd"], 2), pfx(i["pf"]), "YES" if r["ok"] else "no"])

    passing = [r for r in all_rows if r["ok"]]
    passing.sort(key=lambda r: r["o"]["annual"], reverse=True)

    # one row per combo: best passing risk (max OOS annual)
    best_per_combo = {}
    for r in passing:
        c = r["pairs"]
        if c not in best_per_combo or r["o"]["annual"] > best_per_combo[c]["o"]["annual"]:
            best_per_combo[c] = r
    uniq = sorted(best_per_combo.values(), key=lambda r: r["o"]["annual"], reverse=True)

    print("\n" + "=" * 92)
    print(f"PASSING gate (OOS annual>100% & MTM DD<8% on BOTH IS & OOS): "
          f"{len(passing)} (combo,risk) configs, {len(uniq)} distinct portfolios")
    print("=" * 92)
    hdr = (f"{'#':>2} {'pairs':<26}{'risk%':>6}{'OOS_an%':>9}{'OOS_DD':>8}{'IS_an%':>9}"
           f"{'IS_DD':>7}{'OOS_PF':>8}{'OOS_Sh':>7}{'OOS_tr':>7}")
    print(hdr); print("-" * len(hdr))
    for rank, r in enumerate(uniq[:15], 1):
        o, i = r["o"], r["i"]
        print(f"{rank:>2} {r['pairs']:<26}{r['risk']*100:>6.2f}{o['annual']:>9.1f}{o['dd']:>8.2f}"
              f"{i['annual']:>9.1f}{i['dd']:>7.2f}{pfx(o['pf']):>8}{o['sharpe']:>7.2f}{o['trades']:>7}")

    # how close are the near-misses? portfolios with DD<8 both at some risk but OOS annual<100
    near = []
    for c, rs in itertools.groupby(sorted(all_rows, key=lambda r: r["pairs"]), key=lambda r: r["pairs"]):
        rs = list(rs)
        dd_ok = [r for r in rs if r["o"]["dd"] < DD_MAX and r["i"]["dd"] < DD_MAX]
        if not dd_ok:
            continue
        b = max(dd_ok, key=lambda r: r["o"]["annual"])
        if not b["ok"]:
            near.append(b)
    near.sort(key=lambda r: r["o"]["annual"], reverse=True)
    print("\nNEAR-MISS (DD<8% both achievable, best OOS annual at that risk, <100%):")
    print(hdr); print("-" * len(hdr))
    for rank, r in enumerate(near[:15], 1):
        o, i = r["o"], r["i"]
        print(f"{rank:>2} {r['pairs']:<26}{r['risk']*100:>6.2f}{o['annual']:>9.1f}{o['dd']:>8.2f}"
              f"{i['annual']:>9.1f}{i['dd']:>7.2f}{pfx(o['pf']):>8}{o['sharpe']:>7.2f}{o['trades']:>7}")

    if uniq:
        best = uniq[0]
        cfg_path = os.path.join(BASE, "config_bot6.json")
        src = cfg_path if os.path.exists(cfg_path) else os.path.join(BASE, "config_bot5.json")
        cfg = json.load(open(src))
        cfg["risk_per_trade"] = best["risk"]
        cfg["max_concurrent"] = MAX_CONCURRENT
        cfg.pop("symbols", None)
        cfg["symbols"] = list(best["combo"])
        json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
        print(f"\nBEST -> config_bot6.json: {best['pairs']} @ risk {best['risk']*100:.2f}% "
              f"(OOS {best['o']['annual']:.1f}% / DD {best['o']['dd']:.2f}%, "
              f"IS {best['i']['annual']:.1f}% / DD {best['i']['dd']:.2f}%)")
    else:
        print("\nNo (combo,risk) passed the gate with baseline hybrid -> Step 4 (trailing stop).")


if __name__ == "__main__":
    main()
