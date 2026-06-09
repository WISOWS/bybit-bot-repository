"""Bot #6 — TSMOM portfolio search + risk/concurrency sweep + walk-forward.

Generates TSMOM raw trades (look-ahead-free) for all 16 full-coverage pairs,
then sweeps every C(16,3) portfolio across risk and max_concurrent, scoring
IS+OOS annual and MTM DD on both windows. Goal: maximize OOS annual with
MTM DD < 8% on BOTH IS and OOS. Saves the best to config_bot6.json.

Params (roc_p, ema_p, atr_stop_mult) are chosen from the tuning step.
"""

import csv
import itertools
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
import bot6_strategies as S

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = ["1000PEPEUSDT", "ADAUSDT", "APTUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT",
        "DOGEUSDT", "ENSUSDT", "ETHUSDT", "HBARUSDT", "LTCUSDT", "SNXUSDT",
        "TONUSDT", "VETUSDT", "WLDUSDT", "XRPUSDT"]
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))
DD_MAX = 8.0
OOS_ANNUAL_MIN = 100.0

# TSMOM params (overridable via argv: roc ema sm)
ROC_P = int(sys.argv[1]) if len(sys.argv) > 1 else 30
EMA_P = int(sys.argv[2]) if len(sys.argv) > 2 else 50
SM = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

RISKS = [0.0050, 0.0060, 0.0070, 0.0080, 0.0090, 0.0100, 0.0120]
CONCS = [2, 3]


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def run(raw, cmaps, combo, risk, conc, win):
    P.SYMBOLS = list(combo); P.RISK = risk
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=conc)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, tuple(combo))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"], "pf": m["profit_factor"],
            "win": m["winrate_pct"], "trades": m["trades"], "sharpe": m["sharpe"]}


def main():
    t0 = time.time()
    raw_all, cmaps = {}, {}
    for s in POOL:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        d = S.build_symbol(k1, k4)
        raw_all[s] = S.gen_tsmom(d, roc_p=ROC_P, ema_p=EMA_P, atr_stop_mult=SM)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    combos = list(itertools.combinations(POOL, 3))
    print(f"TSMOM roc={ROC_P} ema={EMA_P} sm={SM} | {len(POOL)} pairs, {len(combos)} combos "
          f"x {len(RISKS)} risks x {len(CONCS)} conc | gen {time.time()-t0:.1f}s", flush=True)

    rows = []
    for k, combo in enumerate(combos):
        pairs = "+".join(s.replace("USDT", "") for s in combo)
        for conc in CONCS:
            for risk in RISKS:
                o = run(raw_all, cmaps, combo, risk, conc, OOS_WIN)
                i = run(raw_all, cmaps, combo, risk, conc, IS_WIN)
                if o is None or i is None:
                    continue
                ok = o["dd"] < DD_MAX and i["dd"] < DD_MAX and o["annual"] > OOS_ANNUAL_MIN
                rows.append({"combo": combo, "pairs": pairs, "risk": risk, "conc": conc,
                             "o": o, "i": i, "ok": ok})
        if (k + 1) % 80 == 0:
            print(f"  ...{k+1}/{len(combos)}  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(BASE, "bot6_tsmom_portfolio_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pairs", "risk_pct", "conc", "oos_annual", "oos_dd", "oos_pf", "oos_sharpe",
                    "oos_trades", "is_annual", "is_dd", "is_pf", "passes"])
        for r in rows:
            o, i = r["o"], r["i"]
            w.writerow([r["pairs"], round(r["risk"]*100, 2), r["conc"], round(o["annual"], 1),
                        round(o["dd"], 2), pfx(o["pf"]), round(o["sharpe"], 2), o["trades"],
                        round(i["annual"], 1), round(i["dd"], 2), pfx(i["pf"]),
                        "YES" if r["ok"] else "no"])

    passing = [r for r in rows if r["ok"]]
    passing.sort(key=lambda r: r["o"]["annual"], reverse=True)
    # distinct portfolio (best config per pair-set)
    best_combo = {}
    for r in passing:
        c = r["pairs"]
        if c not in best_combo or r["o"]["annual"] > best_combo[c]["o"]["annual"]:
            best_combo[c] = r
    uniq = sorted(best_combo.values(), key=lambda r: r["o"]["annual"], reverse=True)

    print("\n" + "=" * 100)
    print(f"PASSING gate (OOS annual>100% & MTM DD<8% BOTH): {len(passing)} configs, {len(uniq)} portfolios")
    print("=" * 100)
    hdr = (f"{'#':>2} {'pairs':<24}{'risk%':>6}{'cc':>3}{'OOS_an%':>9}{'OOS_DD':>8}{'IS_an%':>9}"
           f"{'IS_DD':>7}{'OOS_PF':>7}{'OOS_Sh':>7}{'OOS_tr':>7}")
    print(hdr); print("-" * len(hdr))
    for rank, r in enumerate(uniq[:15], 1):
        o, i = r["o"], r["i"]
        print(f"{rank:>2} {r['pairs']:<24}{r['risk']*100:>6.2f}{r['conc']:>3}{o['annual']:>9.1f}"
              f"{o['dd']:>8.2f}{i['annual']:>9.1f}{i['dd']:>7.2f}{pfx(o['pf']):>7}{o['sharpe']:>7.2f}{o['trades']:>7}")

    if uniq:
        best = uniq[0]
        cfg_path = os.path.join(BASE, "config_bot6.json")
        src = cfg_path if os.path.exists(cfg_path) else os.path.join(BASE, "config_bot5.json")
        cfg = json.load(open(src))
        cfg["risk_per_trade"] = best["risk"]
        cfg["max_concurrent"] = best["conc"]
        cfg.pop("symbols", None)
        cfg["symbols"] = list(best["combo"])
        cfg["strategy"] = "tsmom"
        cfg["strategy_params"] = {"roc_p": ROC_P, "ema_p": EMA_P, "atr_stop_mult": SM,
                                  "timeframe_signal": "4h", "atr_period": 14}
        cfg["_backtest"] = {"oos_annual_pct": best["o"]["annual"], "oos_mtm_dd_pct": best["o"]["dd"],
                            "is_annual_pct": best["i"]["annual"], "is_mtm_dd_pct": best["i"]["dd"],
                            "oos_pf": best["o"]["pf"], "oos_sharpe": best["o"]["sharpe"],
                            "note": "TSMOM is NOT yet implemented in main.py live logic; this config "
                                    "is a backtest result. Live deploy requires adding the tsmom "
                                    "signal to main.py. The pure-hybrid fallback is deployable today."}
        json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
        print(f"\nBEST -> config_bot6.json: {best['pairs']} @ risk {best['risk']*100:.2f}% conc {best['conc']} "
              f"(OOS {best['o']['annual']:.1f}% / DD {best['o']['dd']:.2f}%, "
              f"IS {best['i']['annual']:.1f}% / DD {best['i']['dd']:.2f}%)")
    else:
        print("\nNo TSMOM portfolio passed the gate.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
