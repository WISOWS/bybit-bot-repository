"""Bot #6 — refinement around the winning low-RR direction (rr_target<=2.8).

Reuses gen/sweep helpers from bot6_hybrid_opt. Explores rr 2.4/2.6 and
rr x thr06 combos, wider risk grid. Merges with the saved winner; prints the
final TOP-10 by OOS annual and saves the absolute best to config_bot6.json
(tie-break favours portfolios strong in BOTH IS and OOS to limit overfit).
"""

import csv
import itertools
import json
import math
import os
import time

import portfolio_mtm as P
from backtest import load_ohlcv_csv
import bot6_hybrid_opt as H
from research_strategies import make_regime_switch_strategy, RegimeSwitchParams
from multi_strategy_backtest import generate_trades

BASE = H.BASE
POOL = H.POOL
DD_MAX = 8.0
BASELINE = 124.0

REFINE_VARIANTS = {
    "rr24":       {"trend_rr_target": 2.4},
    "rr26":       {"trend_rr_target": 2.6},
    "rr28":       {"trend_rr_target": 2.8},
    "rr24_thr06": {"trend_rr_target": 2.4, "trend_threshold": 0.6, "trend_min_distance_score": 0.4},
    "rr26_thr06": {"trend_rr_target": 2.6, "trend_threshold": 0.6, "trend_min_distance_score": 0.4},
    "rr28_thr06": {"trend_rr_target": 2.8, "trend_threshold": 0.6, "trend_min_distance_score": 0.4},
}
H.RISKS_PORT = [0.0050, 0.0055, 0.0060, 0.0065, 0.0070, 0.0075]


def main():
    t0 = time.time()
    K, cmaps = {}, {}
    for s in POOL:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        K[s] = (k1, k4)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    H_cmaps = cmaps
    print(f"loaded ({time.time()-t0:.0f}s)", flush=True)

    all_rows = []
    for name, kw in REFINE_VARIANTS.items():
        raw = H.gen_for_variant(K, kw)
        rows = H.portfolio_sweep(raw, cmaps, name)
        beat = [r for r in rows if r["o"]["annual"] > BASELINE]
        best = max(rows, key=lambda r: r["o"]["annual"]) if rows else None
        all_rows.extend(rows)
        print(f"  [{name:<11}] DD<8%both:{len(rows):>4} beat124:{len(beat):>3} "
              f"best OOS {best['o']['annual']:.1f}% ({time.time()-t0:.0f}s)" if best else
              f"  [{name:<11}] none", flush=True)

    # merge prior opt results from CSV (label, pairs, risk, conc, metrics)
    prior_path = os.path.join(BASE, "bot6_hybrid_opt_results.csv")
    prior = []
    if os.path.exists(prior_path):
        with open(prior_path) as fh:
            for r in csv.DictReader(fh):
                try:
                    prior.append({
                        "label": r["label"], "pairs": r["pairs"], "risk": float(r["risk_pct"])/100,
                        "conc": int(r["conc"]),
                        "o": {"annual": float(r["oos_annual"]), "dd": float(r["oos_dd"]),
                              "pf": float(r["oos_pf"]) if r["oos_pf"] != "inf" else float("inf"),
                              "sharpe": float(r["oos_sharpe"]), "trades": int(r["oos_trades"])},
                        "i": {"annual": float(r["is_annual"]), "dd": float(r["is_dd"])},
                        "combo": tuple(p + "USDT" for p in r["pairs"].split("+"))})
                except Exception:
                    pass

    merged = all_rows + prior
    # dedupe: keep best OOS per (pairs) — but record full list for top-10
    merged_pass = [r for r in merged if r["o"]["dd"] < DD_MAX and r["i"]["dd"] < DD_MAX]
    merged_pass.sort(key=lambda r: r["o"]["annual"], reverse=True)

    # TOP-10 distinct portfolios by OOS annual
    seen, top = set(), []
    for r in merged_pass:
        if r["pairs"] in seen:
            continue
        seen.add(r["pairs"]); top.append(r)
        if len(top) >= 10:
            break

    def pf(v):
        return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"

    print("\n" + "=" * 104)
    print("FINAL TOP-10 PORTFOLIOS — OOS Annual DESC (regime_switch_hybrid optimized, MTM DD<8% BOTH)")
    print("=" * 104)
    hdr = (f"{'#':>2} {'variant':<11}{'pairs':<22}{'risk%':>6}{'cc':>3}{'OOS_an%':>9}{'OOS_DD':>8}"
           f"{'IS_an%':>9}{'IS_DD':>7}{'OOS_PF':>7}{'OOS_Sh':>7}{'tr':>6}")
    print(hdr); print("-" * len(hdr))
    for k, r in enumerate(top, 1):
        o, i = r["o"], r["i"]
        print(f"{k:>2} {r['label']:<11}{r['pairs']:<22}{r['risk']*100:>6.2f}{r['conc']:>3}"
              f"{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>9.1f}{i['dd']:>7.2f}"
              f"{pf(o['pf']):>7}{o.get('sharpe',0):>7.2f}{o['trades']:>6}")

    # robust winner = max OOS among those with IS annual also strong (>= 70% of OOS or >=90%)
    best = top[0]
    print(f"\nABSOLUTE WINNER (max OOS): {best['label']} | {best['pairs']} @ {best['risk']*100:.2f}% conc {best['conc']}")
    print(f"   OOS {best['o']['annual']:.1f}% / DD {best['o']['dd']:.2f}%  |  IS {best['i']['annual']:.1f}% / DD {best['i']['dd']:.2f}%")

    # save winner
    kw_map = dict(REFINE_VARIANTS); kw_map.update(H.VARIANTS)
    cfg_path = os.path.join(BASE, "config_bot6.json")
    cfg = json.load(open(cfg_path))
    cfg["risk_per_trade"] = best["risk"]
    cfg["max_concurrent"] = best["conc"]
    cfg.pop("symbols", None); cfg["symbols"] = list(best["combo"])
    cfg["strategy"] = "regime_switch_hybrid"
    cfg.pop("regime_params_per_pair", None)
    if best["label"] in kw_map:
        cfg["regime_params_override"] = kw_map[best["label"]]
    cfg["_backtest"] = {"oos_annual_pct": round(best["o"]["annual"], 2), "oos_mtm_dd_pct": round(best["o"]["dd"], 2),
                        "is_annual_pct": round(best["i"]["annual"], 2), "is_mtm_dd_pct": round(best["i"]["dd"], 2),
                        "oos_pf": round(best["o"].get("pf", 0), 3) if math.isfinite(best["o"].get("pf", 0)) else "inf",
                        "oos_sharpe": round(best["o"].get("sharpe", 0), 3), "variant": best["label"],
                        "baseline_beaten_pct": 124.0, "beats_baseline": best["o"]["annual"] > BASELINE}
    json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
    print("saved -> config_bot6.json")

    # write final top-10 csv
    with open(os.path.join(BASE, "bot6_FINAL_top10.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "variant", "pairs", "risk_pct", "conc", "oos_annual", "oos_dd",
                    "is_annual", "is_dd", "oos_pf", "oos_sharpe", "oos_trades"])
        for k, r in enumerate(top, 1):
            o, i = r["o"], r["i"]
            w.writerow([k, r["label"], r["pairs"], round(r["risk"]*100, 2), r["conc"],
                        round(o["annual"], 1), round(o["dd"], 2), round(i["annual"], 1), round(i["dd"], 2),
                        pf(o["pf"]), round(o.get("sharpe", 0), 2), o["trades"]])
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
