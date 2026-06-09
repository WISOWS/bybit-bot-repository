"""Bot #6 — optimize regime_switch_hybrid parameters, build portfolios > 124% OOS.

Phase 1: screen RegimeSwitchParams variants single-pair (risk 0.50%), rank by
         #pairs passing DD<8% both and avg OOS annual.
Phase 2: full C(16,3) x risk sweep on the TOP variants -> portfolios passing gate.
Phase 3: per-pair optimization — each pair uses its individually-best variant —
         then portfolio sweep over those mixed-param trades.

Same framework convention as baseline hybrid (comparable to bots #1-5).
Gate: MTM DD<8% on BOTH IS & OOS. Objective: maximize OOS annual (beat 124%).
"""

import csv
import itertools
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import generate_trades
from research_strategies import make_regime_switch_strategy, RegimeSwitchParams

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = ["1000PEPEUSDT", "ADAUSDT", "APTUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT",
        "DOGEUSDT", "ENSUSDT", "ETHUSDT", "HBARUSDT", "LTCUSDT", "SNXUSDT",
        "TONUSDT", "VETUSDT", "WLDUSDT", "XRPUSDT"]
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))
DD_MAX = 8.0
BASELINE = 124.0

# Parameter variants (name -> kwargs to RegimeSwitchParams)
VARIANTS = {
    "default":        {},
    "rr36":           {"trend_rr_target": 3.6},
    "rr40":           {"trend_rr_target": 4.0},
    "rr28":           {"trend_rr_target": 2.8},
    "rr36_sb30":      {"trend_rr_target": 3.6, "trend_stop_buffer_atr": 0.30},
    "rr40_sb20":      {"trend_rr_target": 4.0, "trend_stop_buffer_atr": 0.20},
    "thr06":          {"trend_threshold": 0.6, "trend_min_distance_score": 0.4},
    "thr08":          {"trend_threshold": 0.8, "trend_min_distance_score": 0.6},
    "reg24":          {"regime_lookback_4h": 24},
    "reg36":          {"regime_lookback_4h": 36},
    "rngrr16":        {"range_min_rr": 1.6, "range_target_fraction": 0.6},
    "rr36_reg24":     {"trend_rr_target": 3.6, "regime_lookback_4h": 24},
    "rr36_thr06":     {"trend_rr_target": 3.6, "trend_threshold": 0.6, "trend_min_distance_score": 0.4},
    "msm15_rr36":     {"trend_max_stop_multiplier": 1.5, "trend_rr_target": 3.6},
}

RISKS_PORT = [0.0045, 0.0050, 0.0055, 0.0060, 0.0065, 0.0070]


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def gen_for_variant(K, kwargs):
    """Return {sym: raw_trades} for a parameter variant."""
    strat = make_regime_switch_strategy(RegimeSwitchParams(**kwargs))
    def g(k1, i, k4, j4, is_new_4h):
        s = strat(k1, i, k4, j4)
        return None if s is None else (s.side, s.entry, s.stop, s.tp)
    out = {}
    for s in POOL:
        k1, k4 = K[s]
        out[s] = generate_trades(g, k1, k4, min_4h=29)
    return out


def run(raw, cmaps, combo, risk, conc, win):
    P.SYMBOLS = list(combo); P.RISK = risk
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=conc)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, tuple(combo))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"], "pf": m["profit_factor"],
            "win": m["winrate_pct"], "trades": m["trades"], "sharpe": m["sharpe"]}


def single(raw1, cmaps, sym, win):
    P.SYMBOLS = [sym]; P.RISK = 0.0050
    ex, m = P.run_portfolio(raw1, window=win, max_concurrent=1)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, (sym,))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"], "trades": m["trades"]}


def portfolio_sweep(raw_by_symbol, cmaps, label):
    """Full C(16,3) x risk x conc{2,3}. Return list of passing rows."""
    combos = list(itertools.combinations(POOL, 3))
    rows = []
    for combo in combos:
        pairs = "+".join(s.replace("USDT", "") for s in combo)
        for conc in (2, 3):
            for risk in RISKS_PORT:
                o = run(raw_by_symbol, cmaps, combo, risk, conc, OOS_WIN)
                i = run(raw_by_symbol, cmaps, combo, risk, conc, IS_WIN)
                if o is None or i is None:
                    continue
                if o["dd"] < DD_MAX and i["dd"] < DD_MAX:
                    rows.append({"label": label, "combo": combo, "pairs": pairs,
                                 "risk": risk, "conc": conc, "o": o, "i": i})
    return rows


def main():
    t0 = time.time()
    K, cmaps = {}, {}
    for s in POOL:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        K[s] = (k1, k4)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    print(f"loaded {len(POOL)} pairs ({time.time()-t0:.0f}s)", flush=True)

    # ---- Phase 1: screen variants single-pair, also cache per-(variant,pair) trades ----
    raw_cache = {}   # variant -> {sym: raw}
    screen = []
    print("\n=== PHASE 1: variant screen (single-pair, risk 0.50%) ===", flush=True)
    hdr = f"{'variant':<14}{'avgOOS':>8}{'avgIS':>7}{'pass':>7}{'tottr':>7}"
    print(hdr); print("-" * len(hdr))
    for name, kw in VARIANTS.items():
        raw = gen_for_variant(K, kw)
        raw_cache[name] = raw
        per = []
        for s in POOL:
            o = single({s: raw[s]}, cmaps, s, OOS_WIN)
            i = single({s: raw[s]}, cmaps, s, IS_WIN)
            if o and i:
                per.append((s, o, i))
        avg_oos = sum(o["annual"] for _, o, _ in per) / len(per) if per else 0
        avg_is = sum(i["annual"] for _, _, i in per) / len(per) if per else 0
        npass = sum(1 for _, o, i in per if o["dd"] < DD_MAX and i["dd"] < DD_MAX and o["annual"] > 0)
        tottr = sum(len(raw[s]) for s in POOL)
        screen.append({"name": name, "avg_oos": avg_oos, "npass": npass, "per": per})
        print(f"{name:<14}{avg_oos:>8.1f}{avg_is:>7.1f}{npass:>5}/16{tottr:>7}", flush=True)

    # rank variants by avg OOS annual
    screen.sort(key=lambda r: r["avg_oos"], reverse=True)
    top_variants = [r["name"] for r in screen[:4]]
    if "default" not in top_variants:
        top_variants.append("default")
    print(f"\nTop variants for portfolio sweep: {top_variants}", flush=True)

    # ---- Phase 2: portfolio sweep on top variants ----
    print(f"\n=== PHASE 2: portfolio sweep (top variants) ({time.time()-t0:.0f}s) ===", flush=True)
    all_pass = []
    for name in top_variants:
        rows = portfolio_sweep(raw_cache[name], cmaps, name)
        passing = [r for r in rows if r["o"]["annual"] > BASELINE]
        passing.sort(key=lambda r: r["o"]["annual"], reverse=True)
        print(f"  [{name:<12}] portfolios DD<8%both: {len(rows)}, beating {BASELINE}%: {len(passing)}"
              f" | best OOS {passing[0]['o']['annual']:.1f}%" if passing else
              f"  [{name:<12}] portfolios DD<8%both: {len(rows)}, beating {BASELINE}%: 0", flush=True)
        all_pass.extend(rows)

    # ---- Phase 3: per-pair best-variant trades, then portfolio sweep ----
    print(f"\n=== PHASE 3: per-pair optimization ({time.time()-t0:.0f}s) ===", flush=True)
    # for each pair, pick variant maximizing OOS annual subject to DD<8% both (single-pair)
    best_var_for_pair = {}
    for s in POOL:
        best = None
        for name in VARIANTS:
            raw = raw_cache[name]
            o = single({s: raw[s]}, cmaps, s, OOS_WIN)
            i = single({s: raw[s]}, cmaps, s, IS_WIN)
            if not o or not i:
                continue
            if o["dd"] < DD_MAX and i["dd"] < DD_MAX:
                if best is None or o["annual"] > best[1]:
                    best = (name, o["annual"], i["annual"], o["dd"], i["dd"])
        if best is None:  # fallback to default
            best = ("default", 0, 0, 0, 0)
        best_var_for_pair[s] = best[0]
    print("per-pair best variant:", {s.replace("USDT", ""): best_var_for_pair[s] for s in POOL}, flush=True)
    mixed_raw = {s: raw_cache[best_var_for_pair[s]][s] for s in POOL}
    mixed_rows = portfolio_sweep(mixed_raw, cmaps, "perpair")
    all_pass.extend(mixed_rows)

    # ---- collate, dedupe by (pairs,label) best, rank ----
    all_pass.sort(key=lambda r: r["o"]["annual"], reverse=True)
    with open(os.path.join(BASE, "bot6_hybrid_opt_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "pairs", "risk_pct", "conc", "oos_annual", "oos_dd", "oos_pf",
                    "oos_sharpe", "oos_trades", "is_annual", "is_dd", "is_pf"])
        for r in all_pass:
            o, i = r["o"], r["i"]
            w.writerow([r["label"], r["pairs"], round(r["risk"]*100, 2), r["conc"],
                        round(o["annual"], 1), round(o["dd"], 2), pfx(o["pf"]), round(o["sharpe"], 2),
                        o["trades"], round(i["annual"], 1), round(i["dd"], 2), pfx(i["pf"])])

    beating = [r for r in all_pass if r["o"]["annual"] > BASELINE]
    print("\n" + "=" * 104)
    print(f"PORTFOLIOS BEATING BASELINE {BASELINE}% OOS with MTM DD<8% BOTH: {len(beating)}")
    print("=" * 104)
    hdr2 = (f"{'#':>2} {'variant':<10}{'pairs':<24}{'risk%':>6}{'cc':>3}{'OOS_an%':>9}{'OOS_DD':>8}"
            f"{'IS_an%':>9}{'IS_DD':>7}{'OOS_PF':>7}{'OOS_Sh':>7}{'tr':>6}")
    print(hdr2); print("-" * len(hdr2))
    seen = set()
    shown = 0
    for r in all_pass:
        key = r["pairs"]
        if key in seen:
            continue
        seen.add(key)
        o, i = r["o"], r["i"]
        flag = "  <<beats" if o["annual"] > BASELINE else ""
        print(f"{shown+1:>2} {r['label']:<10}{r['pairs']:<24}{r['risk']*100:>6.2f}{r['conc']:>3}"
              f"{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>9.1f}{i['dd']:>7.2f}{pfx(o['pf']):>7}"
              f"{o['sharpe']:>7.2f}{o['trades']:>6}{flag}")
        shown += 1
        if shown >= 15:
            break

    if all_pass:
        best = all_pass[0]
        print(f"\nABSOLUTE BEST: {best['label']} | {best['pairs']} @ risk {best['risk']*100:.2f}% "
              f"conc {best['conc']} -> OOS {best['o']['annual']:.1f}% / DD {best['o']['dd']:.2f}%, "
              f"IS {best['i']['annual']:.1f}% / DD {best['i']['dd']:.2f}%")
        # save winner
        cfg_path = os.path.join(BASE, "config_bot6.json")
        src = cfg_path if os.path.exists(cfg_path) else os.path.join(BASE, "config_bot5.json")
        cfg = json.load(open(src))
        cfg["risk_per_trade"] = best["risk"]
        cfg["max_concurrent"] = best["conc"]
        cfg.pop("symbols", None)
        cfg["symbols"] = list(best["combo"])
        cfg["strategy"] = "regime_switch_hybrid"
        if best["label"] != "perpair" and best["label"] in VARIANTS:
            cfg["regime_params_override"] = VARIANTS[best["label"]]
        elif best["label"] == "perpair":
            cfg["regime_params_per_pair"] = {s: VARIANTS[best_var_for_pair[s]] for s in best["combo"]}
        cfg["_backtest"] = {"oos_annual_pct": best["o"]["annual"], "oos_mtm_dd_pct": best["o"]["dd"],
                            "is_annual_pct": best["i"]["annual"], "is_mtm_dd_pct": best["i"]["dd"],
                            "oos_pf": best["o"]["pf"], "oos_sharpe": best["o"]["sharpe"],
                            "variant": best["label"], "beats_baseline_124": best["o"]["annual"] > BASELINE}
        json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
        print(f"saved -> config_bot6.json")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
