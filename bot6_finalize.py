"""Bot #6 — merge all max-worker results + prior sweeps, print final TOP-10,
save the absolute OOS winner (and a robust pick) to config_bot6.json."""
import csv, glob, json, math, os

BASE = os.path.dirname(os.path.abspath(__file__))
VARIANT_KW = {
    "rr20": {"trend_rr_target": 2.0}, "rr22": {"trend_rr_target": 2.2},
    "rr24": {"trend_rr_target": 2.4}, "rr26": {"trend_rr_target": 2.6},
    "rr28": {"trend_rr_target": 2.8}, "rr40_sb20": {"trend_rr_target": 4.0, "trend_stop_buffer_atr": 0.20},
    "thr06": {"trend_threshold": 0.6, "trend_min_distance_score": 0.4},
    "rr22t6": {"trend_rr_target": 2.2, "trend_threshold": 0.6, "trend_min_distance_score": 0.4},
    "rr24t6": {"trend_rr_target": 2.4, "trend_threshold": 0.6, "trend_min_distance_score": 0.4},
    "rr18f": {"trend_rr_target": 1.8}, "rr20f": {"trend_rr_target": 2.0}, "rr22f": {"trend_rr_target": 2.2},
    "rr16f": {"trend_rr_target": 1.6}, "rr18f5": {"trend_rr_target": 1.8}, "rr20f5": {"trend_rr_target": 2.0},
    "default": {},
}


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("inf") if x == "inf" else 0.0


def load(path, has_conc=True):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "label": r["label"], "pairs": r["pairs"], "risk": float(r["risk_pct"]),
                    "conc": int(r.get("conc", 2)),
                    "oos_an": float(r["oos_annual"]), "oos_dd": float(r["oos_dd"]),
                    "oos_pf": fnum(r.get("oos_pf", 0)), "oos_sh": float(r.get("oos_sharpe", 0)),
                    "oos_tr": int(r.get("oos_trades", 0)),
                    "is_an": float(r["is_annual"]), "is_dd": float(r["is_dd"]),
                })
            except Exception:
                pass
    return rows


def main():
    rows = []
    for p in (glob.glob(os.path.join(BASE, "bot6_max_*.csv")) + glob.glob(os.path.join(BASE, "bot6_max4_*.csv"))
              + glob.glob(os.path.join(BASE, "bot6_max5_*.csv"))):
        rows += load(p)
    rows += load(os.path.join(BASE, "bot6_hybrid_opt_results.csv"))
    # gate: DD<8% both (already enforced in workers, re-check)
    rows = [r for r in rows if r["oos_dd"] < 8.0 and r["is_dd"] < 8.0]
    rows.sort(key=lambda r: r["oos_an"], reverse=True)

    # distinct portfolios (by pair-set) best OOS
    seen, top = set(), []
    for r in rows:
        if r["pairs"] in seen:
            continue
        seen.add(r["pairs"]); top.append(r)
        if len(top) >= 20:
            break

    print("=" * 110)
    print("FINAL TOP-10 — OOS Annual DESC | regime_switch_hybrid (optimized) | MTM DD<8% on BOTH IS & OOS")
    print("=" * 110)
    hdr = (f"{'#':>2} {'variant':<8}{'pairs':<22}{'risk%':>6}{'cc':>3}{'OOS_an%':>9}{'OOS_DD':>8}"
           f"{'IS_an%':>9}{'IS_DD':>7}{'OOS_PF':>7}{'OOS_Sh':>7}{'tr':>6}")
    print(hdr); print("-" * len(hdr))
    for k, r in enumerate(top[:10], 1):
        print(f"{k:>2} {r['label']:<8}{r['pairs']:<22}{r['risk']:>6.2f}{r['conc']:>3}{r['oos_an']:>9.1f}"
              f"{r['oos_dd']:>8.2f}{r['is_an']:>9.1f}{r['is_dd']:>7.2f}{r['oos_pf']:>7.2f}{r['oos_sh']:>7.2f}{r['oos_tr']:>6}")

    best = top[0]
    # robust pick: highest OOS among portfolios with IS annual >= 80% (stable both periods)
    robust = next((r for r in top if r["is_an"] >= 80.0), best)

    print(f"\nABSOLUTE OOS WINNER: {best['label']} | {best['pairs']} @ {best['risk']:.2f}% cc{best['conc']} "
          f"-> OOS {best['oos_an']:.1f}%/DD{best['oos_dd']:.2f}% | IS {best['is_an']:.1f}%/DD{best['is_dd']:.2f}%")
    print(f"ROBUST PICK (IS>=80%): {robust['label']} | {robust['pairs']} @ {robust['risk']:.2f}% cc{robust['conc']} "
          f"-> OOS {robust['oos_an']:.1f}%/DD{robust['oos_dd']:.2f}% | IS {robust['is_an']:.1f}%/DD{robust['is_dd']:.2f}%")

    # save ABSOLUTE winner to config
    cfg_path = os.path.join(BASE, "config_bot6.json")
    src = cfg_path if os.path.exists(cfg_path) else os.path.join(BASE, "config_bot5.json")
    cfg = json.load(open(src))
    cfg["risk_per_trade"] = round(best["risk"] / 100, 4)
    cfg["max_concurrent"] = best["conc"]
    cfg.pop("symbols", None)
    cfg["symbols"] = [p + "USDT" for p in best["pairs"].split("+")]
    cfg["strategy"] = "regime_switch_hybrid"
    cfg["regime_params_override"] = VARIANT_KW.get(best["label"], {})
    cfg["_backtest"] = {"oos_annual_pct": round(best["oos_an"], 2), "oos_mtm_dd_pct": round(best["oos_dd"], 2),
                        "is_annual_pct": round(best["is_an"], 2), "is_mtm_dd_pct": round(best["is_dd"], 2),
                        "oos_pf": round(best["oos_pf"], 3), "oos_sharpe": round(best["oos_sh"], 3),
                        "variant": best["label"], "beats_baseline_124": best["oos_an"] > 124.0}
    cfg["_robust_alt"] = {"pairs": robust["pairs"], "variant": robust["label"], "risk_pct": robust["risk"],
                          "conc": robust["conc"], "oos_annual": round(robust["oos_an"], 1),
                          "is_annual": round(robust["is_an"], 1), "oos_dd": round(robust["oos_dd"], 2),
                          "is_dd": round(robust["is_dd"], 2)}
    json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
    print("\nsaved ABSOLUTE winner -> config_bot6.json (robust alt recorded in _robust_alt)")

    with open(os.path.join(BASE, "bot6_FINAL_top10.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "variant", "pairs", "risk_pct", "conc", "oos_annual", "oos_dd",
                    "is_annual", "is_dd", "oos_pf", "oos_sharpe", "oos_trades"])
        for k, r in enumerate(top[:10], 1):
            w.writerow([k, r["label"], r["pairs"], r["risk"], r["conc"], round(r["oos_an"], 1),
                        round(r["oos_dd"], 2), round(r["is_an"], 1), round(r["is_dd"], 2),
                        round(r["oos_pf"], 2), round(r["oos_sh"], 2), r["oos_tr"]])


if __name__ == "__main__":
    main()
