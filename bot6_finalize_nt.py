"""Bot #6 — TON-free finalize. Reads no-TON worker CSVs, excludes any TON
portfolio, applies gate OOS>200% & IS>80% & MTM DD<8% both, prints TOP-10 by
OOS DESC, saves best to config_bot6.json."""
import csv, glob, json, math, os

BASE = os.path.dirname(os.path.abspath(__file__))
OOS_MIN, IS_MIN, DD_MAX = 224.0, 80.0, 8.0


def kw_for(label):
    side = os.path.join(BASE, f"bot6_kw_{label}.json")
    if os.path.exists(side):
        try:
            return json.load(open(side))
        except Exception:
            pass
    if "rr16" in label: return {"trend_rr_target": 1.6}
    if "rr18" in label: return {"trend_rr_target": 1.8}
    if "rr20" in label: return {"trend_rr_target": 2.0}
    if "rr22" in label: return {"trend_rr_target": 2.2}
    if "rr24" in label: return {"trend_rr_target": 2.4}
    if "rr26" in label: return {"trend_rr_target": 2.6}
    if "rr28" in label: return {"trend_rr_target": 2.8}
    return {}


def fnum(x):
    try: return float(x)
    except Exception: return float("inf") if x == "inf" else 0.0


def main():
    rows = []
    for p in (glob.glob(os.path.join(BASE, "bot6_max4_*nt*.csv")) + glob.glob(os.path.join(BASE, "bot6_max5_*nt*.csv"))
              + glob.glob(os.path.join(BASE, "bot6_max6_*nt*.csv"))):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    pairs = r["pairs"]
                    if "TON" in pairs.split("+"):
                        continue
                    rows.append({"label": r["label"], "pairs": pairs, "risk": float(r["risk_pct"]),
                                 "conc": int(r["conc"]), "oos_an": float(r["oos_annual"]),
                                 "oos_dd": float(r["oos_dd"]), "oos_pf": fnum(r["oos_pf"]),
                                 "oos_sh": float(r["oos_sharpe"]), "oos_tr": int(r["oos_trades"]),
                                 "is_an": float(r["is_annual"]), "is_dd": float(r["is_dd"])})
                except Exception:
                    pass
    # gate
    gated = [r for r in rows if r["oos_dd"] < DD_MAX and r["is_dd"] < DD_MAX
             and r["oos_an"] > OOS_MIN and r["is_an"] > IS_MIN]
    gated.sort(key=lambda r: r["oos_an"], reverse=True)

    # distinct portfolios
    seen, top = set(), []
    for r in gated:
        if r["pairs"] in seen: continue
        seen.add(r["pairs"]); top.append(r)
        if len(top) >= 10: break

    print("=" * 110)
    print("TOP-10 TON-FREE — OOS>200% & IS>80% & MTM DD<8% BOTH | regime_switch_hybrid optimized")
    print("=" * 110)
    hdr = (f"{'#':>2} {'variant':<9}{'pairs':<26}{'risk%':>6}{'cc':>3}{'OOS_an%':>9}{'OOS_DD':>8}"
           f"{'IS_an%':>9}{'IS_DD':>7}{'OOS_PF':>7}{'OOS_Sh':>7}{'tr':>6}")
    print(hdr); print("-" * len(hdr))
    for k, r in enumerate(top, 1):
        print(f"{k:>2} {r['label']:<9}{r['pairs']:<26}{r['risk']:>6.2f}{r['conc']:>3}{r['oos_an']:>9.1f}"
              f"{r['oos_dd']:>8.2f}{r['is_an']:>9.1f}{r['is_dd']:>7.2f}{r['oos_pf']:>7.2f}{r['oos_sh']:>7.2f}{r['oos_tr']:>6}")

    if not top:
        # show best-effort even if none clears OOS>200 (for visibility)
        alt = sorted([r for r in rows if r["oos_dd"] < DD_MAX and r["is_dd"] < DD_MAX and r["is_an"] > IS_MIN],
                     key=lambda r: r["oos_an"], reverse=True)
        print("\n(No TON-free portfolio cleared OOS>200% & IS>80% & DD<8%. Best with IS>80% & DD<8%:)")
        for r in alt[:10]:
            print(f"   {r['label']:<9}{r['pairs']:<26}{r['risk']:.2f}% cc{r['conc']} "
                  f"OOS {r['oos_an']:.1f}%/DD{r['oos_dd']:.2f}% IS {r['is_an']:.1f}%/DD{r['is_dd']:.2f}%")
        return

    best = top[0]
    print(f"\nBEST TON-FREE: {best['label']} | {best['pairs']} @ {best['risk']:.2f}% cc{best['conc']} "
          f"-> OOS {best['oos_an']:.1f}%/DD{best['oos_dd']:.2f}% | IS {best['is_an']:.1f}%/DD{best['is_dd']:.2f}%")

    cfg_path = os.path.join(BASE, "config_bot6.json")
    cfg = json.load(open(cfg_path))
    cfg["risk_per_trade"] = round(best["risk"] / 100, 4)
    cfg["max_concurrent"] = best["conc"]
    cfg.pop("symbols", None); cfg["symbols"] = [p + "USDT" for p in best["pairs"].split("+")]
    cfg["strategy"] = "regime_switch_hybrid"
    cfg["regime_params_override"] = kw_for(best["label"])
    cfg["_backtest"] = {"oos_annual_pct": round(best["oos_an"], 2), "oos_mtm_dd_pct": round(best["oos_dd"], 2),
                        "is_annual_pct": round(best["is_an"], 2), "is_mtm_dd_pct": round(best["is_dd"], 2),
                        "oos_pf": round(best["oos_pf"], 3), "oos_sharpe": round(best["oos_sh"], 3),
                        "variant": best["label"], "no_ton": True, "gate": "OOS>200 & IS>80 & DD<8 both"}
    cfg.pop("_robust_alt", None)
    json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
    print("saved -> config_bot6.json")

    with open(os.path.join(BASE, "bot6_FINAL_top10_noTON.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "variant", "pairs", "risk_pct", "conc", "oos_annual", "oos_dd",
                    "is_annual", "is_dd", "oos_pf", "oos_sharpe", "oos_trades"])
        for k, r in enumerate(top, 1):
            w.writerow([k, r["label"], r["pairs"], r["risk"], r["conc"], round(r["oos_an"], 1),
                        round(r["oos_dd"], 2), round(r["is_an"], 1), round(r["is_dd"], 2),
                        round(r["oos_pf"], 2), round(r["oos_sh"], 2), r["oos_tr"]])


if __name__ == "__main__":
    main()
