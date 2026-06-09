"""Bot #6 strategy/portfolio search — regime_switch_hybrid.

Universe = pairs NOT used by bots #1-5 AND with full IS+OOS coverage
(history starting <=2024-01 and extending into OOS). 16 pairs:
  1000PEPE, ADA, APT, BCH, BNB, BTC, DOGE, ENS, ETH, HBAR, LTC, SNX, TON, VET, WLD, XRP

Walk-forward: IS = 2023-06-01..2025-01-01 (<=2024-12), OOS = 2025-01-01..2026-06-01.
Account $100k, leverage 5x, fee 0.055%/side, slip 0.05%/side, funding -0.01%/8h short.

Step 1: single-pair OOS ranking.
Step 2: all C(16,3)=560 portfolios, IS+OOS, MTM DD both periods.
Step 3: risk sweep (0.45..0.75%) on portfolios passing the gate.

Gate: MTM DD < 8% on BOTH IS and OOS, OOS annual > 100% (maximize, no upper cap).
"""

import csv
import itertools
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))

POOL = ["1000PEPEUSDT", "ADAUSDT", "APTUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT",
        "DOGEUSDT", "ENSUSDT", "ETHUSDT", "HBARUSDT", "LTCUSDT", "SNXUSDT",
        "TONUSDT", "VETUSDT", "WLDUSDT", "XRPUSDT"]

IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))   # <= 2024-12
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))  # 2025-01+

DD_MAX = 8.0
OOS_ANNUAL_MIN = 100.0
MAX_CONCURRENT = 3
RISK_SWEEP = [0.0045, 0.0050, 0.0055, 0.0060, 0.0065, 0.0070, 0.0075]
BASELINE_RISK = 0.0060


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def load_cache(pool):
    raw, cmaps = {}, {}
    for s in pool:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    return raw, cmaps


def run(raw, cmaps, combo, risk, win):
    """Run a portfolio (tuple of symbols) at a given risk over a window."""
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
    raw, cmaps = load_cache(POOL)
    print(f"loaded {len(POOL)} pairs", flush=True)

    # ---------- STEP 1: single-pair OOS ranking ----------
    print("\n" + "=" * 78)
    print("STEP 1 — single-pair OOS ranking (regime_switch_hybrid, risk 0.60%)")
    print("=" * 78)
    single = []
    for s in POOL:
        o = run(raw, cmaps, (s,), BASELINE_RISK, OOS_WIN)
        i = run(raw, cmaps, (s,), BASELINE_RISK, IS_WIN)
        if o is None:
            continue
        single.append((s, o, i))
    single.sort(key=lambda r: (P.pf_sort_key(r[1]) if hasattr(P, "pf_sort_key") else
                               (1e9 if math.isinf(r[1]["pf"]) else r[1]["pf"])), reverse=True)
    hdr1 = (f"{'pair':<14}{'OOS_an%':>9}{'OOS_DD':>8}{'OOS_PF':>8}{'OOS_tr':>7}"
            f"{'IS_an%':>9}{'IS_DD':>8}{'IS_PF':>8}{'IS_tr':>7}")
    print(hdr1); print("-" * len(hdr1))
    for s, o, i in single:
        ip = i if i else {"annual": 0, "dd": 0, "pf": 0, "trades": 0}
        print(f"{s:<14}{o['annual']:>9.1f}{o['dd']:>8.2f}{pfx(o['pf']):>8}{o['trades']:>7}"
              f"{ip['annual']:>9.1f}{ip['dd']:>8.2f}{pfx(ip['pf']):>8}{ip['trades']:>7}")
    with open(os.path.join(BASE, "bot6_single_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "oos_annual", "oos_dd", "oos_pf", "oos_trades",
                    "is_annual", "is_dd", "is_pf", "is_trades"])
        for s, o, i in single:
            ip = i if i else {"annual": 0, "dd": 0, "pf": 0, "trades": 0}
            w.writerow([s, round(o["annual"], 1), round(o["dd"], 2), pfx(o["pf"]), o["trades"],
                        round(ip["annual"], 1), round(ip["dd"], 2), pfx(ip["pf"]), ip["trades"]])

    # ---------- STEP 2: all 3-pair combos at baseline risk ----------
    combos = list(itertools.combinations(POOL, 3))
    print("\n" + "=" * 78)
    print(f"STEP 2 — all C(16,3)={len(combos)} portfolios @ risk {BASELINE_RISK*100:.2f}%, conc {MAX_CONCURRENT}")
    print("=" * 78, flush=True)
    rows = []
    for k, combo in enumerate(combos):
        o = run(raw, cmaps, combo, BASELINE_RISK, OOS_WIN)
        i = run(raw, cmaps, combo, BASELINE_RISK, IS_WIN)
        if o is None or i is None:
            continue
        rows.append({"combo": combo,
                     "pairs": "+".join(s.replace("USDT", "") for s in combo),
                     "o": o, "i": i})
        if (k + 1) % 80 == 0:
            print(f"  ...{k+1}/{len(combos)}", flush=True)

    with open(os.path.join(BASE, "bot6_combo_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pairs", "oos_annual", "oos_dd", "oos_pf", "oos_sharpe", "oos_trades",
                    "is_annual", "is_dd", "is_pf", "is_trades", "passes"])
        for r in rows:
            o, i = r["o"], r["i"]
            ok = o["dd"] < DD_MAX and i["dd"] < DD_MAX and o["annual"] > OOS_ANNUAL_MIN
            w.writerow([r["pairs"], round(o["annual"], 1), round(o["dd"], 2), pfx(o["pf"]),
                        round(o["sharpe"], 2), o["trades"], round(i["annual"], 1), round(i["dd"], 2),
                        pfx(i["pf"]), i["trades"], "YES" if ok else "no"])

    passing = [r for r in rows if r["o"]["dd"] < DD_MAX and r["i"]["dd"] < DD_MAX
               and r["o"]["annual"] > OOS_ANNUAL_MIN]
    passing.sort(key=lambda r: r["o"]["annual"], reverse=True)
    print(f"\nPassing gate (DD<8% both + OOS annual>100%): {len(passing)}/{len(rows)}")
    hdr2 = (f"{'pairs':<26}{'OOS_an%':>9}{'OOS_DD':>8}{'IS_an%':>9}{'IS_DD':>7}"
            f"{'OOS_PF':>8}{'OOS_Sh':>7}{'OOS_tr':>7}")
    print(hdr2); print("-" * len(hdr2))
    show = passing if passing else sorted(rows, key=lambda r: r["o"]["annual"], reverse=True)
    for r in show[:25]:
        o, i = r["o"], r["i"]
        print(f"{r['pairs']:<26}{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>9.1f}"
              f"{i['dd']:>7.2f}{pfx(o['pf']):>8}{o['sharpe']:>7.2f}{o['trades']:>7}")
    if not passing:
        print("\n(no portfolio passed gate at baseline risk; showing top-25 by OOS annual)")

    # ---------- STEP 3: risk sweep on top candidates ----------
    # Take top-25 by OOS annual among those with DD headroom (<8% at baseline on both),
    # plus any passing; sweep risk levels.
    cand = passing[:] if passing else []
    headroom = [r for r in rows if r["o"]["dd"] < DD_MAX and r["i"]["dd"] < DD_MAX]
    headroom.sort(key=lambda r: r["o"]["annual"], reverse=True)
    for r in headroom[:30]:
        if r not in cand:
            cand.append(r)
    print("\n" + "=" * 78)
    print(f"STEP 3 — risk sweep {[f'{x*100:.2f}%' for x in RISK_SWEEP]} on {len(cand)} candidates")
    print("=" * 78, flush=True)

    sweep_rows = []
    for r in cand:
        combo = r["combo"]
        for risk in RISK_SWEEP:
            o = run(raw, cmaps, combo, risk, OOS_WIN)
            i = run(raw, cmaps, combo, risk, IS_WIN)
            if o is None or i is None:
                continue
            ok = o["dd"] < DD_MAX and i["dd"] < DD_MAX and o["annual"] > OOS_ANNUAL_MIN
            sweep_rows.append({"combo": combo, "pairs": r["pairs"], "risk": risk,
                               "o": o, "i": i, "ok": ok})

    with open(os.path.join(BASE, "bot6_risk_sweep_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pairs", "risk_pct", "oos_annual", "oos_dd", "oos_pf", "oos_sharpe",
                    "oos_trades", "is_annual", "is_dd", "is_pf", "passes"])
        for r in sweep_rows:
            o, i = r["o"], r["i"]
            w.writerow([r["pairs"], round(r["risk"]*100, 2), round(o["annual"], 1), round(o["dd"], 2),
                        pfx(o["pf"]), round(o["sharpe"], 2), o["trades"], round(i["annual"], 1),
                        round(i["dd"], 2), pfx(i["pf"]), "YES" if r["ok"] else "no"])

    final = [r for r in sweep_rows if r["ok"]]
    final.sort(key=lambda r: r["o"]["annual"], reverse=True)
    print("\n" + "=" * 78)
    print("FINAL TOP-10 PORTFOLIOS (passing gate, sorted by OOS Annual DESC)")
    print("=" * 78)
    hdr3 = (f"{'#':>2} {'pairs':<26}{'risk%':>6}{'OOS_an%':>9}{'OOS_DD':>8}{'IS_an%':>9}"
            f"{'IS_DD':>7}{'OOS_PF':>8}{'OOS_Sh':>7}{'OOS_tr':>7}")
    print(hdr3); print("-" * len(hdr3))
    for rank, r in enumerate(final[:10], 1):
        o, i = r["o"], r["i"]
        print(f"{rank:>2} {r['pairs']:<26}{r['risk']*100:>6.2f}{o['annual']:>9.1f}{o['dd']:>8.2f}"
              f"{i['annual']:>9.1f}{i['dd']:>7.2f}{pfx(o['pf']):>8}{o['sharpe']:>7.2f}{o['trades']:>7}")

    if final:
        best = final[0]
        cfg_path = os.path.join(BASE, "config_bot6.json")
        src = cfg_path if os.path.exists(cfg_path) else os.path.join(BASE, "config_bot5.json")
        cfg = json.load(open(src))
        cfg["risk_per_trade"] = best["risk"]
        cfg["max_concurrent"] = MAX_CONCURRENT
        cfg.pop("symbols", None)
        cfg["symbols"] = list(best["combo"])
        json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
        print(f"\nBEST -> config_bot6.json: {best['pairs']} @ risk {best['risk']*100:.2f}% "
              f"(OOS {best['o']['annual']:.1f}% an / DD {best['o']['dd']:.2f}%, "
              f"IS {best['i']['annual']:.1f}% an / DD {best['i']['dd']:.2f}%)")
    else:
        print("\nNo portfolio passed the gate (OOS annual>100% & DD<8% both) in baseline hybrid.")
        print("=> proceed to Step 4 (improved variant).")

    return bool(final)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 2)
