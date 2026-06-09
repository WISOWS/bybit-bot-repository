"""Bot #6 — screen all new strategies (+ hybrid) single-pair, IS/OOS, MTM DD.

For each strategy and each full-coverage pair: generate raw trades, run a
single-symbol portfolio at risk 0.50%, compute MTM DD on IS and OOS windows.
Reports per-strategy summary and best (strategy,pair) configs.
"""

import csv
import math
import os
import time
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import gen_hybrid, generate_trades
import bot6_strategies as S

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = ["1000PEPEUSDT", "ADAUSDT", "APTUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT",
        "DOGEUSDT", "ENSUSDT", "ETHUSDT", "HBARUSDT", "LTCUSDT", "SNXUSDT",
        "TONUSDT", "VETUSDT", "WLDUSDT", "XRPUSDT"]
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))
RISK = 0.0050
DD_MAX = 8.0


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def run_single(raw_by_symbol, cmaps, sym, win):
    P.SYMBOLS = [sym]; P.RISK = RISK
    ex, m = P.run_portfolio(raw_by_symbol, window=win, max_concurrent=1)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, (sym,))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"], "pf": m["profit_factor"],
            "win": m["winrate_pct"], "trades": m["trades"], "sharpe": m["sharpe"]}


def main():
    # load klines once
    K = {}
    cmaps = {}
    for s in POOL:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        K[s] = (k1, k4)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    print(f"loaded {len(POOL)} pairs", flush=True)

    strat_list = ["hybrid"] + list(S.STRATS.keys())
    rows = []
    for strat in strat_list:
        t0 = time.time()
        raw_by_symbol = {}
        for s in POOL:
            k1, k4 = K[s]
            if strat == "hybrid":
                raw_by_symbol[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
            else:
                raw_by_symbol[s] = S.gen_raw(strat, k1, k4)
        # per-pair metrics
        per = []
        for s in POOL:
            o = run_single(raw_by_symbol, cmaps, s, OOS_WIN)
            i = run_single(raw_by_symbol, cmaps, s, IS_WIN)
            if o is None or i is None:
                continue
            per.append((s, o, i))
            rows.append({"strat": strat, "pair": s, "o": o, "i": i})
        # strategy aggregate
        n_trades = sum(len(raw_by_symbol[s]) for s in POOL)
        passes = [r for r in per if r[1]["dd"] < DD_MAX and r[2]["dd"] < DD_MAX and r[1]["annual"] > 0]
        avg_oos = sum(r[1]["annual"] for r in per) / len(per) if per else 0
        print(f"[{strat:<16}] {n_trades:>6} trades total | "
              f"avgOOS_an={avg_oos:6.1f}% | pairs DD<8%both: {len(passes)}/{len(per)} | "
              f"{time.time()-t0:4.1f}s", flush=True)

    # write all
    with open(os.path.join(BASE, "bot6_strat_scan_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "pair", "oos_annual", "oos_dd", "oos_pf", "oos_sharpe",
                    "oos_trades", "is_annual", "is_dd", "is_pf", "is_trades"])
        for r in rows:
            o, i = r["o"], r["i"]
            w.writerow([r["strat"], r["pair"], round(o["annual"], 1), round(o["dd"], 2),
                        pfx(o["pf"]), round(o["sharpe"], 2), o["trades"], round(i["annual"], 1),
                        round(i["dd"], 2), pfx(i["pf"]), i["trades"]])

    # best single (strategy,pair) passing DD<8% both, by OOS annual
    good = [r for r in rows if r["o"]["dd"] < DD_MAX and r["i"]["dd"] < DD_MAX]
    good.sort(key=lambda r: r["o"]["annual"], reverse=True)
    print("\n" + "=" * 96)
    print("TOP-30 single (strategy,pair) by OOS annual with MTM DD<8% on BOTH IS & OOS, risk 0.50%")
    print("=" * 96)
    hdr = (f"{'strategy':<16}{'pair':<13}{'OOS_an%':>9}{'OOS_DD':>8}{'IS_an%':>9}{'IS_DD':>7}"
           f"{'OOS_PF':>8}{'OOS_Sh':>7}{'OOS_tr':>7}")
    print(hdr); print("-" * len(hdr))
    for r in good[:30]:
        o, i = r["o"], r["i"]
        print(f"{r['strat']:<16}{r['pair']:<13}{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>9.1f}"
              f"{i['dd']:>7.2f}{pfx(o['pf']):>8}{o['sharpe']:>7.2f}{o['trades']:>7}")


if __name__ == "__main__":
    main()
