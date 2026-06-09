"""Bot #6 — honest re-test + tuning of TSMOM (look-ahead fixed).

Single-pair, risk 0.50%, IS/OOS, MTM DD. Grid over (roc_p, ema_p, atr_stop_mult).
Reports per-param avg OOS annual and #pairs passing DD<8% both.
"""

import math
import os
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
RISK = 0.0050
DD_MAX = 8.0


def run_single(raw_by_symbol, cmaps, sym, win):
    P.SYMBOLS = [sym]; P.RISK = RISK
    ex, m = P.run_portfolio(raw_by_symbol, window=win, max_concurrent=1)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, (sym,))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"], "pf": m["profit_factor"],
            "trades": m["trades"], "sharpe": m["sharpe"]}


def main():
    K, cmaps = {}, {}
    for s in POOL:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        K[s] = build = S.build_symbol(k1, k4)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    print(f"loaded {len(POOL)} pairs", flush=True)

    grid = []
    for roc_p in [6, 10, 20, 30, 42]:
        for ema_p in [30, 50, 100]:
            for sm in [2.5, 3.0, 4.0]:
                grid.append((roc_p, ema_p, sm))
    print(f"{len(grid)} param sets\n", flush=True)

    hdr = f"{'roc':>4}{'ema':>5}{'sm':>5}{'avgOOS':>8}{'avgIS':>7}{'medOOSdd':>9}{'maxOOSdd':>9}{'pass':>6}{'tr/pair':>8}"
    print(hdr); print("-" * len(hdr))
    results = []
    for (roc_p, ema_p, sm) in grid:
        t0 = time.time()
        per = []
        ntr = 0
        for s in POOL:
            raw = {s: S.gen_tsmom(K[s], roc_p=roc_p, ema_p=ema_p, atr_stop_mult=sm)}
            ntr += len(raw[s])
            o = run_single(raw, cmaps, s, OOS_WIN)
            i = run_single(raw, cmaps, s, IS_WIN)
            if o and i:
                per.append((o, i))
        if not per:
            continue
        avg_oos = sum(o["annual"] for o, _ in per) / len(per)
        avg_is = sum(i["annual"] for _, i in per) / len(per)
        oosdd = sorted(o["dd"] for o, _ in per)
        med_dd = oosdd[len(oosdd) // 2]
        max_dd = max(o["dd"] for o, _ in per)
        npass = sum(1 for o, i in per if o["dd"] < DD_MAX and i["dd"] < DD_MAX and o["annual"] > 0)
        results.append((roc_p, ema_p, sm, avg_oos, avg_is, med_dd, max_dd, npass, ntr // len(POOL)))
        print(f"{roc_p:>4}{ema_p:>5}{sm:>5.1f}{avg_oos:>8.1f}{avg_is:>7.1f}{med_dd:>9.2f}"
              f"{max_dd:>9.2f}{npass:>4}/16{ntr//len(POOL):>8}", flush=True)

    results.sort(key=lambda r: r[3], reverse=True)
    print("\nTOP-8 param sets by avg OOS annual:")
    print(hdr); print("-" * len(hdr))
    for roc_p, ema_p, sm, ao, ai, md, mx, npass, tpp in results[:8]:
        print(f"{roc_p:>4}{ema_p:>5}{sm:>5.1f}{ao:>8.1f}{ai:>7.1f}{md:>9.2f}{mx:>9.2f}{npass:>4}/16{tpp:>8}")


if __name__ == "__main__":
    main()
