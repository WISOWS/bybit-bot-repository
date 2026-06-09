"""Bot #6 — honest (no look-ahead) new-strategy screen on long lookbacks.

tsmom at academic-scale lookbacks (10-28 days on 4H) + other OHLCV strategies.
Single-pair, risk 0.50%, IS/OOS, MTM DD. Reports best honest (strat,pair).
"""

import math
import os
import time

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


def single(raw1, cmaps, sym, win):
    P.SYMBOLS = [sym]; P.RISK = 0.0050
    ex, m = P.run_portfolio(raw1, window=win, max_concurrent=1)
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
        K[s] = S.build_symbol(k1, k4)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}
    print(f"loaded {len(POOL)} pairs", flush=True)

    configs = []
    # tsmom long lookbacks (4H bars): 60=10d,90=15d,120=20d,168=28d
    for roc_p in [60, 90, 120, 168]:
        for ema_p in [50, 100, 200]:
            for sm in [3.0, 4.0]:
                configs.append((f"tsmom_r{roc_p}e{ema_p}s{sm}", "tsmom",
                                {"roc_p": roc_p, "ema_p": ema_p, "atr_stop_mult": sm}))
    # tsmom long-only variants
    for roc_p in [90, 120]:
        configs.append((f"tsmomLO_r{roc_p}", "tsmom",
                        {"roc_p": roc_p, "ema_p": 100, "atr_stop_mult": 4.0, "long_only": True}))
    # supertrend / donchian with bigger/slower params (1H, no 4H lookahead)
    configs.append(("supertrend_14_4", "supertrend", {"st_period": 14, "st_mult": 4.0, "atr_stop_mult": 3.0}))
    configs.append(("donchian_55_20", "donchian", {"entry_n": 55, "exit_n": 20, "atr_stop_mult": 2.5}))
    configs.append(("chandelier_50", "chandelier", {"ema_fast": 20, "ema_slow": 50, "ch_mult": 3.0}))

    hdr = f"{'config':<22}{'avgOOS':>8}{'avgIS':>7}{'medOOSdd':>9}{'pass':>7}{'bestpairOOS':>12}{'tr/pair':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, strat, kw in configs:
        t0 = time.time()
        per = []
        ntr = 0
        for s in POOL:
            raw = {s: S.STRATS[strat](K[s], **kw)}
            ntr += len(raw[s])
            o = single(raw, cmaps, s, OOS_WIN)
            i = single(raw, cmaps, s, IS_WIN)
            if o and i:
                per.append((s, o, i))
        if not per:
            continue
        avg_oos = sum(o["annual"] for _, o, _ in per) / len(per)
        avg_is = sum(i["annual"] for _, _, i in per) / len(per)
        oosdd = sorted(o["dd"] for _, o, _ in per)
        med_dd = oosdd[len(oosdd) // 2]
        passing = [(s, o, i) for s, o, i in per if o["dd"] < DD_MAX and i["dd"] < DD_MAX and o["annual"] > 0]
        bestpair = max(per, key=lambda t: t[1]["annual"] if (t[1]["dd"] < DD_MAX and t[2]["dd"] < DD_MAX) else -1e9)
        bp = bestpair[1]["annual"] if (bestpair[1]["dd"] < DD_MAX and bestpair[2]["dd"] < DD_MAX) else float("nan")
        rows.append((name, avg_oos, avg_is, med_dd, len(passing), bestpair[0], bp, ntr // len(POOL)))
        print(f"{name:<22}{avg_oos:>8.1f}{avg_is:>7.1f}{med_dd:>9.2f}{len(passing):>5}/16"
              f"{bestpair[0].replace('USDT',''):>8} {bp:>4.0f}{ntr//len(POOL):>8}", flush=True)

    print("\nBest configs by avg OOS:")
    for r in sorted(rows, key=lambda r: r[1], reverse=True)[:8]:
        print(f"  {r[0]:<22} avgOOS={r[1]:.1f}% pass={r[4]}/16 bestpair={r[5].replace('USDT','')} {r[6]:.0f}%")


if __name__ == "__main__":
    main()
