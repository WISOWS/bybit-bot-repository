"""Compare bots #2-#5 at risk 0.50% vs current, plus find max risk where MTM DD<8% on both windows."""

import math
import os
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))
BOTS = [
    ("#2", ["ONDOUSDT", "ZECUSDT", "SUIUSDT"], 0.007),
    ("#3", ["ATOMUSDT", "INJUSDT", "TIAUSDT"], 0.007),
    ("#4", ["DOTUSDT", "AVAXUSDT", "RUNEUSDT"], 0.0065),
    ("#5", ["OPUSDT", "ARBUSDT", "TONUSDT"], 0.007),
]
CONC = 3
TARGET_RISK = 0.005
DD_LIMIT = 8.0
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))


def pfx(v):
    return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"


def run(raw, cmaps, syms, risk, win):
    P.SYMBOLS = list(syms); P.RISK = risk
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=CONC)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, tuple(syms))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"],
            "pf": m["profit_factor"], "win": m["winrate_pct"], "trades": m["trades"]}


def main():
    syms_all = sorted({s for _, ss, _ in BOTS for s in ss})
    raw, cmaps = {}, {}
    for s in syms_all:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}

    # 1) Risk 0.50% — детальные метрики full/IS/OOS
    print("=" * 102)
    print("BOTS @ risk 0.50% (max_concurrent 3, $100k)")
    print("=" * 102)
    hdr = (f"{'bot':<4}{'pairs':<26}{'period':<5}{'annual%':>9}{'MTM_DD%':>9}{'PF':>6}"
           f"{'win%':>7}{'trades':>8}")
    print(hdr); print("-" * len(hdr))
    rows05 = {}
    for name, syms, _ in BOTS:
        rows05[name] = {
            "full": run(raw, cmaps, syms, TARGET_RISK, None),
            "is": run(raw, cmaps, syms, TARGET_RISK, IS_WIN),
            "oos": run(raw, cmaps, syms, TARGET_RISK, OOS_WIN),
        }
        pairs = "+".join(s.replace("USDT", "") for s in syms)
        for period in ["full", "is", "oos"]:
            r = rows05[name][period]
            print(f"{name:<4}{pairs:<26}{period:<5}{r['annual']:>9.1f}{r['dd']:>9.2f}{pfx(r['pf']):>6}{r['win']:>7.1f}{r['trades']:>8}")
        print()

    # 2) Сравнение 0.50% vs текущий риск
    print("=" * 102)
    print(f"COMPARISON: 0.50% vs current risk  (FULL annual / FULL DD / OOS annual / OOS DD)")
    print("=" * 102)
    h2 = f"{'bot':<4}{'pairs':<26}{'cur_r%':>7}{'cur_an':>8}{'cur_DD':>8}{'cur_OOSan':>10}{'cur_OOSDD':>10}  |  {'0.50_an':>8}{'0.50_DD':>8}{'0.50_OOSan':>11}{'0.50_OOSDD':>11}"
    print(h2); print("-" * len(h2))
    for name, syms, cur_risk in BOTS:
        cur_full = run(raw, cmaps, syms, cur_risk, None)
        cur_oos = run(raw, cmaps, syms, cur_risk, OOS_WIN)
        r05 = rows05[name]
        pairs = "+".join(s.replace("USDT", "") for s in syms)
        print(f"{name:<4}{pairs:<26}{cur_risk*100:>7.2f}{cur_full['annual']:>8.1f}{cur_full['dd']:>8.2f}{cur_oos['annual']:>10.1f}{cur_oos['dd']:>10.2f}"
              f"  |  {r05['full']['annual']:>8.1f}{r05['full']['dd']:>8.2f}{r05['oos']['annual']:>11.1f}{r05['oos']['dd']:>11.2f}")

    # 3) Подбор риска для MTM DD<8% на обоих окнах
    print()
    print("=" * 102)
    print(f"RISK SWEEP — макс. риск, при котором MTM DD < {DD_LIMIT}% на ОБОИХ окнах (IS и OOS)")
    print("=" * 102)
    h3 = f"{'bot':<4}{'best_risk%':>11}{'OOS_an%':>9}{'OOS_DD':>8}{'IS_an%':>8}{'IS_DD':>7}{'full_an%':>9}{'full_DD':>9}{'PF':>6}"
    print(h3); print("-" * len(h3))
    grid = [round(0.0030 + 0.0005 * k, 4) for k in range(11)]  # 0.30% .. 0.80%
    for name, syms, cur_risk in BOTS:
        best = None
        for r in grid:
            i = run(raw, cmaps, syms, r, IS_WIN)
            o = run(raw, cmaps, syms, r, OOS_WIN)
            if i and o and i["dd"] < DD_LIMIT and o["dd"] < DD_LIMIT:
                best = (r, i, o)
        if best:
            r, i, o = best
            f = run(raw, cmaps, syms, r, None)
            print(f"{name:<4}{r*100:>11.2f}{o['annual']:>9.1f}{o['dd']:>8.2f}{i['annual']:>8.1f}{i['dd']:>7.2f}{f['annual']:>9.1f}{f['dd']:>9.2f}{pfx(f['pf']):>6}")
        else:
            print(f"{name:<4}{'нет':>11}  -- ни один риск из {grid[0]*100:.2f}-{grid[-1]*100:.2f}% не даёт DD<{DD_LIMIT}% на обоих окнах")


if __name__ == "__main__":
    main()
