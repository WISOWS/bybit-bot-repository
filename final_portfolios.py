"""Final summary: all 5 bots, full-period + walk-forward (IS/OOS) with MTM DD.
Each bot runs hybrid on its production params."""

import math
import os
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))

BOTS = [
    ("#1", ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"], 0.0025, 2),
    ("#2", ["ONDOUSDT", "ZECUSDT", "SUIUSDT"], 0.007, 3),
    ("#3", ["ATOMUSDT", "INJUSDT", "TIAUSDT"], 0.007, 3),
    ("#4", ["DOTUSDT", "AVAXUSDT", "RUNEUSDT"], 0.0065, 3),
    ("#5", ["ETCUSDT", "XLMUSDT", "ALGOUSDT"], 0.005, 3),
]
IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))


def run(raw, cmaps, syms, risk, conc, win):
    P.SYMBOLS = list(syms); P.RISK = risk
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=conc)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, tuple(syms))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"], "pf": m["profit_factor"],
            "win": m["winrate_pct"], "trades": m["trades"]}


def main():
    syms_all = sorted({s for _, ss, _, _ in BOTS for s in ss})
    raw, cmaps = {}, {}
    for s in syms_all:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}

    def pf(v):
        return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"

    hdr = (f"{'bot':<4}{'pairs':<34}{'risk%':>6}{'conc':>5}{'FULL_an%':>9}{'FULL_DD':>8}"
           f"{'OOS_an%':>9}{'OOS_DD':>8}{'IS_an%':>8}{'IS_DD':>7}{'PF':>6}")
    print("=" * len(hdr))
    print("FINAL — 5 BOTS (regime_switch_hybrid, $100k each, fee 0.055%, slip 0.05%, funding -0.01%/8h short)")
    print("=" * len(hdr))
    print(hdr); print("-" * len(hdr))
    for name, syms, risk, conc in BOTS:
        full = run(raw, cmaps, syms, risk, conc, None)
        oos = run(raw, cmaps, syms, risk, conc, OOS_WIN)
        ins = run(raw, cmaps, syms, risk, conc, IS_WIN)
        pairs = "+".join(s.replace("USDT", "") for s in syms)
        print(f"{name:<4}{pairs:<34}{risk*100:>6.2f}{conc:>5}{full['annual']:>9.1f}{full['dd']:>8.2f}"
              f"{oos['annual']:>9.1f}{oos['dd']:>8.2f}{ins['annual']:>8.1f}{ins['dd']:>7.2f}{pf(full['pf']):>6}")


if __name__ == "__main__":
    main()
