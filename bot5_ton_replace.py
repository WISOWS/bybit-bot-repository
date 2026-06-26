"""Bot #5 TON-replacement walk-forward test.

TON показывает 7+ стопов подряд → ищем замену. Кандидаты (не занятые
другими ботами): XRP, ADA, BNB. Те же IS/OOS окна и движок, что в
final_portfolios.py. Risk 0.60%, concurrent 3, фильтр MTM DD < 8% на обоих.
"""

import math
import os

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))

RISK = 0.0060
CONC = 3

CANDIDATES = [
    ("OP+ARB+TON (текущий)", ["OPUSDT", "ARBUSDT", "TONUSDT"]),
    ("OP+ARB+XRP", ["OPUSDT", "ARBUSDT", "XRPUSDT"]),
    ("OP+ARB+ADA", ["OPUSDT", "ARBUSDT", "ADAUSDT"]),
    ("OP+ARB+BNB", ["OPUSDT", "ARBUSDT", "BNBUSDT"]),
]

IS_WIN = (P.to_ms("2023-06-01"), P.to_ms("2025-01-01"))   # до 2024-12
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))  # 2025-01+


def run(raw, cmaps, syms, win):
    P.SYMBOLS = list(syms)
    P.RISK = RISK
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=CONC)
    if not m:
        return None
    dd = mtm_dd_stats(ex, cmaps, tuple(syms))
    return {"annual": m["annual_return_pct"], "dd": dd["max_dd_pct"],
            "pf": m["profit_factor"], "win": m["winrate_pct"], "trades": m["trades"]}


def main():
    syms_all = sorted({s for _, ss in CANDIDATES for s in ss})
    raw, cmaps = {}, {}
    for s in syms_all:
        k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        raw[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
        cmaps[s] = {int(r[0]): float(r[4]) for r in k1}

    def pf(v):
        return "inf" if (isinstance(v, float) and math.isinf(v)) else f"{v:.2f}"

    hdr = (f"{'portfolio':<24}{'IS_an%':>8}{'IS_DD':>7}{'OOS_an%':>9}{'OOS_DD':>8}"
           f"{'PF':>6}{'OOS_win%':>9}{'OOS_tr':>7}{'PASS<8%':>9}")
    print("=" * len(hdr))
    print(f"BOT #5 TON-REPLACEMENT — risk {RISK*100:.2f}%, conc {CONC}, "
          f"regime_switch_hybrid, $100k, IS до 2024-12 / OOS 2025-01+")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, syms in CANDIDATES:
        ins = run(raw, cmaps, syms, IS_WIN)
        oos = run(raw, cmaps, syms, OOS_WIN)
        ok = (ins["dd"] < 8.0 and oos["dd"] < 8.0)
        print(f"{name:<24}{ins['annual']:>8.1f}{ins['dd']:>7.2f}{oos['annual']:>9.1f}"
              f"{oos['dd']:>8.2f}{pf(oos['pf']):>6}{oos['win']:>9.1f}{oos['trades']:>7}"
              f"{('YES' if ok else 'NO'):>9}")


if __name__ == "__main__":
    main()
