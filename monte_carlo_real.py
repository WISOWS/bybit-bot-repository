"""Честный ruin-тест: убираем 3 фантазии идеальной модели.

1. БУТСТРАП: тянем n сделок из колоды С ВОЗВРАТОМ -> будущее не обязано повторить
   золотую выборку 2025 (avgR +0.69). Каждая вселенная = своя реализация edge.
2. ХВОСТОВОЙ ГЭП: с вероятностью p_gap проигрышная сделка проскакивает стоп
   (гэп/фитиль/флэш-краш при плече 5x) -> R *= gap_mult.
3. ДЕГРАДАЦИЯ EDGE: отдельный блок — что если будущий avgR ниже золотого.
"""

import os
import random

import portfolio_mtm as P
from multi_strategy_backtest import generate_trades
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy

random.seed(7)
BASE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"]
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))
N = 5000
RISKS = [0.005, 0.01, 0.02, 0.03]
P_GAP = 0.03          # 3% стопов проскакивают
GAP_MULT = 3.0        # проскок = убыток в 3x от планового стопа
DD_LIQ, DD_RUIN = 50.0, 80.0


def make_gen(extra):
    strat = make_regime_switch_strategy(RegimeSwitchParams(**extra))
    def gen(k1, i, k4, j4, is_new_4h):
        s = strat(k1, i, k4, j4)
        return None if s is None else (s.side, s.entry, s.stop, s.tp)
    return gen


def boot_path(deck, n, f, p_gap, gap_mult, edge_shift=0.0):
    eq, peak, max_dd = 1.0, 1.0, 0.0
    for _ in range(n):
        R = random.choice(deck) + edge_shift
        if R < 0 and random.random() < p_gap:
            R *= gap_mult
        eq *= (1.0 + f * R)
        if eq <= 0:
            return 0.0, 100.0
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100.0)
    return eq, max_dd


def pctl(s, p):
    return s[max(0, min(len(s) - 1, int(p / 100.0 * len(s))))]


def run_block(deck, n, title, p_gap, gap_mult, edge_shift=0.0):
    print("\n" + "=" * 98)
    print(title)
    print("=" * 98)
    hdr = (f"{'risk%':>6}{'медиана x':>12}{'5й перц':>10}{'1й перц':>10}{'худший':>9}"
           f"{'>50%DD':>9}{'>80%DD':>9}{'ниже старта':>13}")
    print(hdr); print("-" * len(hdr))
    for f in RISKS:
        finals, liq, ruin, below = [], 0, 0, 0
        for _ in range(N):
            fin, dd = boot_path(deck, n, f, p_gap, gap_mult, edge_shift)
            finals.append(fin)
            liq += dd >= DD_LIQ
            ruin += dd >= DD_RUIN
            below += fin < 1.0
        finals.sort()
        print(f"{f*100:>6.1f}{pctl(finals,50):>12,.2f}{pctl(finals,5):>10,.2f}"
              f"{pctl(finals,1):>10,.2f}{finals[0]:>9,.2f}{liq/N*100:>8.1f}%"
              f"{ruin/N*100:>8.1f}%{below/N*100:>12.1f}%")


def main():
    P.SYMBOLS = SYMBOLS
    P.RISK = 0.001
    k1m, k4m, close_maps, ts_set = {}, {}, {}, set()
    for s in SYMBOLS:
        k1m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1m[s]}
        ts_set.update(close_maps[s].keys())
    raw = {s: generate_trades(make_gen({}), k1m[s], k4m[s], min_4h=29) for s in SYMBOLS}
    ex, _ = P.run_portfolio(raw, window=OOS_WIN, max_concurrent=2)
    ref = 0.001 * P.INITIAL
    deck = [p["pnl"] / ref for p in ex]
    n = len(deck)
    avg = sum(deck) / n
    print(f"OOS deck: {n} сделок | avgR {avg:+.3f} (золотая выборка 2025)")

    run_block(deck, n, "А) БУТСТРАП, без гэпов (только неуверенность в выборке edge)",
              p_gap=0.0, gap_mult=1.0)
    run_block(deck, n, f"Б) БУТСТРАП + хвостовой гэп ({int(P_GAP*100)}% стопов проскакивают x{GAP_MULT})",
              p_gap=P_GAP, gap_mult=GAP_MULT)
    # деградация edge: сдвигаем каждый R так, чтобы avgR упал с +0.69 до ~+0.20
    shift = 0.20 - avg
    run_block(deck, n, f"В) БУТСТРАП + гэп + ДЕГРАДАЦИЯ edge (avgR +0.69 -> +0.20, нормальный режим)",
              p_gap=P_GAP, gap_mult=GAP_MULT, edge_shift=shift)

    print("\n>50%DD: с плечом 5x в реале это почти всегда ликвидация = полный ноль.")


if __name__ == "__main__":
    main()
