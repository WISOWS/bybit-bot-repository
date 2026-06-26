"""Monte-Carlo разорения: бэктест = ОДИН порядок сделок. Тасуем 2000 раз.

Берём реальные сделки baseline (OOS 2025+), извлекаем net-R каждой сделки
(с комиссиями/слиппеджем/фандингом — из портфельного симулятора при risk 0.1%),
затем компаундим в случайном порядке при разном риске. Считаем, в скольких
вселенных из 2000 ты обнуляешься/ловишь катастрофу, а не иксы.
"""

import os
import random

import portfolio_mtm as P
from multi_strategy_backtest import generate_trades
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy

random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"]
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))
N_SHUFFLE = 2000
RISKS = [0.005, 0.01, 0.02, 0.03]
# пороги катастрофы по просадке капитала
DD_RUIN = 80.0    # -80% = практически труп
DD_LIQ = 50.0     # -50% с плечом 5x в реале = почти наверняка ликвидация


def make_gen(extra):
    strat = make_regime_switch_strategy(RegimeSwitchParams(**extra))
    def gen(k1, i, k4, j4, is_new_4h):
        s = strat(k1, i, k4, j4)
        return None if s is None else (s.side, s.entry, s.stop, s.tp)
    return gen


def compound_path(Rs, f):
    """Последовательный компаундинг: eq *= (1 + f*R). Возвращает (final, maxDD%)."""
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for R in Rs:
        eq *= (1.0 + f * R)
        if eq <= 0:
            return 0.0, 100.0  # пробили ноль -> ликвидация
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100.0
        max_dd = max(max_dd, dd)
    return eq, max_dd


def pct(sorted_list, p):
    if not sorted_list:
        return 0.0
    i = max(0, min(len(sorted_list) - 1, int(p / 100.0 * len(sorted_list))))
    return sorted_list[i]


def main():
    P.SYMBOLS = SYMBOLS
    P.RISK = 0.001  # реф-риск: pnl ≈ линеен, компаундинг пренебрежим
    k1m, k4m, close_maps, ts_set = {}, {}, {}, set()
    for s in SYMBOLS:
        k1m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1m[s]}
        ts_set.update(close_maps[s].keys())

    raw = {s: generate_trades(make_gen({}), k1m[s], k4m[s], min_4h=29) for s in SYMBOLS}
    ex, m = P.run_portfolio(raw, window=OOS_WIN, max_concurrent=2)
    ref = 0.001 * P.INITIAL
    Rs = [p["pnl"] / ref for p in ex]   # net-R каждой реально исполненной сделки
    n = len(Rs)
    wins = sum(1 for R in Rs if R > 0)
    print(f"OOS deck: {n} сделок | WR {wins/n*100:.1f}% | "
          f"avgR {sum(Rs)/n:+.3f} | bestR {max(Rs):+.2f} | worstR {min(Rs):+.2f}", flush=True)

    # «как в бэктесте» — исходный порядок (тот самый лотерейный путь)
    print("\nИСХОДНЫЙ порядок (то, что показал бэктест):")
    for f in RISKS:
        fin, dd = compound_path(Rs, f)
        print(f"   risk {f*100:>4.1f}%:  x{fin:>10,.1f}   (maxDD {dd:.1f}%)")

    print("\n" + "=" * 96)
    print(f"MONTE-CARLO: {N_SHUFFLE} перетасовок порядка тех же сделок")
    print("=" * 96)
    hdr = (f"{'risk%':>6}{'медиана x':>12}{'5-й перц x':>12}{'худший x':>11}"
           f"{'>50%DD':>9}{'>80%DD':>9}{'ниже старта':>13}")
    print(hdr); print("-" * len(hdr))
    for f in RISKS:
        finals, liq, ruin, below = [], 0, 0, 0
        for _ in range(N_SHUFFLE):
            order = Rs[:]
            random.shuffle(order)
            fin, dd = compound_path(order, f)
            finals.append(fin)
            if dd >= DD_LIQ:
                liq += 1
            if dd >= DD_RUIN:
                ruin += 1
            if fin < 1.0:
                below += 1
        finals.sort()
        print(f"{f*100:>6.1f}{pct(finals,50):>12,.1f}{pct(finals,5):>12,.2f}"
              f"{finals[0]:>11,.2f}{liq/N_SHUFFLE*100:>8.1f}%{ruin/N_SHUFFLE*100:>8.1f}%"
              f"{below/N_SHUFFLE*100:>12.1f}%")

    print("\n>50%DD = с плечом 5x в реале почти наверняка ликвидация (полный ноль).")
    print(">80%DD = труп. 'ниже старта' = ты в минусе несмотря на 'плюсовую' стратегию.")


if __name__ == "__main__":
    main()
