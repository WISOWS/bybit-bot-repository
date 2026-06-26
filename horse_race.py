"""Скачки стратегий: наша гибридка против 4 других семейств.

Один движок, одни пары (NEAR/SOL/LINK/ENA), risk 0.5%, conc 2, IS/OOS.
Вопрос: есть ли вообще что-то, что РОБАСТНО бьёт нашу regime_switch_hybrid?
"""

import os

import portfolio_mtm as P
from multi_strategy_backtest import (
    generate_trades,
    gen_hybrid,
    make_momentum_breakout,
    make_ema_crossover,
    make_bb_mean_reversion,
    make_volatility_breakout,
)

BASE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"]
RISK = 0.005
MC = 2
IS_WIN = (P.to_ms("2020-01-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))

CONTENDERS = [
    ("regime_switch_hybrid (НАША)", gen_hybrid, 29),
    ("momentum_breakout 20/2.5", make_momentum_breakout(20, 2.5), 50),
    ("ema_crossover 20/50", make_ema_crossover(20, 50, 10), 50),
    ("bb_mean_reversion 20/2.0", make_bb_mean_reversion(20, 2.0), 50),
    ("volatility_breakout", make_volatility_breakout(), 50),
]


def win(raw, close_maps, ts_set, w):
    ex, m = P.run_portfolio(raw, window=w, max_concurrent=MC)
    if not ex:
        return None
    fe = min(p["entry_ms"] for p in ex); lx = max(p["exit_ms"] for p in ex)
    grid = sorted(t for t in ts_set if fe <= t <= lx)
    dd = P.dd_stats(P.build_mtm_curve(ex, close_maps, grid))
    pf = m["profit_factor"]
    return {"wr": m["winrate_pct"], "pf": (1e9 if pf == float("inf") else pf),
            "ann": m["annual_return_pct"], "dd": dd["max_dd_pct"], "n": m["trades"]}


def main():
    P.SYMBOLS = SYMBOLS
    P.RISK = RISK
    k1m, k4m, close_maps, ts_set = {}, {}, {}, set()
    for s in SYMBOLS:
        k1m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1m[s]}
        ts_set.update(close_maps[s].keys())
    print(f"loaded {len(SYMBOLS)} pairs | risk {RISK*100:.2f}% | conc {MC}\n", flush=True)

    def fmt(r):
        if not r:
            return "no trades"
        pf = "inf" if r["pf"] >= 1e9 else f"{r['pf']:.2f}"
        return f"WR {r['wr']:5.1f}  PF {pf:>5}  ann {r['ann']:7.1f}%  DD {r['dd']:6.2f}%  n={r['n']:>4}"

    print("=" * 100)
    print(f"{'СТРАТЕГИЯ':<30}{'окно':<6}{'результат'}")
    print("=" * 100)
    rows = []
    for name, gen, min4 in CONTENDERS:
        raw = {s: generate_trades(gen, k1m[s], k4m[s], min_4h=min4) for s in SYMBOLS}
        i = win(raw, close_maps, ts_set, IS_WIN)
        o = win(raw, close_maps, ts_set, OOS_WIN)
        rows.append((name, i, o))
        print(f"{name:<30}{'IS':<6}{fmt(i)}")
        print(f"{'':<30}{'OOS':<6}{fmt(o)}")
        print("-" * 100)

    # robust verdict: положителен и DD<15% на ОБОИХ окнах, ранжируем по min(PF_is, PF_oos)
    print("\nРОБАСТНЫЙ РЕЙТИНГ (положителен + DD<15% на ОБОИХ окнах; ключ = худшее окно по PF):")
    ranked = []
    for name, i, o in rows:
        if not i or not o:
            continue
        survives = i["ann"] > 0 and o["ann"] > 0 and i["dd"] < 15 and o["dd"] < 15
        worst_pf = min(i["pf"], o["pf"])
        ranked.append((survives, worst_pf, name, i, o))
    ranked.sort(key=lambda x: (-x[0], -x[1]))
    for k, (surv, wpf, name, i, o) in enumerate(ranked, 1):
        mark = "OK " if surv else "X  "
        print(f"  {k}. {mark}{name:<30} worstPF {wpf:.2f} | "
              f"IS PF {i['pf']:.2f}/DD {i['dd']:.1f}  OOS PF {o['pf']:.2f}/DD {o['dd']:.1f}/ann {o['ann']:.0f}%")


if __name__ == "__main__":
    main()
