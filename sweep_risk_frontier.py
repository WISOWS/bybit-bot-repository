"""Где реально лежат 300-500%: фронтир risk/concurrent, не поиск стратегии.

Edge (PF) фиксирован — крутим ТОЛЬКО размер риска и число позиций.
Стратегия: regime_switch_hybrid, baseline и reclaim+skipRound. OOS 2025-01+.
Показывает annual% vs MTM MaxDD% — честный размен доходность/риск.
"""

import os

import portfolio_mtm as P
from multi_strategy_backtest import generate_trades
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy

BASE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"]
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))
IS_WIN = (P.to_ms("2020-01-01"), P.to_ms("2025-01-01"))

RISKS = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03]
CONCURR = [2, 3, 5]

VARIANTS = {
    "baseline": dict(),
    "reclaim+skipRound": dict(require_sweep_reclaim=True, skip_entry_near_round=True),
}


def make_gen(extra):
    strat = make_regime_switch_strategy(RegimeSwitchParams(**extra))
    def gen(k1, i, k4, j4, is_new_4h):
        s = strat(k1, i, k4, j4)
        return None if s is None else (s.side, s.entry, s.stop, s.tp)
    return gen


def win(raw, close_maps, ts_set, w, mc):
    ex, m = P.run_portfolio(raw, window=w, max_concurrent=mc)
    if not ex:
        return None
    fe = min(p["entry_ms"] for p in ex); lx = max(p["exit_ms"] for p in ex)
    grid = sorted(t for t in ts_set if fe <= t <= lx)
    dd = P.dd_stats(P.build_mtm_curve(ex, close_maps, grid))
    return {"ann": m["annual_return_pct"], "dd": dd["max_dd_pct"], "wr": m["winrate_pct"],
            "pf": m["profit_factor"], "n": m["trades"], "fb": m["final_balance"]}


def main():
    P.SYMBOLS = SYMBOLS
    k1m, k4m, close_maps, ts_set = {}, {}, {}, set()
    for s in SYMBOLS:
        k1m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1m[s]}
        ts_set.update(close_maps[s].keys())
    print(f"loaded {len(SYMBOLS)} pairs\n", flush=True)

    for vname, extra in VARIANTS.items():
        raw = {s: generate_trades(make_gen(extra), k1m[s], k4m[s], min_4h=29) for s in SYMBOLS}
        print("=" * 92)
        print(f"{vname}   (edge fixed, dialing risk & concurrent)  — OOS 2025-01+")
        print("=" * 92)
        print(f"{'risk%':>6}{'conc':>6}{'OOS_ann%':>11}{'OOS_DD%':>10}{'ret/DD':>9}{'WR%':>7}{'PF':>7}{'n':>6}{'$100k->':>14}")
        print("-" * 82)
        for mc in CONCURR:
            for r in RISKS:
                P.RISK = r
                o = win(raw, close_maps, ts_set, OOS_WIN, mc)
                if not o:
                    continue
                pf = "inf" if o["pf"] == float("inf") else f"{o['pf']:.2f}"
                ratio = o["ann"] / o["dd"] if o["dd"] > 0 else 0.0
                print(f"{r*100:>6.2f}{mc:>6}{o['ann']:>11.1f}{o['dd']:>10.2f}{ratio:>9.2f}"
                      f"{o['wr']:>7.1f}{pf:>7}{o['n']:>6}{o['fb']:>14,.0f}")
        print()


if __name__ == "__main__":
    main()
