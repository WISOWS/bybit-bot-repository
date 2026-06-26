"""Walk-forward тест anti–stop-hunting защит на стратегии бота #1.

Стратегия: regime_switch_hybrid (как у live-бота #1), пары NEAR/SOL/LINK/ENA.
risk 0.25%, max_concurrent 2. IS: ..2024-12, OOS: 2025-01+.
5 конфигов: baseline / sweep_buffer / jitter / sweep_reclaim / все три.
Метрики на каждое окно: WR, PF, annual %, MTM MaxDD %, trades.
"""

import os

import portfolio_mtm as P
from multi_strategy_backtest import generate_trades
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy

BASE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"]
RISK = 0.0025
MAX_CONCURRENT = 2

IS_WIN = (P.to_ms("2020-01-01"), P.to_ms("2025-01-01"))   # ..2024-12
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))  # 2025-01+

CONFIGS = [
    ("1. baseline", dict()),
    ("2. sweep_buffer_atr=0.4", dict(sweep_buffer_atr=0.4)),
    ("3. sl_jitter_pct=0.2", dict(sl_jitter_pct=0.2)),
    ("4. require_sweep_reclaim", dict(require_sweep_reclaim=True)),
    ("5. all three", dict(sweep_buffer_atr=0.4, sl_jitter_pct=0.2, require_sweep_reclaim=True)),
    ("6. reclaim+jitter", dict(sl_jitter_pct=0.2, require_sweep_reclaim=True)),
]


def make_gen(extra):
    strat = make_regime_switch_strategy(RegimeSwitchParams(**extra))

    def gen(k1, i, k4, j4, is_new_4h):
        setup = strat(k1, i, k4, j4)
        if setup is None:
            return None
        return (setup.side, setup.entry, setup.stop, setup.tp)

    return gen


def window_metrics(raw_by_symbol, close_maps, ts_set, win):
    ex, m = P.run_portfolio(raw_by_symbol, window=win, max_concurrent=MAX_CONCURRENT)
    if not ex:
        return None
    fe = min(p["entry_ms"] for p in ex)
    lx = max(p["exit_ms"] for p in ex)
    grid = sorted(t for t in ts_set if fe <= t <= lx)
    dd = P.dd_stats(P.build_mtm_curve(ex, close_maps, grid))
    return {
        "wr": m["winrate_pct"],
        "pf": m["profit_factor"],
        "annual": m["annual_return_pct"],
        "dd": dd["max_dd_pct"],
        "trades": m["trades"],
    }


def main():
    # configure portfolio globals to bot #1
    P.SYMBOLS = SYMBOLS
    P.RISK = RISK

    k1m, k4m, close_maps, ts_set = {}, {}, {}, set()
    for s in SYMBOLS:
        k1m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1m[s]}
        ts_set.update(close_maps[s].keys())
    print(f"loaded {len(SYMBOLS)} pairs: {', '.join(SYMBOLS)}", flush=True)

    def fmt(r):
        if r is None:
            return f"{'no trades':>48}"
        pf = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
        return (f"WR {r['wr']:5.1f}%  PF {pf:>5}  ann {r['annual']:7.1f}%  "
                f"DD {r['dd']:5.2f}%  n={r['trades']:>4}")

    print("\n" + "=" * 100)
    print(f"WALK-FORWARD anti-stop-hunt | regime_switch_hybrid | risk {RISK*100:.2f}% | "
          f"max_concurrent {MAX_CONCURRENT}")
    print("=" * 100)

    results = []
    for name, extra in CONFIGS:
        gen = make_gen(extra)
        raw = {s: generate_trades(gen, k1m[s], k4m[s], min_4h=29) for s in SYMBOLS}
        is_m = window_metrics(raw, close_maps, ts_set, IS_WIN)
        oos_m = window_metrics(raw, close_maps, ts_set, OOS_WIN)
        results.append((name, is_m, oos_m))
        print(f"\n{name}")
        print(f"   IS  : {fmt(is_m)}")
        print(f"   OOS : {fmt(oos_m)}")

    # --- OOS comparison table ---
    print("\n" + "=" * 100)
    print("OOS COMPARISON (2025-01+)  — цель: WR↑ и/или PF↑ без DD > 6%")
    print("=" * 100)
    hdr = f"{'config':<28}{'WR%':>7}{'PF':>7}{'annual%':>10}{'MaxDD%':>9}{'trades':>8}{'DD<6':>6}"
    print(hdr)
    print("-" * len(hdr))
    base = results[0][2]
    for name, _is_m, o in results:
        if o is None:
            print(f"{name:<28}{'no trades':>40}")
            continue
        pf = "inf" if o["pf"] == float("inf") else f"{o['pf']:.2f}"
        flag = "ok" if o["dd"] <= 6.0 else "X"
        print(f"{name:<28}{o['wr']:>7.1f}{pf:>7}{o['annual']:>10.1f}{o['dd']:>9.2f}"
              f"{o['trades']:>8}{flag:>6}")
    if base:
        print("-" * len(hdr))
        print(f"baseline OOS reference: WR {base['wr']:.1f}%  PF "
              f"{'inf' if base['pf']==float('inf') else f'{base['pf']:.2f}'}  "
              f"DD {base['dd']:.2f}%  n={base['trades']}")


if __name__ == "__main__":
    main()
