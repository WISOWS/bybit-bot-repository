"""Свип anti–stop-hunting комбинаций на стратегии бота #1 (regime_switch_hybrid).

Ищем конфиг, который РОБАСТНО обыгрывает baseline: PF и WR выше И на IS, И на OOS,
DD<6% на обоих окнах. Топ-OOS без проверки IS = переобучение, не берём.
"""

import itertools
import os

import portfolio_mtm as P
from multi_strategy_backtest import generate_trades
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy

BASE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"]
RISK = 0.0025
MC = 2
IS_WIN = (P.to_ms("2020-01-01"), P.to_ms("2025-01-01"))
OOS_WIN = (P.to_ms("2025-01-01"), P.to_ms("2026-06-01"))
DD_CAP = 6.0

# grid
G_reclaim = [False, True]
G_buf = [0.0, 0.1, 0.2]
G_jit = [0.0, 0.2]
G_skip_round = [False, True]
G_avoid_round = [False, True]


def make_gen(extra):
    strat = make_regime_switch_strategy(RegimeSwitchParams(**extra))

    def gen(k1, i, k4, j4, is_new_4h):
        s = strat(k1, i, k4, j4)
        return None if s is None else (s.side, s.entry, s.stop, s.tp)

    return gen


def win_metrics(raw, close_maps, ts_set, win):
    ex, m = P.run_portfolio(raw, window=win, max_concurrent=MC)
    if not ex:
        return None
    fe = min(p["entry_ms"] for p in ex)
    lx = max(p["exit_ms"] for p in ex)
    grid = sorted(t for t in ts_set if fe <= t <= lx)
    dd = P.dd_stats(P.build_mtm_curve(ex, close_maps, grid))
    pf = m["profit_factor"]
    return {"wr": m["winrate_pct"], "pf": (1e9 if pf == float("inf") else pf),
            "annual": m["annual_return_pct"], "dd": dd["max_dd_pct"], "trades": m["trades"]}


def main():
    P.SYMBOLS = SYMBOLS
    P.RISK = RISK
    k1m, k4m, close_maps, ts_set = {}, {}, {}, set()
    for s in SYMBOLS:
        k1m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
        k4m[s] = P.load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
        close_maps[s] = {int(r[0]): float(r[4]) for r in k1m[s]}
        ts_set.update(close_maps[s].keys())

    combos = list(itertools.product(G_reclaim, G_buf, G_jit, G_skip_round, G_avoid_round))
    print(f"loaded {len(SYMBOLS)} pairs | {len(combos)} configs", flush=True)

    rows = []
    for n, (rec, buf, jit, skipr, avoidr) in enumerate(combos):
        extra = dict(require_sweep_reclaim=rec, sweep_buffer_atr=buf, sl_jitter_pct=jit,
                     skip_entry_near_round=skipr, avoid_round_numbers=avoidr)
        gen = make_gen(extra)
        raw = {s: generate_trades(gen, k1m[s], k4m[s], min_4h=29) for s in SYMBOLS}
        is_m = win_metrics(raw, close_maps, ts_set, IS_WIN)
        oos_m = win_metrics(raw, close_maps, ts_set, OOS_WIN)
        if is_m and oos_m:
            rows.append((extra, is_m, oos_m))
        if (n + 1) % 8 == 0:
            print(f"  {n+1}/{len(combos)} done", flush=True)

    # baseline = all-off
    base = next(r for r in rows if not any([r[0]["require_sweep_reclaim"], r[0]["sweep_buffer_atr"],
              r[0]["sl_jitter_pct"], r[0]["skip_entry_near_round"], r[0]["avoid_round_numbers"]]))
    b_is, b_oos = base[1], base[2]
    print(f"\nBASELINE  IS: WR {b_is['wr']:.1f} PF {b_is['pf']:.2f} DD {b_is['dd']:.2f} | "
          f"OOS: WR {b_oos['wr']:.1f} PF {b_oos['pf']:.2f} DD {b_oos['dd']:.2f} n={b_oos['trades']}")

    def tag(e):
        on = [k.replace("require_sweep_", "").replace("_entry_near", "").replace("sweep_", "")
              for k, v in e.items() if v]
        parts = []
        if e["require_sweep_reclaim"]: parts.append("reclaim")
        if e["sweep_buffer_atr"]: parts.append(f"buf{e['sweep_buffer_atr']}")
        if e["sl_jitter_pct"]: parts.append(f"jit{e['sl_jitter_pct']}")
        if e["skip_entry_near_round"]: parts.append("skipRound")
        if e["avoid_round_numbers"]: parts.append("avoidRound")
        return "+".join(parts) if parts else "baseline"

    # ROBUST filter: beats baseline on PF on BOTH windows, WR not worse on both, DD<cap both
    robust = []
    for e, i, o in rows:
        if i["dd"] > DD_CAP or o["dd"] > DD_CAP:
            continue
        if o["pf"] <= b_oos["pf"] or i["pf"] <= b_is["pf"]:
            continue
        if o["wr"] < b_oos["wr"] or i["wr"] < b_is["wr"]:
            continue
        # robust score = worst-window PF edge over baseline
        score = min(o["pf"] - b_oos["pf"], i["pf"] - b_is["pf"])
        robust.append((score, e, i, o))
    robust.sort(key=lambda x: -x[0])

    print("\n" + "=" * 104)
    print("РОБАСТНЫЕ (бьют baseline по PF+WR на ОБОИХ окнах, DD<6% на обоих) — отсортированы по худшему окну")
    print("=" * 104)
    hdr = f"{'config':<34}{'IS_WR':>7}{'IS_PF':>7}{'IS_DD':>7}{'OOS_WR':>8}{'OOS_PF':>7}{'OOS_DD':>7}{'OOS_ann':>9}{'n':>6}"
    print(hdr); print("-" * len(hdr))
    for _sc, e, i, o in robust[:12]:
        print(f"{tag(e):<34}{i['wr']:>7.1f}{i['pf']:>7.2f}{i['dd']:>7.2f}"
              f"{o['wr']:>8.1f}{o['pf']:>7.2f}{o['dd']:>7.2f}{o['annual']:>9.1f}{o['trades']:>6}")
    if not robust:
        print("  НИ ОДИН конфиг не обыграл baseline робастно на обоих окнах.")

    # also show raw top-OOS-PF (may be overfit) for honesty
    top_oos = sorted(rows, key=lambda r: -r[2]["pf"])[:6]
    print("\n" + "-" * 104)
    print("Топ по OOS PF (БЕЗ проверки IS — для контроля переобучения):")
    for e, i, o in top_oos:
        beats_is = "IS-ok" if i["pf"] > b_is["pf"] else "IS-WORSE"
        print(f"  {tag(e):<34} OOS_PF {o['pf']:.2f} WR {o['wr']:.1f} DD {o['dd']:.2f} | "
              f"IS_PF {i['pf']:.2f} -> {beats_is}")


if __name__ == "__main__":
    main()
