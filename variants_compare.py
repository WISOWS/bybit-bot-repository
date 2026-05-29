"""Compare improvement variants A-D for the bot-1 portfolio.

Same data window (2025-01 .. 2026-04), hybrid strategy, $100k,
fee 0.055%/side, slippage 0.05%/side, funding -0.01%/8h short.
"""

import math
import os
from typing import Any, Dict, List, Tuple

import portfolio_mtm as P
from backtest import load_ohlcv_csv
from multi_strategy_backtest import gen_hybrid, generate_trades

BASE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = [
    ("A: risk0.5% conc2", ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"], 0.005, 2),
    ("B: risk0.25% conc4", ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"], 0.0025, 4),
    ("C: risk0.5% conc4", ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"], 0.005, 4),
    ("D: 6 pairs risk0.25% conc3", ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT", "ENSUSDT", "SNXUSDT"], 0.0025, 3),
    ("baseline: risk0.25% conc2", ["NEARUSDT", "SOLUSDT", "LINKUSDT", "ENAUSDT"], 0.0025, 2),
]

ALL = sorted({s for _, syms, _, _ in VARIANTS for s in syms})
raw_by_symbol: Dict[str, List[Tuple]] = {}
close_maps: Dict[str, Dict[int, float]] = {}
for s in ALL:
    k1 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_60.csv"))
    k4 = load_ohlcv_csv(os.path.join(BASE, "ohlcv", f"{s}_240.csv"))
    raw_by_symbol[s] = generate_trades(gen_hybrid, k1, k4, min_4h=29)
    close_maps[s] = {int(r[0]): float(r[4]) for r in k1}


def mtm_dd(symbols: List[str], executed: List[Dict[str, Any]]) -> float:
    ts = set()
    for s in symbols:
        ts.update(close_maps[s].keys())
    fe = min(p["entry_ms"] for p in executed)
    lx = max(p["exit_ms"] for p in executed)
    grid = sorted(t for t in ts if fe <= t <= lx)
    return P.dd_stats(P.build_mtm_curve(executed, close_maps, grid))["max_dd_pct"]


rows = []
for label, syms, risk, conc in VARIANTS:
    P.SYMBOLS = syms
    P.RISK = risk
    ex, m = P.run_portfolio(raw_by_symbol, max_concurrent=conc)
    dd = mtm_dd(syms, ex)
    tpm = m["trades"] / (m["years"] * 12)
    pf = float("inf") if (isinstance(m["profit_factor"], float) and math.isinf(m["profit_factor"])) else m["profit_factor"]
    rows.append({
        "label": label, "annual": m["annual_return_pct"], "mtm_dd": dd, "pf": pf,
        "win": m["winrate_pct"], "tpm": round(tpm, 2), "trades": m["trades"],
        "skip_pct": m["skip_pct"], "final": m["final_balance"],
        "ratio": round(m["annual_return_pct"] / dd, 2) if dd > 0 else 0.0,
    })

h = f"{'variant':<28}{'annual%':>9}{'MTM_DD%':>9}{'PF':>7}{'win%':>7}{'tr/mo':>7}{'skip%':>7}{'ret/DD':>8}"
print("\n" + "=" * len(h))
print("BOT-1 PORTFOLIO IMPROVEMENT VARIANTS  (data 2025-01 .. 2026-04)")
print("=" * len(h))
print(h)
print("-" * len(h))
for r in rows:
    pf = "inf" if math.isinf(r["pf"]) else f"{r['pf']:.3f}"
    flag = "" if r["mtm_dd"] < 9.0 else "  <-- DD>=9% (fails)"
    print(f"{r['label']:<28}{r['annual']:>9.2f}{r['mtm_dd']:>9.2f}{pf:>7}{r['win']:>7.1f}"
          f"{r['tpm']:>7.2f}{r['skip_pct']:>7.2f}{r['ratio']:>8.2f}{flag}")

passing = [r for r in rows if r["mtm_dd"] < 9.0 and not r["label"].startswith("baseline")]
passing.sort(key=lambda r: r["ratio"], reverse=True)
print("\nЦель: MTM DD < 9%, лучшее соотношение доходность/DD (ret/DD)")
if passing:
    b = passing[0]
    print(f"  Лучший вариант: {b['label']}  ->  {b['annual']:.1f}% годовых, DD {b['mtm_dd']:.2f}%, "
          f"ret/DD {b['ratio']:.2f}, PF {b['pf'] if not math.isinf(b['pf']) else 'inf'}")
    print(f"  Все прошедшие DD<9% по ret/DD: " + " > ".join(f"{r['label'].split(':')[0]}({r['ratio']})" for r in passing))
else:
    print("  Ни один вариант не уложился в MTM DD < 9%")
