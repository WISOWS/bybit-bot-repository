"""Multi-strategy backtest sweep across all OHLCV pairs and risk levels.

Strategies:
  1. regime_switch_hybrid  (the live bot strategy, reused as-is)
  2. momentum_breakout      (1H N-bar high/low breakout, stop at opposite extreme, TP=2.5R)
  3. ema_crossover          (4H EMA 20/50 cross, stop under/over last swing, exit on reverse cross)
  4. bb_mean_reversion      (1H Bollinger 2.0 touch, TP = middle band)
  5. volatility_breakout    (ATR compression then 1H breakout, stop 1.5x ATR, TP=2.5R)

Risk levels: 0.25% / 0.5% / 0.75% / 1.0%.
Account: $100k, leverage 5x, taker fee 0.055%/side, slippage 0.05%/side,
funding -0.01%/8h on shorts.

Trades for a (strategy, symbol) pair do not depend on risk (sizing only scales
position size), so trades are generated once and replayed across risk levels.
"""

import argparse
import csv
import math
import os
import statistics
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from backtest import (
    BACKTEST_SLIPPAGE_RATE,
    BACKTEST_TAKER_FEE_RATE,
    apply_adverse_slippage,
    load_ohlcv_csv,
    simulate_exit,
)
from main import LEVERAGE, calc_atr_from_klines, calc_qty_by_margin, calc_qty_by_risk
from research_strategies import (
    candle_close,
    candle_high,
    candle_low,
    ema,
    recent_high,
    recent_low,
    regime_switch_hybrid,
)

FUNDING_RATE_PER_8H = 0.0001
FUNDING_INTERVAL_MS = 8 * 3600 * 1000
MS_PER_YEAR = 365.0 * 24 * 3600 * 1000
RISK_LEVELS = [0.0025, 0.005, 0.0075, 0.01]

# A raw trade is independent of risk/balance.
# (entry_ms, exit_ms, side, entry, stop, exit_price)
RawTrade = Tuple[int, int, str, float, float, float]


# -------------------- signal generators --------------------
# gen(klines_1h, idx_1h, klines_4h, idx_4h, is_new_4h)
#   -> (side, entry, stop, tp) | None
# tp is None  => signal-exit (handled as EMA reverse-cross exit)


def gen_hybrid(k1, i, k4, j4, is_new_4h):
    setup = regime_switch_hybrid(k1, i, k4, j4)
    if setup is None:
        return None
    return (setup.side, setup.entry, setup.stop, setup.tp)


def make_momentum_breakout(n: int = 20, rr: float = 2.5):
    def gen(k1, i, k4, j4, is_new_4h):
        if i < n:
            return None
        prev = k1[i - n:i]
        bh = max(candle_high(c) for c in prev)
        bl = min(candle_low(c) for c in prev)
        entry = candle_close(k1[i])
        if entry > bh:
            stop = bl
            if entry - stop <= 0:
                return None
            return ("Buy", entry, stop, entry + rr * (entry - stop))
        if entry < bl:
            stop = bh
            if stop - entry <= 0:
                return None
            return ("Sell", entry, stop, entry - rr * (stop - entry))
        return None
    return gen


def make_ema_crossover(fast: int = 20, slow: int = 50, swing: int = 10):
    def gen(k1, i, k4, j4, is_new_4h):
        if not is_new_4h or j4 < slow:
            return None
        now = [candle_close(c) for c in k4[j4 - slow + 1:j4 + 1]]
        prev = [candle_close(c) for c in k4[j4 - slow:j4]]
        ef, es = ema(now, fast), ema(now, slow)
        efp, esp = ema(prev, fast), ema(prev, slow)
        entry = candle_close(k1[i])
        if efp <= esp and ef > es:
            stop = recent_low(k4, j4, swing)
            if entry - stop <= 0:
                return None
            return ("Buy", entry, stop, None)
        if efp >= esp and ef < es:
            stop = recent_high(k4, j4, swing)
            if stop - entry <= 0:
                return None
            return ("Sell", entry, stop, None)
        return None
    return gen


def make_bb_mean_reversion(period: int = 20, mult: float = 2.0):
    def gen(k1, i, k4, j4, is_new_4h):
        if i < period:
            return None
        closes = [candle_close(c) for c in k1[i - period + 1:i + 1]]
        sma = sum(closes) / period
        sd = statistics.pstdev(closes)
        if sd <= 0:
            return None
        upper = sma + mult * sd
        lower = sma - mult * sd
        entry = candle_close(k1[i])
        lo = candle_low(k1[i])
        hi = candle_high(k1[i])
        if lo <= lower and entry < sma:
            stop = lower - 0.5 * sd
            if entry - stop <= 0 or sma - entry <= 0:
                return None
            return ("Buy", entry, stop, sma)
        if hi >= upper and entry > sma:
            stop = upper + 0.5 * sd
            if stop - entry <= 0 or entry - sma <= 0:
                return None
            return ("Sell", entry, stop, sma)
        return None
    return gen


def make_volatility_breakout(n: int = 10, atr_fast: int = 6, atr_slow: int = 20,
                             ratio: float = 0.8, rr: float = 2.5, stop_atr: float = 1.5):
    def gen(k1, i, k4, j4, is_new_4h):
        if i < max(n, atr_slow + 1):
            return None
        af = calc_atr_from_klines(k1[i - atr_fast:i + 1], atr_fast)
        as_ = calc_atr_from_klines(k1[i - atr_slow:i + 1], atr_slow)
        if as_ <= 0 or af / as_ >= ratio:
            return None
        prev = k1[i - n:i]
        bh = max(candle_high(c) for c in prev)
        bl = min(candle_low(c) for c in prev)
        entry = candle_close(k1[i])
        risk = stop_atr * as_
        if risk <= 0:
            return None
        if entry > bh:
            return ("Buy", entry, entry - risk, entry + rr * risk)
        if entry < bl:
            return ("Sell", entry, entry + risk, entry - rr * risk)
        return None
    return gen


# -------------------- exit handling --------------------


def ema_reverse_exit(k1, entry_index, side, stop, k4, times4h,
                     fast=20, slow=50) -> Tuple[float, int]:
    """Hold until stop hit (intrabar) or 4H EMA fast crosses back against the trade."""
    j4 = -1
    n4 = len(times4h)
    for j in range(entry_index + 1, len(k1)):
        t = int(k1[j][0])
        while j4 + 1 < n4 and times4h[j4 + 1] <= t:
            j4 += 1
        high = candle_high(k1[j])
        low = candle_low(k1[j])
        if side == "Buy" and low <= stop:
            return stop, j
        if side == "Sell" and high >= stop:
            return stop, j
        if j4 >= slow:
            closes = [candle_close(c) for c in k4[j4 - slow + 1:j4 + 1]]
            ef, es = ema(closes, fast), ema(closes, slow)
            if side == "Buy" and ef < es:
                return candle_close(k1[j]), j
            if side == "Sell" and ef > es:
                return candle_close(k1[j]), j
    last = len(k1) - 1
    return candle_close(k1[last]), last


def generate_trades(gen: Callable, k1: List[List[Any]], k4: List[List[Any]],
                    min_4h: int = 0) -> List[RawTrade]:
    times4h = [int(r[0]) for r in k4]
    raw: List[RawTrade] = []
    j4 = -1
    prev_j4 = -1
    i = 1
    n1 = len(k1)
    while i < n1 - 1:
        t = int(k1[i][0])
        while j4 + 1 < len(times4h) and times4h[j4 + 1] <= t:
            j4 += 1
        is_new_4h = j4 != prev_j4
        prev_j4 = j4
        if j4 < min_4h:
            i += 1
            continue
        sig = gen(k1, i, k4, j4, is_new_4h)
        if sig is None:
            i += 1
            continue
        side, entry, stop, tp = sig
        if abs(entry - stop) <= 0:
            i += 1
            continue
        if tp is not None:
            exit_price, _reason, xi = simulate_exit(k1, i, side, stop, tp)
        else:
            exit_price, xi = ema_reverse_exit(k1, i, side, stop, k4, times4h)
        raw.append((int(k1[i][0]), int(k1[xi][0]), side, entry, stop, exit_price))
        i = xi + 1
    return raw


# -------------------- metrics --------------------


def count_funding_periods(entry_ms: int, exit_ms: int) -> int:
    if exit_ms <= entry_ms:
        return 0
    first = (entry_ms // FUNDING_INTERVAL_MS) + 1
    last = exit_ms // FUNDING_INTERVAL_MS
    return max(0, int(last - first + 1))


def profit_factor(pnls: List[float]) -> float:
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def max_drawdown_pct(curve: List[float]) -> float:
    peak = curve[0] if curve else 0.0
    dd = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            dd = min(dd, (v - peak) / peak)
    return abs(dd) * 100.0


def daily_sharpe(events: List[Tuple[int, float]], start_ms: int, end_ms: int,
                 initial: float) -> float:
    if end_ms <= start_ms or not events:
        return 0.0
    day = 24 * 3600 * 1000
    n_days = int((end_ms - start_ms) // day)
    if n_days < 2:
        return 0.0
    balances = []
    ei = 0
    cur = initial
    for d in range(n_days + 1):
        t = start_ms + d * day
        while ei < len(events) and events[ei][0] <= t:
            cur = events[ei][1]
            ei += 1
        balances.append(cur)
    rets = [(balances[k] - balances[k - 1]) / balances[k - 1]
            for k in range(1, len(balances)) if balances[k - 1] > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    return (mean / sd) * math.sqrt(365.0) if sd > 0 else 0.0


def simulate_risk(raw: List[RawTrade], initial: float, risk: float,
                  span_start_ms: int, span_end_ms: int) -> Dict[str, Any]:
    balance = initial
    pnls: List[float] = []
    curve = [initial]
    events: List[Tuple[int, float]] = []
    longs = 0
    monthly: Dict[str, float] = {}

    for entry_ms, exit_ms, side, entry, stop, exit_price in raw:
        qty = min(
            calc_qty_by_risk(balance, risk, entry, stop),
            calc_qty_by_margin(balance, LEVERAGE, entry),
        )
        if qty <= 0:
            continue
        eff_entry = apply_adverse_slippage(entry, side, "entry")
        eff_exit = apply_adverse_slippage(exit_price, side, "exit")
        pnl_unit = (eff_exit - eff_entry) if side == "Buy" else (eff_entry - eff_exit)
        entry_fee = qty * eff_entry * BACKTEST_TAKER_FEE_RATE
        exit_fee = qty * eff_exit * BACKTEST_TAKER_FEE_RATE
        funding = 0.0
        if side == "Sell":
            funding = qty * eff_entry * FUNDING_RATE_PER_8H * count_funding_periods(entry_ms, exit_ms)
        pnl = qty * pnl_unit - entry_fee - exit_fee - funding
        balance += pnl
        pnls.append(pnl)
        curve.append(balance)
        events.append((exit_ms, balance))
        if side == "Buy":
            longs += 1
        mkey = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc).strftime("%Y-%m")
        monthly[mkey] = monthly.get(mkey, 0.0) + pnl

    n = len(pnls)
    years = (span_end_ms - span_start_ms) / MS_PER_YEAR
    if n == 0 or years <= 0:
        return {"trades": n, "annual_return_pct": 0.0, "max_dd_pct": 0.0,
                "winrate_pct": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
                "years": round(years, 2), "final_balance": round(balance, 2),
                "long_pct": 0.0, "short_pct": 0.0, "trades_per_month": 0.0,
                "best_month": "-", "best_month_pnl": 0.0,
                "worst_month": "-", "worst_month_pnl": 0.0}

    total_return = balance / initial
    annual = (total_return ** (1.0 / years) - 1.0) * 100.0 if total_return > 0 else -100.0
    wins = sum(1 for p in pnls if p > 0)
    pf = profit_factor(pnls)
    best = max(monthly.items(), key=lambda kv: kv[1])
    worst = min(monthly.items(), key=lambda kv: kv[1])
    return {
        "trades": n,
        "annual_return_pct": round(annual, 2),
        "max_dd_pct": round(max_drawdown_pct(curve), 2),
        "winrate_pct": round(wins / n * 100.0, 2),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else float("inf"),
        "sharpe": round(daily_sharpe(events, span_start_ms, span_end_ms, initial), 3),
        "years": round(years, 2),
        "final_balance": round(balance, 2),
        "long_pct": round(longs / n * 100.0, 1),
        "short_pct": round((n - longs) / n * 100.0, 1),
        "trades_per_month": round(n / (years * 12.0), 2),
        "best_month": best[0], "best_month_pnl": round(best[1], 2),
        "worst_month": worst[0], "worst_month_pnl": round(worst[1], 2),
    }


STRATEGIES: Dict[str, Tuple[Callable, int]] = {
    "regime_switch_hybrid": (gen_hybrid, 29),
    "momentum_breakout": (make_momentum_breakout(), 0),
    "ema_crossover": (make_ema_crossover(), 50),
    "bb_mean_reversion": (make_bb_mean_reversion(), 0),
    "volatility_breakout": (make_volatility_breakout(), 0),
}

CSV_FIELDS = [
    "strategy", "symbol", "risk_pct", "trades", "annual_return_pct", "max_dd_pct",
    "winrate_pct", "profit_factor", "sharpe", "years", "final_balance",
    "long_pct", "short_pct", "trades_per_month",
    "best_month", "best_month_pnl", "worst_month", "worst_month_pnl",
]


def pf_sort_key(r: Dict[str, Any]) -> float:
    pf = r["profit_factor"]
    return 1e9 if (isinstance(pf, float) and math.isinf(pf)) else pf


def main() -> None:
    parser = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--ohlcv-dir", default=os.path.join(base, "ohlcv"))
    parser.add_argument("--out", default=os.path.join(base, "backtest_results.csv"))
    parser.add_argument("--initial-balance", type=float, default=100_000.0)
    parser.add_argument("--min-annual-return", type=float, default=100.0)
    parser.add_argument("--max-dd", type=float, default=10.0)
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--min-years", type=float, default=1.0)
    parser.add_argument("--min-pf", type=float, default=2.0)
    args = parser.parse_args()

    files = os.listdir(args.ohlcv_dir)
    symbols = sorted({f[:-7] for f in files if f.endswith("_60.csv")})

    all_results: List[Dict[str, Any]] = []
    for symbol in symbols:
        f1 = os.path.join(args.ohlcv_dir, f"{symbol}_60.csv")
        f4 = os.path.join(args.ohlcv_dir, f"{symbol}_240.csv")
        if not (os.path.exists(f1) and os.path.exists(f4)):
            continue
        try:
            k1 = load_ohlcv_csv(f1)
            k4 = load_ohlcv_csv(f4)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {symbol}: {exc}")
            continue
        if len(k1) < 100 or len(k4) < 60:
            continue
        span_start, span_end = int(k1[0][0]), int(k1[-1][0])

        for sname, (gen, min4) in STRATEGIES.items():
            try:
                raw = generate_trades(gen, k1, k4, min_4h=min4)
            except Exception as exc:  # noqa: BLE001
                print(f"SKIP {symbol}/{sname}: {exc}")
                continue
            for risk in RISK_LEVELS:
                m = simulate_risk(raw, args.initial_balance, risk, span_start, span_end)
                row = {"strategy": sname, "symbol": symbol, "risk_pct": round(risk * 100, 2), **m}
                all_results.append(row)
        print(f"done {symbol}: {len(STRATEGIES)} strategies")

    # write full CSV
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            row = dict(r)
            if isinstance(row["profit_factor"], float) and math.isinf(row["profit_factor"]):
                row["profit_factor"] = "inf"
            w.writerow(row)

    valid = [r for r in all_results if r["trades"] > 0 and r["years"] > 0]
    matched = [
        r for r in valid
        if r["annual_return_pct"] > args.min_annual_return
        and r["max_dd_pct"] < args.max_dd
        and r["trades"] > args.min_trades
        and r["years"] > args.min_years
        and (math.isinf(r["profit_factor"]) or r["profit_factor"] > args.min_pf)
    ]
    matched.sort(key=pf_sort_key, reverse=True)

    hdr = (f"{'strategy':<22}{'symbol':<13}{'risk%':>6}{'trades':>7}{'annual%':>10}"
           f"{'maxDD%':>9}{'win%':>7}{'PF':>9}{'Sharpe':>8}{'yrs':>6}")

    print("\n" + "=" * 96)
    print(f"FILTER: annual>{args.min_annual_return}% maxDD<{args.max_dd}% "
          f"trades>{args.min_trades} span>{args.min_years}y PF>{args.min_pf}")
    print(f"MATCHED: {len(matched)} of {len(valid)} combinations")
    print("=" * 96)

    def show(rows: List[Dict[str, Any]]) -> None:
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            pf = "inf" if math.isinf(r["profit_factor"]) else f"{r['profit_factor']:.3f}"
            print(f"{r['strategy']:<22}{r['symbol']:<13}{r['risk_pct']:>6.2f}{r['trades']:>7}"
                  f"{r['annual_return_pct']:>10.2f}{r['max_dd_pct']:>9.2f}{r['winrate_pct']:>7.2f}"
                  f"{pf:>9}{r['sharpe']:>8.2f}{r['years']:>6.2f}")

    if matched:
        show(matched)
    else:
        top = sorted(valid, key=pf_sort_key, reverse=True)[:10]
        print("\nNo combination met all criteria. TOP-10 by Profit Factor (no filters):\n")
        show(top)

    print(f"\nFull results saved to {args.out} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
