"""Bot #6 — new candidate strategies + trailing-exit engine.

Each strategy produces RAW trades compatible with portfolio_mtm.run_portfolio:
    (entry_ms, exit_ms, side, entry, init_stop, exit_price)
where `init_stop` is the stop used at entry for risk-based position sizing
(calc_qty_by_risk), and `exit_price` is the realized exit after any trailing.

Engine features the existing framework lacks:
  - per-1H-bar trailing stops (ATR/Chandelier/Supertrend lines, breakeven moves)
  - 4H regime/indicator values forward-filled onto 1H bars
  - intrabar stop/TP priority (stop checked before TP, conservative)

Indicators implemented from scratch on OHLCV only: EMA, SMA, Wilder ATR, RSI,
Bollinger, Donchian, Supertrend, ADX, Keltner, ROC.

Account/cost assumptions are applied later by portfolio_mtm (fee 0.055%/side,
slip 0.05%/side, funding -0.01%/8h short, $100k, lev 5x).
"""

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------- low-level kline access ----------------

def _col(klines: List[List[Any]], idx: int) -> List[float]:
    return [float(r[idx]) for r in klines]


def to_arrays(klines: List[List[Any]]):
    ts = [int(r[0]) for r in klines]
    o = _col(klines, 1); h = _col(klines, 2); l = _col(klines, 3); c = _col(klines, 4)
    return ts, o, h, l, c


# ---------------- indicators (return list aligned to input) ----------------

def ema_series(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append((v - out[-1]) * k + out[-1])
    return out


def sma_series(values: List[float], period: int) -> List[float]:
    out = [float("nan")] * len(values)
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= period:
            s -= values[i - period]
        if i >= period - 1:
            out[i] = s / period
    return out


def stdev_series(values: List[float], period: int) -> List[float]:
    out = [float("nan")] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        m = sum(window) / period
        out[i] = math.sqrt(sum((x - m) ** 2 for x in window) / period)
    return out


def wilder_atr(h: List[float], l: List[float], c: List[float], period: int) -> List[float]:
    n = len(c)
    tr = [0.0] * n
    for i in range(n):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = [float("nan")] * n
    if n >= period:
        first = sum(tr[1:period + 1]) / period if n > period else sum(tr[:period]) / period
        atr[period - 1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def rsi_series(c: List[float], period: int) -> List[float]:
    n = len(c)
    out = [float("nan")] * n
    if n <= period:
        return out
    gains = 0.0; losses = 0.0
    for i in range(1, period + 1):
        ch = c[i] - c[i - 1]
        gains += max(ch, 0.0); losses += max(-ch, 0.0)
    avg_g = gains / period; avg_l = losses / period
    out[period] = 100.0 - 100.0 / (1.0 + (avg_g / avg_l if avg_l > 0 else 1e9))
    for i in range(period + 1, n):
        ch = c[i] - c[i - 1]
        avg_g = (avg_g * (period - 1) + max(ch, 0.0)) / period
        avg_l = (avg_l * (period - 1) + max(-ch, 0.0)) / period
        rs = avg_g / avg_l if avg_l > 0 else 1e9
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def donchian(h: List[float], l: List[float], period: int):
    n = len(h)
    up = [float("nan")] * n; dn = [float("nan")] * n
    for i in range(period - 1, n):
        up[i] = max(h[i - period + 1:i + 1])
        dn[i] = min(l[i - period + 1:i + 1])
    return up, dn


def supertrend(h: List[float], l: List[float], c: List[float], period: int, mult: float):
    """Return (st_line, direction) where direction=+1 uptrend, -1 downtrend."""
    n = len(c)
    atr = wilder_atr(h, l, c, period)
    st = [float("nan")] * n
    dir_ = [0] * n
    final_ub = [float("nan")] * n
    final_lb = [float("nan")] * n
    for i in range(n):
        if math.isnan(atr[i]):
            continue
        hl2 = (h[i] + l[i]) / 2.0
        bub = hl2 + mult * atr[i]
        blb = hl2 - mult * atr[i]
        if i == 0 or math.isnan(final_ub[i - 1]):
            final_ub[i] = bub; final_lb[i] = blb; dir_[i] = 1; st[i] = blb
            continue
        final_ub[i] = bub if (bub < final_ub[i - 1] or c[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = blb if (blb > final_lb[i - 1] or c[i - 1] < final_lb[i - 1]) else final_lb[i - 1]
        if c[i] > final_ub[i]:
            dir_[i] = 1
        elif c[i] < final_lb[i]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
        st[i] = final_lb[i] if dir_[i] == 1 else final_ub[i]
    return st, dir_


def adx_series(h: List[float], l: List[float], c: List[float], period: int) -> List[float]:
    n = len(c)
    out = [float("nan")] * n
    if n < 2 * period:
        return out
    plus_dm = [0.0] * n; minus_dm = [0.0] * n; tr = [0.0] * n
    for i in range(1, n):
        up = h[i] - h[i - 1]; dn = l[i - 1] - l[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = sum(tr[1:period + 1])
    pdm = sum(plus_dm[1:period + 1]); mdm = sum(minus_dm[1:period + 1])
    dx_list = []
    for i in range(period + 1, n):
        atr = atr - atr / period + tr[i]
        pdm = pdm - pdm / period + plus_dm[i]
        mdm = mdm - mdm / period + minus_dm[i]
        if atr <= 0:
            continue
        pdi = 100 * pdm / atr; mdi = 100 * mdm / atr
        dx = 100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) > 0 else 0.0
        dx_list.append((i, dx))
    if len(dx_list) >= period:
        adx = sum(d for _, d in dx_list[:period]) / period
        out[dx_list[period - 1][0]] = adx
        for k in range(period, len(dx_list)):
            i, dx = dx_list[k]
            adx = (adx * (period - 1) + dx) / period
            out[i] = adx
    return out


def roc_series(c: List[float], period: int) -> List[float]:
    n = len(c)
    out = [float("nan")] * n
    for i in range(period, n):
        if c[i - period] > 0:
            out[i] = (c[i] / c[i - period] - 1.0) * 100.0
    return out


# ---------------- 4H -> 1H alignment ----------------

def map_1h_to_4h(ts1: List[int], ts4: List[int]) -> List[int]:
    """For each 1H bar, index of the most recent CLOSED 4H bar (<= bar time)."""
    out = [-1] * len(ts1)
    j = -1
    n4 = len(ts4)
    for i, t in enumerate(ts1):
        while j + 1 < n4 and ts4[j + 1] <= t:
            j += 1
        out[i] = j
    return out


# ---------------- trailing exit simulator ----------------

def simulate_trailing(ts1, h1, l1, c1, entry_i: int, side: str,
                      init_stop: float, tp: Optional[float],
                      trail_line: Optional[List[float]] = None,
                      flip: Optional[List[bool]] = None,
                      breakeven_R: Optional[float] = None,
                      atr1: Optional[List[float]] = None,
                      atr_trail_mult: Optional[float] = None,
                      max_bars: Optional[int] = None) -> Tuple[int, float]:
    """Walk 1H bars from entry. Returns (exit_index, exit_price).
    Priority each bar: stop (intrabar) -> tp (intrabar) -> flip/trail-cross (close) -> timeout.
    Trailing stop = max(running) of: trail_line value, chandelier (close-atr*mult), breakeven.
    """
    entry = c1[entry_i]
    risk = abs(entry - init_stop)
    stop = init_stop
    n = len(c1)
    end = n if max_bars is None else min(n, entry_i + 1 + max_bars)
    be_done = False
    for j in range(entry_i + 1, end):
        # update trailing stop BEFORE testing this bar (uses info up to prior close)
        new_stop = stop
        if trail_line is not None and not math.isnan(trail_line[j - 1]):
            tl = trail_line[j - 1]
            new_stop = max(new_stop, tl) if side == "Buy" else min(new_stop, tl)
        if atr1 is not None and atr_trail_mult is not None and not math.isnan(atr1[j - 1]):
            ch = c1[j - 1] - atr_trail_mult * atr1[j - 1] if side == "Buy" else c1[j - 1] + atr_trail_mult * atr1[j - 1]
            new_stop = max(new_stop, ch) if side == "Buy" else min(new_stop, ch)
        if breakeven_R is not None and not be_done:
            fav = (c1[j - 1] - entry) if side == "Buy" else (entry - c1[j - 1])
            if risk > 0 and fav >= breakeven_R * risk:
                be = entry  # move to breakeven
                new_stop = max(new_stop, be) if side == "Buy" else min(new_stop, be)
                be_done = True
        stop = new_stop
        hi = h1[j]; lo = l1[j]
        if side == "Buy":
            if lo <= stop:
                return j, stop
            if tp is not None and hi >= tp:
                return j, tp
        else:
            if hi >= stop:
                return j, stop
            if tp is not None and lo <= tp:
                return j, tp
        if flip is not None and flip[j]:
            return j, c1[j]
    last = end - 1
    return last, c1[last]


# ---------------- strategy generators ----------------
# Signature: gen(sym_data) -> List[raw_trade]
# sym_data = dict with precomputed 1H/4H arrays.

def build_symbol(k1: List[List[Any]], k4: List[List[Any]]) -> Dict[str, Any]:
    ts1, o1, h1, l1, c1 = to_arrays(k1)
    ts4, o4, h4, l4, c4 = to_arrays(k4)
    m = map_1h_to_4h(ts1, ts4)
    return {"ts1": ts1, "o1": o1, "h1": h1, "l1": l1, "c1": c1,
            "ts4": ts4, "o4": o4, "h4": h4, "l4": l4, "c4": c4, "map4": m,
            "k1": k1, "k4": k4}


RawTrade = Tuple[int, int, str, float, float, float]


def _emit(d, entry_i, exit_i, side, entry, init_stop, exit_price) -> RawTrade:
    return (d["ts1"][entry_i], d["ts1"][exit_i], side, entry, init_stop, exit_price)


# --- 1. Supertrend trend-follow (4H regime + 1H Supertrend trail) ---

def gen_supertrend(d, st_period=10, st_mult=3.0, atr_stop_mult=2.0,
                   trend_ema=200):
    h1, l1, c1, ts1 = d["h1"], d["l1"], d["c1"], d["ts1"]
    st_line, st_dir = supertrend(h1, l1, c1, st_period, st_mult)
    atr1 = wilder_atr(h1, l1, c1, st_period)
    ema_t = ema_series(c1, trend_ema)
    trades = []
    i = 1
    n = len(c1)
    while i < n - 1:
        if math.isnan(atr1[i]) or st_dir[i] == 0:
            i += 1; continue
        flipped_up = st_dir[i] == 1 and st_dir[i - 1] == -1
        flipped_dn = st_dir[i] == -1 and st_dir[i - 1] == 1
        side = None
        if flipped_up and c1[i] > ema_t[i]:
            side = "Buy"
        elif flipped_dn and c1[i] < ema_t[i]:
            side = "Sell"
        if side is None:
            i += 1; continue
        entry = c1[i]
        init_stop = entry - atr_stop_mult * atr1[i] if side == "Buy" else entry + atr_stop_mult * atr1[i]
        # trail on supertrend line; exit when ST flips (dir change) via flip series
        flip = [False] * n
        for j in range(i + 1, n):
            if (side == "Buy" and st_dir[j] == -1) or (side == "Sell" and st_dir[j] == 1):
                flip[j] = True
        xi, xp = simulate_trailing(ts1, h1, l1, c1, i, side, init_stop, None,
                                   trail_line=st_line, flip=flip)
        trades.append(_emit(d, i, xi, side, entry, init_stop, xp))
        i = xi + 1
    return trades


# --- 2. Donchian / Turtle breakout (1H 20-bar, ATR trail, 10-bar exit) ---

def gen_donchian(d, entry_n=20, exit_n=10, atr_period=14, atr_stop_mult=2.0,
                 trend_ema4=50):
    h1, l1, c1 = d["h1"], d["l1"], d["c1"]
    up, dn = donchian(h1, l1, entry_n)
    ex_up, ex_dn = donchian(h1, l1, exit_n)
    atr1 = wilder_atr(h1, l1, c1, atr_period)
    c4, m4 = d["c4"], d["map4"]
    ema4 = ema_series(c4, trend_ema4)
    n = len(c1)
    trades = []
    i = entry_n
    while i < n - 1:
        if math.isnan(up[i - 1]) or math.isnan(atr1[i]):
            i += 1; continue
        j4 = m4[i]
        trend_ok_long = j4 >= 0 and j4 < len(ema4) and not math.isnan(ema4[j4]) and c4[j4] > ema4[j4]
        trend_ok_short = j4 >= 0 and j4 < len(ema4) and not math.isnan(ema4[j4]) and c4[j4] < ema4[j4]
        side = None
        if c1[i] > up[i - 1] and trend_ok_long:
            side = "Buy"
        elif c1[i] < dn[i - 1] and trend_ok_short:
            side = "Sell"
        if side is None:
            i += 1; continue
        entry = c1[i]
        init_stop = entry - atr_stop_mult * atr1[i] if side == "Buy" else entry + atr_stop_mult * atr1[i]
        # flip when price closes beyond opposite exit-channel
        flip = [False] * n
        for j in range(i + 1, n):
            if side == "Buy" and not math.isnan(ex_dn[j]) and c1[j] < ex_dn[j]:
                flip[j] = True
            elif side == "Sell" and not math.isnan(ex_up[j]) and c1[j] > ex_up[j]:
                flip[j] = True
        xi, xp = simulate_trailing(d["ts1"], h1, l1, c1, i, side, init_stop, None,
                                   flip=flip, atr1=atr1, atr_trail_mult=atr_stop_mult)
        trades.append(_emit(d, i, xi, side, entry, init_stop, xp))
        i = xi + 1
    return trades


# --- 3. Chandelier-exit trend (EMA trend + ATR chandelier trailing) ---

def gen_chandelier(d, ema_fast=20, ema_slow=50, atr_period=22, ch_mult=3.0,
                   rr_init=1.0):
    h1, l1, c1 = d["h1"], d["l1"], d["c1"]
    atr1 = wilder_atr(h1, l1, c1, atr_period)
    ef = ema_series(c1, ema_fast); es = ema_series(c1, ema_slow)
    n = len(c1)
    trades = []
    i = ema_slow
    while i < n - 1:
        if math.isnan(atr1[i]):
            i += 1; continue
        cross_up = ef[i] > es[i] and ef[i - 1] <= es[i - 1]
        cross_dn = ef[i] < es[i] and ef[i - 1] >= es[i - 1]
        side = None
        if cross_up:
            side = "Buy"
        elif cross_dn:
            side = "Sell"
        if side is None:
            i += 1; continue
        entry = c1[i]
        init_stop = entry - ch_mult * atr1[i] if side == "Buy" else entry + ch_mult * atr1[i]
        # chandelier line: highest-high - mult*ATR (long) trailing
        flip = [False] * n
        for j in range(i + 1, n):
            if side == "Buy" and ef[j] < es[j]:
                flip[j] = True
            elif side == "Sell" and ef[j] > es[j]:
                flip[j] = True
        # build chandelier trailing line from running extreme
        trail = [float("nan")] * n
        ext = h1[i] if side == "Buy" else l1[i]
        for j in range(i + 1, n):
            if side == "Buy":
                ext = max(ext, h1[j - 1])
                trail[j - 1] = ext - ch_mult * atr1[j - 1] if not math.isnan(atr1[j - 1]) else float("nan")
            else:
                ext = min(ext, l1[j - 1])
                trail[j - 1] = ext + ch_mult * atr1[j - 1] if not math.isnan(atr1[j - 1]) else float("nan")
        xi, xp = simulate_trailing(d["ts1"], h1, l1, c1, i, side, init_stop, None,
                                   trail_line=trail, flip=flip)
        trades.append(_emit(d, i, xi, side, entry, init_stop, xp))
        i = xi + 1
    return trades


# --- 4. Keltner squeeze breakout (BB inside Keltner -> breakout) ---

def gen_keltner_squeeze(d, period=20, bb_mult=2.0, kc_mult=1.5, atr_period=20,
                        atr_stop_mult=2.0, rr=2.5):
    h1, l1, c1 = d["h1"], d["l1"], d["c1"]
    basis = sma_series(c1, period)
    sd = stdev_series(c1, period)
    atr1 = wilder_atr(h1, l1, c1, atr_period)
    n = len(c1)
    trades = []
    i = period
    while i < n - 1:
        if math.isnan(basis[i - 1]) or math.isnan(sd[i - 1]) or math.isnan(atr1[i - 1]):
            i += 1; continue
        bb_u = basis[i - 1] + bb_mult * sd[i - 1]; bb_l = basis[i - 1] - bb_mult * sd[i - 1]
        kc_u = basis[i - 1] + kc_mult * atr1[i - 1]; kc_l = basis[i - 1] - kc_mult * atr1[i - 1]
        squeeze = bb_u < kc_u and bb_l > kc_l   # BB inside KC = low vol
        side = None
        if squeeze:
            if c1[i] > kc_u:
                side = "Buy"
            elif c1[i] < kc_l:
                side = "Sell"
        if side is None:
            i += 1; continue
        entry = c1[i]
        init_stop = entry - atr_stop_mult * atr1[i] if side == "Buy" else entry + atr_stop_mult * atr1[i]
        risk = abs(entry - init_stop)
        tp = entry + rr * risk if side == "Buy" else entry - rr * risk
        xi, xp = simulate_trailing(d["ts1"], h1, l1, c1, i, side, init_stop, tp,
                                   atr1=atr1, atr_trail_mult=atr_stop_mult, breakeven_R=1.0)
        trades.append(_emit(d, i, xi, side, entry, init_stop, xp))
        i = xi + 1
    return trades


# --- 5. RSI(2) Connors mean reversion (long-only in uptrend) ---

def gen_rsi2_meanrev(d, rsi_p=2, sma_long=200, sma_exit=5, rsi_buy=10, rsi_sell=90,
                     atr_period=14, atr_stop_mult=3.0, allow_short=True):
    h1, l1, c1 = d["h1"], d["l1"], d["c1"]
    rsi = rsi_series(c1, rsi_p)
    smaL = sma_series(c1, sma_long)
    smaE = sma_series(c1, sma_exit)
    atr1 = wilder_atr(h1, l1, c1, atr_period)
    n = len(c1)
    trades = []
    i = sma_long
    while i < n - 1:
        if math.isnan(rsi[i]) or math.isnan(smaL[i]) or math.isnan(atr1[i]):
            i += 1; continue
        side = None
        if c1[i] > smaL[i] and rsi[i] < rsi_buy:
            side = "Buy"
        elif allow_short and c1[i] < smaL[i] and rsi[i] > rsi_sell:
            side = "Sell"
        if side is None:
            i += 1; continue
        entry = c1[i]
        init_stop = entry - atr_stop_mult * atr1[i] if side == "Buy" else entry + atr_stop_mult * atr1[i]
        # exit when close crosses back above/below short SMA
        flip = [False] * n
        for j in range(i + 1, n):
            if math.isnan(smaE[j]):
                continue
            if side == "Buy" and c1[j] > smaE[j]:
                flip[j] = True
            elif side == "Sell" and c1[j] < smaE[j]:
                flip[j] = True
        xi, xp = simulate_trailing(d["ts1"], h1, l1, c1, i, side, init_stop, None, flip=flip)
        trades.append(_emit(d, i, xi, side, entry, init_stop, xp))
        i = xi + 1
    return trades


# --- 6. Time-series momentum (4H ROC sign + EMA filter, ATR trail) ---
# NO LOOK-AHEAD: signal uses only the LAST CLOSED 4H bar (index j4-1 at a new-4H
# boundary). Entry at the close of the 1H bar that opens the new 4H period.

def gen_tsmom(d, roc_p=30, ema_p=50, atr_period=14, atr_stop_mult=3.0,
              long_only=False):
    h1, l1, c1 = d["h1"], d["l1"], d["c1"]
    c4, m4 = d["c4"], d["map4"]
    roc4 = roc_series(c4, roc_p)
    ema4 = ema_series(c4, ema_p)
    atr1 = wilder_atr(h1, l1, c1, atr_period)
    need = max(roc_p, ema_p)
    n = len(c1)
    # precompute flip arrays once: a new 4H close that reverses the side's momentum
    flip_buy = [False] * n
    flip_sell = [False] * n
    pj = -1
    for j in range(n):
        jj = m4[j]
        if jj != pj:
            cj = jj - 1
            if cj >= 0 and not math.isnan(roc4[cj]) and not math.isnan(ema4[cj]):
                if roc4[cj] < 0 or c4[cj] < ema4[cj]:
                    flip_buy[j] = True
                if roc4[cj] > 0 or c4[cj] > ema4[cj]:
                    flip_sell[j] = True
            pj = jj
    trades = []
    i = 1
    prev_j4 = -1
    while i < n - 1:
        j4 = m4[i]
        is_new = j4 != prev_j4
        prev_j4 = j4
        sj = j4 - 1                       # last CLOSED 4H bar
        if not is_new or sj < need or math.isnan(atr1[i]):
            i += 1; continue
        if math.isnan(roc4[sj]) or math.isnan(ema4[sj]):
            i += 1; continue
        side = None
        if roc4[sj] > 0 and c4[sj] > ema4[sj]:
            side = "Buy"
        elif not long_only and roc4[sj] < 0 and c4[sj] < ema4[sj]:
            side = "Sell"
        if side is None:
            i += 1; continue
        entry = c1[i]
        init_stop = entry - atr_stop_mult * atr1[i] if side == "Buy" else entry + atr_stop_mult * atr1[i]
        # flip when the momentum of a newly-CLOSED 4H bar reverses (uses shared flip arrays)
        flip = flip_buy if side == "Buy" else flip_sell
        xi, xp = simulate_trailing(d["ts1"], h1, l1, c1, i, side, init_stop, None,
                                   flip=flip, atr1=atr1, atr_trail_mult=atr_stop_mult)
        trades.append(_emit(d, i, xi, side, entry, init_stop, xp))
        i = xi + 1
    return trades


# --- 7. ADX-filtered EMA trend pullback with ATR trail ---

def gen_adx_trend(d, ema_fast=20, ema_slow=50, adx_p=14, adx_min=25,
                  atr_period=14, atr_stop_mult=2.5, pullback_atr=0.5):
    h1, l1, c1 = d["h1"], d["l1"], d["c1"]
    ef = ema_series(c1, ema_fast); es = ema_series(c1, ema_slow)
    adx = adx_series(h1, l1, c1, adx_p)
    atr1 = wilder_atr(h1, l1, c1, atr_period)
    n = len(c1)
    trades = []
    i = ema_slow
    while i < n - 1:
        if math.isnan(adx[i]) or math.isnan(atr1[i]):
            i += 1; continue
        if adx[i] < adx_min:
            i += 1; continue
        side = None
        # trend up, price pulled back near fast EMA then closes up
        if ef[i] > es[i] and c1[i - 1] <= ef[i - 1] + pullback_atr * atr1[i] and c1[i] > c1[i - 1] and c1[i] > ef[i]:
            side = "Buy"
        elif ef[i] < es[i] and c1[i - 1] >= ef[i - 1] - pullback_atr * atr1[i] and c1[i] < c1[i - 1] and c1[i] < ef[i]:
            side = "Sell"
        if side is None:
            i += 1; continue
        entry = c1[i]
        init_stop = entry - atr_stop_mult * atr1[i] if side == "Buy" else entry + atr_stop_mult * atr1[i]
        flip = [False] * n
        for j in range(i + 1, n):
            if side == "Buy" and ef[j] < es[j]:
                flip[j] = True
            elif side == "Sell" and ef[j] > es[j]:
                flip[j] = True
        xi, xp = simulate_trailing(d["ts1"], h1, l1, c1, i, side, init_stop, None,
                                   flip=flip, atr1=atr1, atr_trail_mult=atr_stop_mult, breakeven_R=1.0)
        trades.append(_emit(d, i, xi, side, entry, init_stop, xp))
        i = xi + 1
    return trades


# --- 8. Bollinger mean reversion with trailing (range regime) ---

def gen_bb_meanrev(d, period=20, mult=2.0, atr_period=14, atr_stop_mult=1.5,
                   adx_p=14, adx_max=25):
    h1, l1, c1 = d["h1"], d["l1"], d["c1"]
    basis = sma_series(c1, period); sd = stdev_series(c1, period)
    atr1 = wilder_atr(h1, l1, c1, atr_period)
    adx = adx_series(h1, l1, c1, adx_p)
    n = len(c1)
    trades = []
    i = period
    while i < n - 1:
        if math.isnan(basis[i]) or math.isnan(sd[i]) or math.isnan(atr1[i]) or math.isnan(adx[i]):
            i += 1; continue
        if adx[i] > adx_max or sd[i] <= 0:
            i += 1; continue
        up = basis[i] + mult * sd[i]; lo = basis[i] - mult * sd[i]
        side = None
        if l1[i] <= lo and c1[i] < basis[i]:
            side = "Buy"
        elif h1[i] >= up and c1[i] > basis[i]:
            side = "Sell"
        if side is None:
            i += 1; continue
        entry = c1[i]
        init_stop = entry - atr_stop_mult * atr1[i] if side == "Buy" else entry + atr_stop_mult * atr1[i]
        tp = basis[i]  # revert to mean
        # exit at mean (tp) or stop
        xi, xp = simulate_trailing(d["ts1"], h1, l1, c1, i, side, init_stop, tp)
        trades.append(_emit(d, i, xi, side, entry, init_stop, xp))
        i = xi + 1
    return trades


STRATS: Dict[str, Callable] = {
    "supertrend": gen_supertrend,
    "donchian": gen_donchian,
    "chandelier": gen_chandelier,
    "keltner_squeeze": gen_keltner_squeeze,
    "rsi2_meanrev": gen_rsi2_meanrev,
    "tsmom": gen_tsmom,
    "adx_trend": gen_adx_trend,
    "bb_meanrev": gen_bb_meanrev,
}


def gen_raw(strat: str, k1, k4, **kw) -> List[RawTrade]:
    d = build_symbol(k1, k4)
    return STRATS[strat](d, **kw)
