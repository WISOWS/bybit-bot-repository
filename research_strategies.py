from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from main import (
    LIVE_MIN_DISTANCE_SCORE,
    LIVE_SIGNAL_THRESHOLD,
    LIVE_SL_BUFFER,
    MAX_LEVEL_DISTANCE_ATR,
    MAX_STOP_ATR,
    calc_atr_from_klines,
    calc_signal_score,
    detect_trend_4h,
    find_level_break_trend,
    impulse_filter_ok,
    is_range_dirty_around_level,
    level_touched,
)


@dataclass
class StrategySetup:
    strategy: str
    side: str
    entry: float
    stop: float
    tp: float
    atr_4h: float
    level: float
    planned_rr: float
    impulse_ok: bool
    meta: Dict[str, Any]


def candle_open(candle: List[Any]) -> float:
    return float(candle[1])


def candle_high(candle: List[Any]) -> float:
    return float(candle[2])


def candle_low(candle: List[Any]) -> float:
    return float(candle[3])


def candle_close(candle: List[Any]) -> float:
    return float(candle[4])


def bullish_close(candle: List[Any]) -> bool:
    return candle_close(candle) > candle_open(candle)


def bearish_close(candle: List[Any]) -> bool:
    return candle_close(candle) < candle_open(candle)


def ema(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    multiplier = 2 / (period + 1)
    ema_value = values[0]
    for value in values[1:]:
        ema_value = (value - ema_value) * multiplier + ema_value
    return ema_value


def recent_low(klines: List[List[Any]], idx: int, lookback: int) -> float:
    start = max(0, idx - lookback + 1)
    return min(candle_low(candle) for candle in klines[start : idx + 1])


def recent_high(klines: List[List[Any]], idx: int, lookback: int) -> float:
    start = max(0, idx - lookback + 1)
    return max(candle_high(candle) for candle in klines[start : idx + 1])


def ema_trend_4h(klines_4h: List[List[Any]], idx_4h: int) -> str:
    if idx_4h < 49:
        return "FLAT"
    closes = [candle_close(candle) for candle in klines_4h[idx_4h - 49 : idx_4h + 1]]
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    last_close = closes[-1]
    if ema20 > ema50 and last_close >= ema20:
        return "UP"
    if ema20 < ema50 and last_close <= ema20:
        return "DOWN"
    return "FLAT"


def simple_trend_pullback(
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
) -> Optional[StrategySetup]:
    if idx_4h < 29 or idx_1h < 11:
        return None

    recent_4h_30 = klines_4h[idx_4h - 29 : idx_4h + 1]
    recent_4h_20 = klines_4h[idx_4h - 19 : idx_4h + 1]
    recent_4h_15 = klines_4h[idx_4h - 14 : idx_4h + 1]
    recent_1h_12 = klines_1h[idx_1h - 11 : idx_1h + 1]
    current_1h = klines_1h[idx_1h]

    trend = detect_trend_4h(recent_4h_30)
    if trend == "FLAT":
        return None

    level_pair = find_level_break_trend(recent_4h_30)
    if level_pair is None:
        return None

    atr_4h = calc_atr_from_klines(recent_4h_15, period=14)
    if atr_4h <= 0:
        return None

    level_low, level_high = level_pair
    level = level_low if trend == "UP" else level_high
    side = "Buy" if trend == "UP" else "Sell"
    entry = candle_close(current_1h)
    max_level_distance = atr_4h * MAX_LEVEL_DISTANCE_ATR

    trend_score = 1.0 if (
        (trend == "UP" and entry >= level)
        or (trend == "DOWN" and entry <= level)
    ) else 0.3
    distance_score = max(0.1, 1 - abs(entry - level) / max_level_distance) if max_level_distance > 0 else 0.1
    if distance_score < LIVE_MIN_DISTANCE_SCORE:
        return None

    impulse_ok = impulse_filter_ok(recent_1h_12, len(recent_1h_12) - 1, side)
    if not impulse_ok:
        return None

    clean_level = not is_range_dirty_around_level(recent_4h_20, level)
    if not clean_level:
        return None

    structure_touch = level_touched(current_1h, level, atr_4h * 0.2)
    structure_score = 0.3 if structure_touch else 0.0
    edge_score = calc_signal_score(
        trend_score,
        1.0,
        distance_score,
        1.0,
        structure_score,
    )
    if edge_score < LIVE_SIGNAL_THRESHOLD:
        return None

    if side == "Buy":
        stop = level - atr_4h * 0.2
        tp = entry + (entry - stop) * 2.8
        risk = entry - stop
        reward = tp - entry
    else:
        stop = level + atr_4h * 0.2
        tp = entry - (stop - entry) * 2.8
        risk = stop - entry
        reward = entry - tp

    if risk <= 0 or reward <= 0:
        return None
    if risk > atr_4h * MAX_STOP_ATR * 1.3:
        return None
    if risk < atr_4h * LIVE_SL_BUFFER:
        return None

    return StrategySetup(
        strategy="simple_trend_pullback",
        side=side,
        entry=entry,
        stop=stop,
        tp=tp,
        atr_4h=atr_4h,
        level=level,
        planned_rr=reward / risk,
        impulse_ok=impulse_ok,
        meta={
            "trend": trend,
            "trend_score": trend_score,
            "level_score": 1.0,
            "distance_score": round(distance_score, 6),
            "impulse_score": 1.0,
            "structure_score": round(structure_score, 6),
            "edge_score": round(edge_score, 6),
        },
    )


def breakout_retest(
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
) -> Optional[StrategySetup]:
    if idx_4h < 25 or idx_1h < 2:
        return None

    current_1h = klines_1h[idx_1h]
    previous_1h = klines_1h[idx_1h - 1]
    recent_4h_20 = klines_4h[idx_4h - 20 : idx_4h]
    recent_4h_15 = klines_4h[idx_4h - 14 : idx_4h + 1]

    if not recent_4h_20:
        return None

    trend = ema_trend_4h(klines_4h, idx_4h)
    if trend == "FLAT":
        return None

    breakout_high = max(candle_high(candle) for candle in recent_4h_20)
    breakout_low = min(candle_low(candle) for candle in recent_4h_20)
    atr_4h = calc_atr_from_klines(recent_4h_15, period=14)
    if atr_4h <= 0:
        return None

    entry = candle_close(current_1h)

    if (
        trend == "UP"
        and candle_close(previous_1h) > breakout_high
        and candle_low(current_1h) <= breakout_high * 1.002
        and candle_close(current_1h) > breakout_high
        and bullish_close(current_1h)
    ):
        side = "Buy"
        level = breakout_high
        stop = min(recent_low(klines_1h, idx_1h, 5), breakout_high) - atr_4h * 0.15
        risk = entry - stop
        tp = entry + risk * 2.8
    elif (
        trend == "DOWN"
        and candle_close(previous_1h) < breakout_low
        and candle_high(current_1h) >= breakout_low * 0.998
        and candle_close(current_1h) < breakout_low
        and bearish_close(current_1h)
    ):
        side = "Sell"
        level = breakout_low
        stop = max(recent_high(klines_1h, idx_1h, 5), breakout_low) + atr_4h * 0.15
        risk = stop - entry
        tp = entry - risk * 2.8
    else:
        return None

    if risk <= 0:
        return None
    if risk > atr_4h * MAX_STOP_ATR * 1.3:
        return None
    if risk < atr_4h * LIVE_SL_BUFFER:
        return None

    reward = abs(tp - entry)
    return StrategySetup(
        strategy="breakout_retest",
        side=side,
        entry=entry,
        stop=stop,
        tp=tp,
        atr_4h=atr_4h,
        level=level,
        planned_rr=reward / risk,
        impulse_ok=True,
        meta={
            "trend": trend,
            "entry_pattern": "breakout_retest",
            "edge_score": None,
        },
    )


def range_mean_reversion(
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
) -> Optional[StrategySetup]:
    if idx_4h < 29 or idx_1h < 1:
        return None

    recent_4h_30 = klines_4h[idx_4h - 29 : idx_4h + 1]
    recent_4h_20 = klines_4h[idx_4h - 19 : idx_4h + 1]
    recent_4h_15 = klines_4h[idx_4h - 14 : idx_4h + 1]
    current_1h = klines_1h[idx_1h]

    if detect_trend_4h(recent_4h_30) != "FLAT":
        return None

    atr_4h = calc_atr_from_klines(recent_4h_15, period=14)
    if atr_4h <= 0:
        return None

    range_high = max(candle_high(candle) for candle in recent_4h_20)
    range_low = min(candle_low(candle) for candle in recent_4h_20)
    range_width = range_high - range_low
    if range_width <= 0:
        return None

    entry = candle_close(current_1h)
    midpoint = range_low + range_width * 0.5
    buy_zone = range_low + range_width * 0.12
    sell_zone = range_high - range_width * 0.12

    if candle_low(current_1h) <= buy_zone and bullish_close(current_1h):
        side = "Buy"
        level = range_low
        stop = range_low - atr_4h * 0.2
        risk = entry - stop
        tp = midpoint
        reward = tp - entry
    elif candle_high(current_1h) >= sell_zone and bearish_close(current_1h):
        side = "Sell"
        level = range_high
        stop = range_high + atr_4h * 0.2
        risk = stop - entry
        tp = midpoint
        reward = entry - tp
    else:
        return None

    if risk <= 0 or reward <= 0:
        return None
    if risk > atr_4h * MAX_STOP_ATR:
        return None
    if reward / risk < 1.5:
        return None

    return StrategySetup(
        strategy="range_mean_reversion",
        side=side,
        entry=entry,
        stop=stop,
        tp=tp,
        atr_4h=atr_4h,
        level=level,
        planned_rr=reward / risk,
        impulse_ok=True,
        meta={
            "trend": "FLAT",
            "entry_pattern": "range_reversion",
            "range_high": round(range_high, 8),
            "range_low": round(range_low, 8),
        },
    )


STRATEGIES = {
    "simple_trend_pullback": simple_trend_pullback,
    "breakout_retest": breakout_retest,
    "range_mean_reversion": range_mean_reversion,
}


STRATEGY_DESCRIPTIONS = {
    "simple_trend_pullback": "4H trend + clean level + distance + impulse",
    "breakout_retest": "4H breakout with 1H retest confirmation",
    "range_mean_reversion": "4H flat range with 1H mean reversion entry",
}
