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


@dataclass(frozen=True)
class TrendPullbackParams:
    threshold: float = LIVE_SIGNAL_THRESHOLD
    min_distance_score: float = LIVE_MIN_DISTANCE_SCORE
    rr_target: float = 2.8
    stop_buffer_atr: float = 0.2
    max_stop_multiplier: float = 1.3
    require_impulse: bool = True
    require_clean_level: bool = True
    require_structure_touch: bool = False
    strict_trend_alignment: bool = False


@dataclass(frozen=True)
class BreakoutRetestParams:
    breakout_lookback_4h: int = 20
    rr_target: float = 2.8
    stop_buffer_atr: float = 0.15
    max_stop_multiplier: float = 1.3
    retest_buffer_pct: float = 0.002
    stop_lookback_1h: int = 5
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    require_directional_close: bool = True


@dataclass(frozen=True)
class RangeMeanReversionParams:
    flat_lookback_4h: int = 30
    range_lookback_4h: int = 20
    zone_fraction: float = 0.12
    stop_buffer_atr: float = 0.2
    target_fraction: float = 0.5
    min_rr: float = 1.5
    max_stop_multiplier: float = 1.0
    require_directional_close: bool = True


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


def ema_trend_4h(
    klines_4h: List[List[Any]],
    idx_4h: int,
    fast_period: int = 20,
    slow_period: int = 50,
) -> str:
    if slow_period <= 0 or fast_period <= 0 or fast_period >= slow_period:
        return "FLAT"
    if idx_4h < slow_period - 1:
        return "FLAT"
    closes = [candle_close(candle) for candle in klines_4h[idx_4h - slow_period + 1 : idx_4h + 1]]
    ema_fast = ema(closes, fast_period)
    ema_slow = ema(closes, slow_period)
    last_close = closes[-1]
    if ema_fast > ema_slow and last_close >= ema_fast:
        return "UP"
    if ema_fast < ema_slow and last_close <= ema_fast:
        return "DOWN"
    return "FLAT"


def make_simple_trend_pullback_strategy(params: TrendPullbackParams):
    def strategy(
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
        if params.strict_trend_alignment and trend_score < 1.0:
            return None

        distance_score = (
            max(0.1, 1 - abs(entry - level) / max_level_distance) if max_level_distance > 0 else 0.1
        )
        if distance_score < params.min_distance_score:
            return None

        impulse_ok = impulse_filter_ok(recent_1h_12, len(recent_1h_12) - 1, side)
        if params.require_impulse and not impulse_ok:
            return None

        clean_level = not is_range_dirty_around_level(recent_4h_20, level)
        if params.require_clean_level and not clean_level:
            return None

        structure_touch = level_touched(current_1h, level, atr_4h * 0.2)
        if params.require_structure_touch and not structure_touch:
            return None

        structure_score = 0.3 if structure_touch else 0.0
        impulse_score = 1.0 if impulse_ok else 0.3
        level_score = 1.0 if clean_level else 0.2
        edge_score = calc_signal_score(
            trend_score,
            level_score,
            distance_score,
            impulse_score,
            structure_score,
        )
        if edge_score < params.threshold:
            return None

        if side == "Buy":
            stop = level - atr_4h * params.stop_buffer_atr
            tp = entry + (entry - stop) * params.rr_target
            risk = entry - stop
            reward = tp - entry
        else:
            stop = level + atr_4h * params.stop_buffer_atr
            tp = entry - (stop - entry) * params.rr_target
            risk = stop - entry
            reward = entry - tp

        if risk <= 0 or reward <= 0:
            return None
        if risk > atr_4h * MAX_STOP_ATR * params.max_stop_multiplier:
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
                "level_score": level_score,
                "distance_score": round(distance_score, 6),
                "impulse_score": impulse_score,
                "structure_score": round(structure_score, 6),
                "edge_score": round(edge_score, 6),
                "params": {
                    "threshold": params.threshold,
                    "min_distance_score": params.min_distance_score,
                    "rr_target": params.rr_target,
                    "stop_buffer_atr": params.stop_buffer_atr,
                    "max_stop_multiplier": params.max_stop_multiplier,
                    "require_impulse": params.require_impulse,
                    "require_clean_level": params.require_clean_level,
                    "require_structure_touch": params.require_structure_touch,
                    "strict_trend_alignment": params.strict_trend_alignment,
                },
            },
        )

    return strategy


def simple_trend_pullback(
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
) -> Optional[StrategySetup]:
    return make_simple_trend_pullback_strategy(TrendPullbackParams())(
        klines_1h,
        idx_1h,
        klines_4h,
        idx_4h,
    )


def breakout_retest(
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
) -> Optional[StrategySetup]:
    return make_breakout_retest_strategy(BreakoutRetestParams())(
        klines_1h,
        idx_1h,
        klines_4h,
        idx_4h,
    )


def make_breakout_retest_strategy(params: BreakoutRetestParams):
    def strategy(
        klines_1h: List[List[Any]],
        idx_1h: int,
        klines_4h: List[List[Any]],
        idx_4h: int,
    ) -> Optional[StrategySetup]:
        min_4h_idx = max(params.breakout_lookback_4h + 1, params.ema_slow_period)
        if idx_4h < min_4h_idx or idx_1h < 2:
            return None

        current_1h = klines_1h[idx_1h]
        previous_1h = klines_1h[idx_1h - 1]
        recent_4h = klines_4h[idx_4h - params.breakout_lookback_4h : idx_4h]
        recent_4h_15 = klines_4h[idx_4h - 14 : idx_4h + 1]

        if len(recent_4h) < params.breakout_lookback_4h:
            return None

        trend = ema_trend_4h(
            klines_4h,
            idx_4h,
            fast_period=params.ema_fast_period,
            slow_period=params.ema_slow_period,
        )
        if trend == "FLAT":
            return None

        breakout_high = max(candle_high(candle) for candle in recent_4h)
        breakout_low = min(candle_low(candle) for candle in recent_4h)
        atr_4h = calc_atr_from_klines(recent_4h_15, period=14)
        if atr_4h <= 0:
            return None

        entry = candle_close(current_1h)
        buffer_up = breakout_high * (1 + params.retest_buffer_pct)
        buffer_down = breakout_low * (1 - params.retest_buffer_pct)

        buy_confirm = bullish_close(current_1h) if params.require_directional_close else True
        sell_confirm = bearish_close(current_1h) if params.require_directional_close else True

        if (
            trend == "UP"
            and candle_close(previous_1h) > breakout_high
            and candle_low(current_1h) <= buffer_up
            and candle_close(current_1h) > breakout_high
            and buy_confirm
        ):
            side = "Buy"
            level = breakout_high
            stop = min(recent_low(klines_1h, idx_1h, params.stop_lookback_1h), breakout_high) - atr_4h * params.stop_buffer_atr
            risk = entry - stop
            tp = entry + risk * params.rr_target
        elif (
            trend == "DOWN"
            and candle_close(previous_1h) < breakout_low
            and candle_high(current_1h) >= buffer_down
            and candle_close(current_1h) < breakout_low
            and sell_confirm
        ):
            side = "Sell"
            level = breakout_low
            stop = max(recent_high(klines_1h, idx_1h, params.stop_lookback_1h), breakout_low) + atr_4h * params.stop_buffer_atr
            risk = stop - entry
            tp = entry - risk * params.rr_target
        else:
            return None

        if risk <= 0:
            return None
        if risk > atr_4h * MAX_STOP_ATR * params.max_stop_multiplier:
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
                "params": {
                    "breakout_lookback_4h": params.breakout_lookback_4h,
                    "rr_target": params.rr_target,
                    "stop_buffer_atr": params.stop_buffer_atr,
                    "max_stop_multiplier": params.max_stop_multiplier,
                    "retest_buffer_pct": params.retest_buffer_pct,
                    "stop_lookback_1h": params.stop_lookback_1h,
                    "ema_fast_period": params.ema_fast_period,
                    "ema_slow_period": params.ema_slow_period,
                    "require_directional_close": params.require_directional_close,
                },
            },
        )

    return strategy


def range_mean_reversion(
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
) -> Optional[StrategySetup]:
    return make_range_mean_reversion_strategy(RangeMeanReversionParams())(
        klines_1h,
        idx_1h,
        klines_4h,
        idx_4h,
    )


def make_range_mean_reversion_strategy(params: RangeMeanReversionParams):
    def strategy(
        klines_1h: List[List[Any]],
        idx_1h: int,
        klines_4h: List[List[Any]],
        idx_4h: int,
    ) -> Optional[StrategySetup]:
        min_idx = max(params.flat_lookback_4h, params.range_lookback_4h, 15) - 1
        if idx_4h < min_idx or idx_1h < 1:
            return None

        recent_flat = klines_4h[idx_4h - params.flat_lookback_4h + 1 : idx_4h + 1]
        recent_range = klines_4h[idx_4h - params.range_lookback_4h + 1 : idx_4h + 1]
        recent_4h_15 = klines_4h[idx_4h - 14 : idx_4h + 1]
        current_1h = klines_1h[idx_1h]

        if detect_trend_4h(recent_flat) != "FLAT":
            return None

        atr_4h = calc_atr_from_klines(recent_4h_15, period=14)
        if atr_4h <= 0:
            return None

        range_high = max(candle_high(candle) for candle in recent_range)
        range_low = min(candle_low(candle) for candle in recent_range)
        range_width = range_high - range_low
        if range_width <= 0:
            return None

        entry = candle_close(current_1h)
        target_price = range_low + range_width * params.target_fraction
        buy_zone = range_low + range_width * params.zone_fraction
        sell_zone = range_high - range_width * params.zone_fraction

        buy_confirm = bullish_close(current_1h) if params.require_directional_close else True
        sell_confirm = bearish_close(current_1h) if params.require_directional_close else True

        if candle_low(current_1h) <= buy_zone and buy_confirm:
            side = "Buy"
            level = range_low
            stop = range_low - atr_4h * params.stop_buffer_atr
            risk = entry - stop
            tp = target_price
            reward = tp - entry
        elif candle_high(current_1h) >= sell_zone and sell_confirm:
            side = "Sell"
            level = range_high
            stop = range_high + atr_4h * params.stop_buffer_atr
            risk = stop - entry
            tp = target_price
            reward = entry - tp
        else:
            return None

        if risk <= 0 or reward <= 0:
            return None
        if risk > atr_4h * MAX_STOP_ATR * params.max_stop_multiplier:
            return None
        if reward / risk < params.min_rr:
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
                "params": {
                    "flat_lookback_4h": params.flat_lookback_4h,
                    "range_lookback_4h": params.range_lookback_4h,
                    "zone_fraction": params.zone_fraction,
                    "stop_buffer_atr": params.stop_buffer_atr,
                    "target_fraction": params.target_fraction,
                    "min_rr": params.min_rr,
                    "max_stop_multiplier": params.max_stop_multiplier,
                    "require_directional_close": params.require_directional_close,
                },
            },
        )

    return strategy


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
