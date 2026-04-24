import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Set, Tuple

from main import (
    BASE_URL,
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    LIVE_SL_BUFFER,
    MAX_LEVEL_DISTANCE_ATR,
    MAX_STOP_ATR,
    calc_atr_from_klines,
    detect_trend_4h,
    find_level_break_trend,
    impulse_filter_ok,
    is_range_dirty_around_level,
    CONFIG_SOURCE,
    JOURNAL_PATH,
    LEVERAGE,
    MODE,
    POSITION_IDX,
    REQUEST_TIMEOUT,
    RISK_PER_TRADE,
    TIMEFRAME_SIGNAL,
    TIMEFRAME_TREND,
    TRIGGER_BY,
    BybitClient,
    append_journal_row,
    build_order_link_id,
    calc_qty_by_margin,
    calc_qty_by_risk,
    force_close_position,
    get_symbol_constraints,
    load_journal_rows,
    normalize_price,
    normalize_qty,
    parse_note_payload,
    sync_closed_trades_to_journal,
    validate_symbols,
    wait_for_position,
    level_touched,
)
from research_search_meta_portfolio import MODEL_SPECS, btc_mode_ok, build_4h_index, hot_vol_ratio_at
from research_strategies import make_regime_switch_strategy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "meta_portfolio.log")

# TODO: Refactor strategy call sites to return a structured StrategyDecision
# instead of Optional[StrategySetup]. That would let the strategy itself return
# authoritative reject_reason/debug metadata and remove the mirrored
# _explain_* helpers below, which currently duplicate rejection logic from
# /Users/egor/bybit-bot/research_strategies.py and can drift over time.

FORWARD_RUNNER_NAME = "meta_portfolio_forward_v1"
DEFAULT_SYMBOLS = "NEARUSDT,SOLUSDT,LINKUSDT,ENAUSDT"
DEFAULT_BTC_SYMBOL = "BTCUSDT"
DEFAULT_BTC_MODE = "allow_flat"
DEFAULT_MIN_HOT_VOL_RATIO = 0.75
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_SLEEP_SECONDS = 300


logger = logging.getLogger("meta_portfolio_forward")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_btc_state(state: bool | None, side: str | None = None) -> str:
    if state is None:
        return "n/a"
    if side:
        return f"{'pass' if state else 'fail'}(side={side})"
    return "pass" if state else "fail"


def _current_regime_4h(model: Any, klines_4h: List[List[Any]]) -> str:
    idx_4h = len(klines_4h) - 1
    if idx_4h < model.params.regime_lookback_4h - 1:
        return "n/a"
    recent_regime = klines_4h[idx_4h - model.params.regime_lookback_4h + 1 : idx_4h + 1]
    return detect_trend_4h(recent_regime)


def _expected_side_for_regime(regime: str) -> str | None:
    if regime == "UP":
        return "Buy"
    if regime == "DOWN":
        return "Sell"
    return None


def _explain_trend_setup_none(model: Any, klines_1h: List[List[Any]], klines_4h: List[List[Any]]) -> str:
    # WARNING: This helper intentionally mirrors the rejection path from
    # make_simple_trend_pullback_strategy() in
    # /Users/egor/bybit-bot/research_strategies.py:182-304.
    # In practice it is duplicating the trend branch selected by
    # make_regime_switch_strategy() in
    # /Users/egor/bybit-bot/research_strategies.py:484-570,
    # especially the call at lines 547-549.
    # If any trend entry condition changes there, this helper MUST be updated
    # in sync. Otherwise meta_portfolio_forward.py will keep logging stale or
    # false rejection reasons even though the trading logic itself changed.
    idx_1h = len(klines_1h) - 1
    idx_4h = len(klines_4h) - 1
    if idx_4h < 29 or idx_1h < 11:
        return "no_trend_setup:insufficient_history"

    recent_4h_30 = klines_4h[idx_4h - 29 : idx_4h + 1]
    recent_4h_20 = klines_4h[idx_4h - 19 : idx_4h + 1]
    recent_4h_15 = klines_4h[idx_4h - 14 : idx_4h + 1]
    recent_1h_12 = klines_1h[idx_1h - 11 : idx_1h + 1]
    current_1h = klines_1h[idx_1h]

    trend = detect_trend_4h(recent_4h_30)
    if trend == "FLAT":
        return "no_trend_setup:trend_is_flat"

    level_pair = find_level_break_trend(recent_4h_30)
    if level_pair is None:
        return "no_trend_setup:no_level_break"

    atr_4h = calc_atr_from_klines(recent_4h_15, period=14)
    if atr_4h <= 0:
        return "no_trend_setup:atr_non_positive"

    level_low, level_high = level_pair
    level = level_low if trend == "UP" else level_high
    side = "Buy" if trend == "UP" else "Sell"
    entry = float(current_1h[4])
    max_level_distance = atr_4h * MAX_LEVEL_DISTANCE_ATR

    trend_score = 1.0 if (
        (trend == "UP" and entry >= level)
        or (trend == "DOWN" and entry <= level)
    ) else 0.3
    if model.params.strict_trend_alignment and trend_score < 1.0:
        return "no_trend_setup:strict_trend_alignment_failed"

    distance_score = (
        max(0.1, 1 - abs(entry - level) / max_level_distance) if max_level_distance > 0 else 0.1
    )
    if distance_score < model.params.trend_min_distance_score:
        return f"no_trend_setup:distance_too_far score={distance_score:.3f}<{model.params.trend_min_distance_score:.3f}"

    impulse_ok = impulse_filter_ok(recent_1h_12, len(recent_1h_12) - 1, side)
    if not impulse_ok:
        return "no_trend_setup:impulse_missing"

    clean_level = not is_range_dirty_around_level(recent_4h_20, level)
    if not clean_level:
        return "no_trend_setup:dirty_level"

    structure_touch = level_touched(current_1h, level, atr_4h * 0.2)
    structure_score = 0.3 if structure_touch else 0.0
    impulse_score = 1.0 if impulse_ok else 0.3
    level_score = 1.0 if clean_level else 0.2
    edge_score = (
        trend_score * 0.25
        + level_score * 0.25
        + distance_score * 0.20
        + impulse_score * 0.20
        + structure_score * 0.10
    )
    if edge_score < model.params.trend_threshold:
        return f"no_trend_setup:edge_below_threshold edge={edge_score:.3f}<{model.params.trend_threshold:.3f}"

    if side == "Buy":
        stop = level - atr_4h * model.params.trend_stop_buffer_atr
        tp = entry + (entry - stop) * model.params.trend_rr_target
        risk = entry - stop
        reward = tp - entry
    else:
        stop = level + atr_4h * model.params.trend_stop_buffer_atr
        tp = entry - (stop - entry) * model.params.trend_rr_target
        risk = stop - entry
        reward = entry - tp

    if risk <= 0 or reward <= 0:
        return "no_trend_setup:invalid_risk_reward"
    if risk > atr_4h * MAX_STOP_ATR * model.params.trend_max_stop_multiplier:
        return "no_trend_setup:stop_too_wide"
    if risk < atr_4h * LIVE_SL_BUFFER:
        return "no_trend_setup:stop_too_tight"
    return "no_trend_setup:unknown"


def _explain_range_setup_none(model: Any, klines_1h: List[List[Any]], klines_4h: List[List[Any]]) -> str:
    # WARNING: This helper intentionally mirrors the rejection path from
    # make_range_mean_reversion_strategy() in
    # /Users/egor/bybit-bot/research_strategies.py:720-809.
    # In practice it is duplicating the FLAT-regime branch selected by
    # make_regime_switch_strategy() in
    # /Users/egor/bybit-bot/research_strategies.py:484-570,
    # especially the call at lines 523-526.
    # If any range entry condition changes there, this helper MUST be updated
    # in sync. Otherwise meta_portfolio_forward.py will keep logging stale or
    # false rejection reasons even though the trading logic itself changed.
    idx_1h = len(klines_1h) - 1
    idx_4h = len(klines_4h) - 1
    min_idx = max(model.params.range_flat_lookback_4h, model.params.range_lookback_4h, 15) - 1
    if idx_4h < min_idx or idx_1h < 1:
        return "no_range_setup:insufficient_history"

    recent_flat = klines_4h[idx_4h - model.params.range_flat_lookback_4h + 1 : idx_4h + 1]
    recent_range = klines_4h[idx_4h - model.params.range_lookback_4h + 1 : idx_4h + 1]
    recent_4h_15 = klines_4h[idx_4h - 14 : idx_4h + 1]
    current_1h = klines_1h[idx_1h]

    if detect_trend_4h(recent_flat) != "FLAT":
        return "no_range_setup:not_flat_regime"

    atr_4h = calc_atr_from_klines(recent_4h_15, period=14)
    if atr_4h <= 0:
        return "no_range_setup:atr_non_positive"

    range_high = max(float(c[2]) for c in recent_range)
    range_low = min(float(c[3]) for c in recent_range)
    range_width = range_high - range_low
    if range_width <= 0:
        return "no_range_setup:zero_range_width"

    entry = float(current_1h[4])
    target_price = range_low + range_width * model.params.range_target_fraction
    buy_zone = range_low + range_width * model.params.range_zone_fraction
    sell_zone = range_high - range_width * model.params.range_zone_fraction

    buy_confirm = float(current_1h[4]) > float(current_1h[1])
    sell_confirm = float(current_1h[4]) < float(current_1h[1])

    if float(current_1h[3]) <= buy_zone and buy_confirm:
        risk = entry - (range_low - atr_4h * model.params.range_stop_buffer_atr)
        reward = target_price - entry
    elif float(current_1h[2]) >= sell_zone and sell_confirm:
        risk = (range_high + atr_4h * model.params.range_stop_buffer_atr) - entry
        reward = entry - target_price
    else:
        return "no_range_setup:no_range_touch"

    if risk <= 0 or reward <= 0:
        return "no_range_setup:invalid_risk_reward"
    if risk > atr_4h * MAX_STOP_ATR * model.params.range_max_stop_multiplier:
        return "no_range_setup:stop_too_wide"
    if reward / risk < model.params.range_min_rr:
        return f"no_range_setup:rr_too_low rr={reward / risk:.3f}<{model.params.range_min_rr:.3f}"
    return "no_range_setup:unknown"


def _explain_strategy_none(model: Any, klines_1h: List[List[Any]], klines_4h: List[List[Any]], regime_4h: str) -> str:
    if regime_4h == "FLAT":
        return _explain_range_setup_none(model, klines_1h, klines_4h)
    return _explain_trend_setup_none(model, klines_1h, klines_4h)


def normalize_symbols_list(raw: str) -> List[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def selected_models(symbols: List[str]):
    selected = {symbol.upper() for symbol in symbols}
    return [model for model in MODEL_SPECS if model.symbol in selected]


def load_forward_signal_keys() -> Set[Tuple[str, str, int]]:
    keys: Set[Tuple[str, str, int]] = set()
    for row in load_journal_rows(JOURNAL_PATH):
        if str(row.get("forward_runner", "")) != FORWARD_RUNNER_NAME:
            continue
        symbol = str(row.get("symbol", "")).upper()
        model_name = str(row.get("model_name", ""))
        try:
            signal_time_ms = int(row.get("signal_time_ms", 0) or 0)
        except (TypeError, ValueError):
            signal_time_ms = 0
        if symbol and model_name and signal_time_ms > 0:
            keys.add((symbol, model_name, signal_time_ms))
    return keys


def count_open_positions(client: BybitClient, models: List[Any]) -> int:
    count = 0
    for model in models:
        if client.get_position(model.symbol):
            count += 1
    return count


def validate_runtime(client: BybitClient, models: List[Any], btc_symbol: str) -> Tuple[List[Any], str]:
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise ValueError("Не заданы BYBIT_API_KEY / BYBIT_API_SECRET в .env")
    if MODE != "DEMO":
        logger.warning("Скрипт рассчитан на DEMO-режим, сейчас MODE=%s", MODE)

    model_symbols = [model.symbol for model in models]
    valid_symbols = validate_symbols(client, model_symbols)
    valid_models = [model for model in models if model.symbol in valid_symbols]
    if not valid_models:
        raise ValueError("После валидации не осталось ни одной модели")

    btc_valid = validate_symbols(client, [btc_symbol])
    if not btc_valid:
        raise ValueError(f"BTC-фильтр символ {btc_symbol} недоступен")

    return valid_models, btc_valid[0]


def process_model(
    client: BybitClient,
    model: Any,
    btc_4h: List[List[Any]],
    btc_times_4h: List[int],
    signal_keys: Set[Tuple[str, str, int]],
    btc_mode: str,
    min_hot_vol_ratio: float,
) -> bool:
    constraints = get_symbol_constraints(client, model.symbol)
    if not constraints:
        logger.info("%s[%s]: не удалось получить параметры инструмента", model.symbol, model.name)
        return False

    existing_pos = client.get_position(model.symbol)
    if existing_pos:
        logger.info(
            "%s[%s]: уже есть открытая позиция side=%s size=%s",
            model.symbol,
            model.name,
            existing_pos.get("side"),
            existing_pos.get("size"),
        )
        return False

    klines_4h = client.get_kline(model.symbol, TIMEFRAME_TREND, limit=250, closed_only=True)
    klines_1h = client.get_kline(model.symbol, TIMEFRAME_SIGNAL, limit=80, closed_only=True)
    if len(klines_4h) < 50 or len(klines_1h) < 30:
        logger.info(
            "%s[%s]: regime_4h=n/a hot_vol_ratio=n/a hot_vol_pass=n/a btc_filter=n/a reject=insufficient_history 4h=%s 1h=%s",
            model.symbol,
            model.name,
            len(klines_4h),
            len(klines_1h),
        )
        return False

    regime_4h = _current_regime_4h(model, klines_4h)
    symbol_times_4h = build_4h_index(klines_4h)
    signal_time_ms = int(klines_1h[-1][0])
    hot_vol_ratio = hot_vol_ratio_at(klines_4h, symbol_times_4h, signal_time_ms)
    hot_vol_pass = not (min_hot_vol_ratio > 0 and (hot_vol_ratio is None or hot_vol_ratio < min_hot_vol_ratio))
    preview_side = _expected_side_for_regime(regime_4h)
    btc_filter_preview = (
        btc_mode_ok(btc_mode, preview_side, btc_4h, btc_times_4h, signal_time_ms)
        if preview_side is not None else None
    )

    strategy_fn = make_regime_switch_strategy(model.params)
    setup = strategy_fn(klines_1h, len(klines_1h) - 1, klines_4h, len(klines_4h) - 1)
    if setup is None:
        logger.info(
            "%s[%s]: regime_4h=%s hot_vol_ratio=%s hot_vol_pass=%s btc_filter=%s setup=None reason=%s",
            model.symbol,
            model.name,
            regime_4h,
            _fmt_ratio(hot_vol_ratio),
            hot_vol_pass,
            _fmt_btc_state(btc_filter_preview, preview_side),
            _explain_strategy_none(model, klines_1h, klines_4h, regime_4h),
        )
        return False

    actual_btc_state = btc_mode_ok(btc_mode, setup.side, btc_4h, btc_times_4h, signal_time_ms)
    logger.info(
        "%s[%s]: regime_4h=%s side=%s sub_strategy=%s hot_vol_ratio=%s hot_vol_pass=%s btc_filter=%s setup=found",
        model.symbol,
        model.name,
        regime_4h,
        setup.side,
        setup.meta.get("sub_strategy"),
        _fmt_ratio(hot_vol_ratio),
        hot_vol_pass,
        _fmt_btc_state(actual_btc_state, setup.side),
    )

    signal_key = (model.symbol, model.name, signal_time_ms)
    if signal_key in signal_keys:
        logger.info(
            "%s[%s]: reject=duplicate_signal signal_time_ms=%s",
            model.symbol,
            model.name,
            signal_time_ms,
        )
        return False

    if not actual_btc_state:
        logger.info(
            "%s[%s]: reject=btc_market_filter mode=%s side=%s",
            model.symbol,
            model.name,
            btc_mode,
            setup.side,
        )
        return False

    if min_hot_vol_ratio > 0 and (hot_vol_ratio is None or hot_vol_ratio < min_hot_vol_ratio):
        logger.info(
            "%s[%s]: reject=hot_vol_ratio value=%s threshold=%.2f",
            model.symbol,
            model.name,
            _fmt_ratio(hot_vol_ratio),
            min_hot_vol_ratio,
        )
        return False

    live_price = client.get_last_price(model.symbol)
    if live_price <= 0:
        logger.info("%s[%s]: reject=no_live_price", model.symbol, model.name)
        return False

    if setup.side == "Buy":
        stop_dec, stop_str = normalize_price(setup.stop, constraints, ROUND_DOWN)
        tp_dec, tp_str = normalize_price(setup.tp, constraints, ROUND_UP)
    else:
        stop_dec, stop_str = normalize_price(setup.stop, constraints, ROUND_UP)
        tp_dec, tp_str = normalize_price(setup.tp, constraints, ROUND_DOWN)

    stop = float(stop_dec)
    tp = float(tp_dec)
    if stop <= 0 or tp <= 0:
        logger.info("%s[%s]: reject=invalid_normalized_stop_tp", model.symbol, model.name)
        return False

    if setup.side == "Buy":
        risk = live_price - stop
        reward = tp - live_price
    else:
        risk = stop - live_price
        reward = live_price - tp

    if risk <= 0 or reward <= 0:
        logger.info(
            "%s[%s]: reject=invalid_live_drift entry=%.6f stop=%s tp=%s",
            model.symbol,
            model.name,
            live_price,
            stop_str,
            tp_str,
        )
        return False

    balance_info = client.get_usdt_balance_info("USDT")
    wallet_balance = float(balance_info.get("walletBalance", 0) or 0)
    available_balance = float(balance_info.get("availableBalance", 0) or 0)
    equity_balance = float(balance_info.get("equity", 0) or 0)

    margin_balance = available_balance if available_balance > 0 else wallet_balance
    if margin_balance <= 0:
        margin_balance = equity_balance
    risk_balance_candidates = [value for value in (wallet_balance, margin_balance) if value > 0]
    risk_balance = min(risk_balance_candidates) if risk_balance_candidates else 0.0
    if risk_balance <= 0:
        logger.info("%s[%s]: reject=no_usable_balance", model.symbol, model.name)
        return False

    qty_by_risk = calc_qty_by_risk(risk_balance, RISK_PER_TRADE, live_price, stop)
    qty_by_margin = calc_qty_by_margin(margin_balance, LEVERAGE, live_price)
    qty_raw = min(qty_by_risk, qty_by_margin)
    if qty_raw <= 0:
        logger.info("%s[%s]: reject=qty_raw_le_zero", model.symbol, model.name)
        return False

    qty_dec, qty_str = normalize_qty(qty_raw, constraints, order_type="Market")
    if qty_dec <= 0:
        logger.info("%s[%s]: reject=qty_below_exchange_constraints", model.symbol, model.name)
        return False

    planned_risk_usdt = float(qty_dec) * risk
    notional = qty_dec * stop_dec.__class__(str(live_price))
    if notional < constraints["minNotionalValue"]:
        logger.info(
            "%s[%s]: reject=min_notional notional=%s min=%s",
            model.symbol,
            model.name,
            notional,
            constraints["minNotionalValue"],
        )
        return False

    if setup.side == "Buy" and stop >= live_price:
        logger.info("%s[%s]: reject=stop_ge_live_price", model.symbol, model.name)
        return False
    if setup.side == "Sell" and stop <= live_price:
        logger.info("%s[%s]: reject=stop_le_live_price", model.symbol, model.name)
        return False

    try:
        client.set_leverage(model.symbol, LEVERAGE)
    except Exception as exc:
        logger.warning("%s[%s]: ошибка установки плеча: %s", model.symbol, model.name, exc)

    order_resp = client.place_order(
        symbol=model.symbol,
        side=setup.side,
        qty=qty_str,
        order_type="Market",
        position_idx=POSITION_IDX,
        order_link_id=build_order_link_id(model.symbol, setup.side, "meta-open"),
    )
    logger.info("%s[%s]: ответ на open order: %s", model.symbol, model.name, order_resp)
    if order_resp.get("retCode") != 0:
        return False

    order_result = order_resp.get("result", {}) or {}
    order_id = str(order_result.get("orderId", "") or "")
    order_link_id = str(order_result.get("orderLinkId", "") or "")

    journal_base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": model.symbol,
        "side": setup.side,
        "entry": round(live_price, 8),
        "stop": round(stop, 8),
        "tp": round(tp, 8),
        "risk_usdt": round(planned_risk_usdt, 4),
        "planned_rr": round(reward / risk, 4),
        "atr_4h": round(setup.atr_4h, 8),
        "trend": str(setup.meta.get("trend", "")),
        "level": round(setup.level, 8),
        "impulse_ok": bool(setup.impulse_ok),
    }
    journal_meta = {
        "message": "meta portfolio forward",
        "forward_runner": FORWARD_RUNNER_NAME,
        "model_name": model.name,
        "signal_time_ms": signal_time_ms,
        "signal_entry": round(setup.entry, 8),
        "signal_stop": round(setup.stop, 8),
        "signal_tp": round(setup.tp, 8),
        "btc_mode": btc_mode,
        "hot_vol_ratio_filter": min_hot_vol_ratio,
        "hot_vol_ratio_value": None if hot_vol_ratio is None else round(hot_vol_ratio, 6),
        "wallet_balance": round(wallet_balance, 8),
        "available_balance": round(available_balance, 8),
        "equity_balance": round(equity_balance, 8),
        "risk_balance": round(risk_balance, 8),
        "margin_balance": round(margin_balance, 8),
        "qty": qty_str,
        "order_id": order_id,
        "order_link_id": order_link_id,
        "strategy": setup.strategy,
        "sub_strategy": setup.meta.get("sub_strategy"),
        "regime": setup.meta.get("regime"),
        "planned_risk_usdt": round(planned_risk_usdt, 8),
        "model_params": {
            "trend_threshold": model.params.trend_threshold,
            "trend_min_distance_score": model.params.trend_min_distance_score,
            "trend_rr_target": model.params.trend_rr_target,
            "trend_stop_buffer_atr": model.params.trend_stop_buffer_atr,
            "range_zone_fraction": model.params.range_zone_fraction,
            "range_target_fraction": model.params.range_target_fraction,
        },
        **setup.meta,
    }

    def journal(status: str, extra: Dict[str, Any] | None = None) -> None:
        payload = dict(journal_meta)
        if extra:
            payload.update(extra)
        append_journal_row(
            {
                **journal_base,
                "status": status,
                "note": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
        )

    position = wait_for_position(client, model.symbol, setup.side)
    if not position:
        logger.info("%s[%s]: позиция не появилась после ACK", model.symbol, model.name)
        journal("entry_ack_no_position")
        return False

    actual_position_idx = int(position.get("positionIdx", POSITION_IDX))
    ts_resp = client.set_trading_stop(
        symbol=model.symbol,
        position_idx=actual_position_idx,
        stop_loss=stop_str,
        take_profit=tp_str,
        trigger_by=TRIGGER_BY,
    )
    logger.info("%s[%s]: установка SL/TP: %s", model.symbol, model.name, ts_resp)
    if ts_resp.get("retCode") != 0:
        close_resp = force_close_position(client, model.symbol, setup.side, qty_str, actual_position_idx)
        logger.info("%s[%s]: аварийное закрытие позиции: %s", model.symbol, model.name, close_resp)
        journal("force_closed_trading_stop_error", {"trading_stop_response": ts_resp, "close_response": close_resp})
        return False

    journal("opened_with_sl_tp", {"trading_stop_response": ts_resp, "position_idx": actual_position_idx})
    signal_keys.add(signal_key)
    logger.info(
        "%s[%s]: OPEN %s entry=%.6f stop=%s tp=%s rr=%.2f risk=%.2f hot_vol=%s regime=%s",
        model.symbol,
        model.name,
        setup.side,
        live_price,
        stop_str,
        tp_str,
        reward / risk,
        planned_risk_usdt,
        "n/a" if hot_vol_ratio is None else f"{hot_vol_ratio:.3f}",
        setup.meta.get("regime"),
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--btc-symbol", default=DEFAULT_BTC_SYMBOL)
    parser.add_argument("--btc-mode", default=DEFAULT_BTC_MODE)
    parser.add_argument("--min-hot-vol-ratio", type=float, default=DEFAULT_MIN_HOT_VOL_RATIO)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--sleep-seconds", type=int, default=DEFAULT_SLEEP_SECONDS)
    args = parser.parse_args()

    models = selected_models(normalize_symbols_list(args.symbols))
    client = BybitClient(BYBIT_API_KEY, BYBIT_API_SECRET, BASE_URL)
    models, btc_symbol = validate_runtime(client, models, args.btc_symbol)

    synced_rows = sync_closed_trades_to_journal(client)
    if synced_rows > 0:
        logger.info("Журнал синхронизирован: обновлено закрытых сделок %s", synced_rows)

    logger.info(
        "Старт meta forward MODE=%s base_url=%s leverage=%s risk=%.2f%% config=%s models=%s btc_mode=%s hot_vol>=%.2f max_concurrent=%s",
        MODE,
        BASE_URL,
        LEVERAGE,
        RISK_PER_TRADE * 100,
        os.path.basename(CONFIG_SOURCE),
        ",".join(model.symbol for model in models),
        args.btc_mode,
        args.min_hot_vol_ratio,
        args.max_concurrent,
    )

    while True:
        synced_rows = sync_closed_trades_to_journal(client)
        if synced_rows > 0:
            logger.info("Журнал синхронизирован: обновлено закрытых сделок %s", synced_rows)

        btc_4h = client.get_kline(btc_symbol, TIMEFRAME_TREND, limit=250, closed_only=True)
        if len(btc_4h) < 30:
            logger.warning("%s: недостаточно BTC 4H свечей для market filter", btc_symbol)
            time.sleep(args.sleep_seconds)
            continue

        btc_times_4h = build_4h_index(btc_4h)
        signal_keys = load_forward_signal_keys()
        open_count = count_open_positions(client, models)

        logger.info("Новый цикл: открытых позиций в meta-портфеле=%s/%s", open_count, args.max_concurrent)

        for model in models:
            if open_count >= args.max_concurrent:
                logger.info("Лимит открытых позиций достигнут: %s/%s", open_count, args.max_concurrent)
                break
            try:
                opened = process_model(
                    client=client,
                    model=model,
                    btc_4h=btc_4h,
                    btc_times_4h=btc_times_4h,
                    signal_keys=signal_keys,
                    btc_mode=args.btc_mode,
                    min_hot_vol_ratio=args.min_hot_vol_ratio,
                )
                if opened:
                    open_count += 1
            except Exception as exc:
                logger.exception("Ошибка при обработке %s[%s]: %s", model.symbol, model.name, exc)

        logger.info("Цикл meta-портфеля завершён, спим %s секунд", args.sleep_seconds)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
