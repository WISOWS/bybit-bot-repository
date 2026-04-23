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
)
from research_search_meta_portfolio import MODEL_SPECS, btc_mode_ok, build_4h_index, hot_vol_ratio_at
from research_strategies import make_regime_switch_strategy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "meta_portfolio.log")

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
        logger.info("%s[%s]: недостаточно закрытых свечей", model.symbol, model.name)
        return False

    strategy_fn = make_regime_switch_strategy(model.params)
    setup = strategy_fn(klines_1h, len(klines_1h) - 1, klines_4h, len(klines_4h) - 1)
    if setup is None:
        return False

    signal_time_ms = int(klines_1h[-1][0])
    signal_key = (model.symbol, model.name, signal_time_ms)
    if signal_key in signal_keys:
        logger.info("%s[%s]: сигнал уже был обработан для свечи %s", model.symbol, model.name, signal_time_ms)
        return False

    if not btc_mode_ok(btc_mode, setup.side, btc_4h, btc_times_4h, signal_time_ms):
        logger.info("%s[%s]: BTC market filter=%s, NO TRADE", model.symbol, model.name, btc_mode)
        return False

    symbol_times_4h = build_4h_index(klines_4h)
    hot_vol_ratio = hot_vol_ratio_at(klines_4h, symbol_times_4h, signal_time_ms)
    if min_hot_vol_ratio > 0 and (hot_vol_ratio is None or hot_vol_ratio < min_hot_vol_ratio):
        logger.info(
            "%s[%s]: hot_vol_ratio=%s < %.2f, NO TRADE",
            model.symbol,
            model.name,
            "n/a" if hot_vol_ratio is None else f"{hot_vol_ratio:.3f}",
            min_hot_vol_ratio,
        )
        return False

    live_price = client.get_last_price(model.symbol)
    if live_price <= 0:
        logger.info("%s[%s]: не удалось получить актуальную цену", model.symbol, model.name)
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
        logger.info("%s[%s]: stop/tp после нормализации некорректны", model.symbol, model.name)
        return False

    if setup.side == "Buy":
        risk = live_price - stop
        reward = tp - live_price
    else:
        risk = stop - live_price
        reward = live_price - tp

    if risk <= 0 or reward <= 0:
        logger.info(
            "%s[%s]: live drift сделал сделку некорректной entry=%.6f stop=%s tp=%s",
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
        logger.info("%s[%s]: не удалось определить usable balance", model.symbol, model.name)
        return False

    qty_by_risk = calc_qty_by_risk(risk_balance, RISK_PER_TRADE, live_price, stop)
    qty_by_margin = calc_qty_by_margin(margin_balance, LEVERAGE, live_price)
    qty_raw = min(qty_by_risk, qty_by_margin)
    if qty_raw <= 0:
        logger.info("%s[%s]: qty_raw <= 0", model.symbol, model.name)
        return False

    qty_dec, qty_str = normalize_qty(qty_raw, constraints, order_type="Market")
    if qty_dec <= 0:
        logger.info("%s[%s]: размер позиции не проходит по шагу/минимуму", model.symbol, model.name)
        return False

    planned_risk_usdt = float(qty_dec) * risk
    notional = qty_dec * stop_dec.__class__(str(live_price))
    if notional < constraints["minNotionalValue"]:
        logger.info("%s[%s]: notional %s < minNotionalValue %s", model.symbol, model.name, notional, constraints["minNotionalValue"])
        return False

    if setup.side == "Buy" and stop >= live_price:
        logger.info("%s[%s]: stop >= live price, NO TRADE", model.symbol, model.name)
        return False
    if setup.side == "Sell" and stop <= live_price:
        logger.info("%s[%s]: stop <= live price, NO TRADE", model.symbol, model.name)
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
