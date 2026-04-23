import argparse
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backtest import (
    BACKTEST_TAKER_FEE_RATE,
    apply_adverse_slippage,
    load_ohlcv_csv,
    simulate_exit,
)
from main import BASE_DIR, LEVERAGE, RISK_PER_TRADE, calc_atr_from_klines, calc_qty_by_margin, calc_qty_by_risk, detect_trend_4h
from research_backtest import normalize_symbols, summarize_rows
from research_search import annualized_return_pct, split_rows_by_time, summarize_portfolio
from research_strategies import (
    bullish_close,
    bearish_close,
    candle_body_fraction,
    candle_close,
    candle_high,
    candle_low,
    candle_range_value,
    ema,
)


DEFAULT_SYMBOLS = "NEARUSDT,SOLUSDT,ENAUSDT,LINKUSDT"


@dataclass(frozen=True)
class IntradayParams:
    breakout_lookback_15m: int = 8
    pullback_lookback_15m: int = 4
    min_pullback_atr_1h: float = 0.15
    ema_fast_period_1h: int = 10
    ema_slow_period_1h: int = 30
    rr_target: float = 3.0
    stop_buffer_atr_1h: float = 0.05
    max_stop_multiplier_atr_1h: float = 1.2
    min_body_fraction_15m: float = 0.55
    min_range_expansion_15m: float = 1.2
    btc_filter_mode: str = "off"
    min_hot_vol_ratio_4h: float = 0.0


@dataclass
class IntradaySetup:
    side: str
    entry: float
    stop: float
    tp: float
    atr_1h: float
    planned_rr: float
    meta: Dict[str, Any]


def ms_to_iso(ms_value: int) -> str:
    return datetime.fromtimestamp(ms_value / 1000, tz=timezone.utc).isoformat()


def load_symbol_data(ohlcv_dir: str, symbols: List[str]) -> Dict[str, Dict[str, List[List[Any]]]]:
    loaded: Dict[str, Dict[str, List[List[Any]]]] = {}
    for symbol in symbols:
        file_15m = os.path.join(ohlcv_dir, f"{symbol}_15.csv")
        file_1h = os.path.join(ohlcv_dir, f"{symbol}_60.csv")
        file_4h = os.path.join(ohlcv_dir, f"{symbol}_240.csv")
        if not os.path.exists(file_15m) or not os.path.exists(file_1h) or not os.path.exists(file_4h):
            continue
        klines_15m = load_ohlcv_csv(file_15m)
        klines_1h = load_ohlcv_csv(file_1h)
        klines_4h = load_ohlcv_csv(file_4h)
        if len(klines_15m) < 200 or len(klines_1h) < 100 or len(klines_4h) < 50:
            continue
        loaded[symbol] = {"15m": klines_15m, "1h": klines_1h, "4h": klines_4h}
    return loaded


def find_global_period(loaded_data: Dict[str, Dict[str, List[List[Any]]]]) -> Tuple[datetime, datetime]:
    starts: List[int] = []
    ends: List[int] = []
    for payload in loaded_data.values():
        klines_15m = payload["15m"]
        starts.append(int(klines_15m[0][0]))
        ends.append(int(klines_15m[-1][0]))
    return (
        datetime.fromtimestamp(min(starts) / 1000, tz=timezone.utc),
        datetime.fromtimestamp(max(ends) / 1000, tz=timezone.utc),
    )


def latest_idx(times: List[int], timestamp_ms: int) -> int:
    lo = 0
    hi = len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] <= timestamp_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def hot_vol_ratio_4h(klines_4h: List[List[Any]], idx_4h: int) -> Optional[float]:
    if idx_4h < 14:
        return None
    short_atr = calc_atr_from_klines(klines_4h[idx_4h - 6 : idx_4h + 1], period=6)
    long_atr = calc_atr_from_klines(klines_4h[idx_4h - 14 : idx_4h + 1], period=14)
    if short_atr <= 0 or long_atr <= 0:
        return None
    return short_atr / long_atr


def btc_filter_ok(side: str, btc_4h: List[List[Any]], idx_btc_4h: int, mode: str) -> bool:
    if mode == "off":
        return True
    if idx_btc_4h < 29:
        return False
    trend = detect_trend_4h(btc_4h[idx_btc_4h - 29 : idx_btc_4h + 1])
    if mode == "allow_flat":
        if side == "Buy":
            return trend in {"UP", "FLAT"}
        return trend in {"DOWN", "FLAT"}
    return True


def build_setup(
    klines_15m: List[List[Any]],
    idx_15m: int,
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
    btc_4h: List[List[Any]],
    idx_btc_4h: int,
    params: IntradayParams,
) -> Optional[IntradaySetup]:
    min_15m_idx = max(
        params.breakout_lookback_15m,
        params.pullback_lookback_15m,
        params.ema_slow_period_1h,
        6,
    )
    if idx_15m < min_15m_idx or idx_1h < max(params.ema_slow_period_1h - 1, 14) or idx_4h < 29:
        return None

    recent_4h = klines_4h[idx_4h - 29 : idx_4h + 1]
    trend_4h = detect_trend_4h(recent_4h)
    if trend_4h == "FLAT":
        return None

    recent_1h_closes = [candle_close(candle) for candle in klines_1h[idx_1h - params.ema_slow_period_1h + 1 : idx_1h + 1]]
    ema_fast = ema(recent_1h_closes, params.ema_fast_period_1h)
    ema_slow = ema(recent_1h_closes, params.ema_slow_period_1h)
    last_1h_close = candle_close(klines_1h[idx_1h])
    atr_1h = calc_atr_from_klines(klines_1h[idx_1h - 14 : idx_1h + 1], period=14)
    if atr_1h <= 0:
        return None

    hot_vol = hot_vol_ratio_4h(klines_4h, idx_4h)
    if params.min_hot_vol_ratio_4h > 0:
        if hot_vol is None or hot_vol < params.min_hot_vol_ratio_4h:
            return None

    current_15m = klines_15m[idx_15m]
    recent_breakout = klines_15m[idx_15m - params.breakout_lookback_15m : idx_15m]
    pullback_window = klines_15m[idx_15m - params.pullback_lookback_15m : idx_15m]
    avg_recent_range = sum(candle_range_value(candle) for candle in recent_breakout) / len(recent_breakout)
    current_range = candle_range_value(current_15m)
    if avg_recent_range <= 0:
        return None
    expansion_factor = current_range / avg_recent_range
    if expansion_factor < params.min_range_expansion_15m:
        return None

    body_fraction = candle_body_fraction(current_15m)
    if body_fraction < params.min_body_fraction_15m:
        return None

    if trend_4h == "UP":
        if not (ema_fast > ema_slow and last_1h_close >= ema_fast):
            return None
        if not bullish_close(current_15m):
            return None
        breakout_level = max(candle_high(candle) for candle in recent_breakout)
        pullback_low = min(candle_low(candle) for candle in pullback_window)
        if breakout_level - pullback_low < atr_1h * params.min_pullback_atr_1h:
            return None
        entry = candle_close(current_15m)
        if entry <= breakout_level:
            return None
        stop = min(candle_low(candle) for candle in klines_15m[idx_15m - params.pullback_lookback_15m : idx_15m + 1]) - atr_1h * params.stop_buffer_atr_1h
        risk = entry - stop
        if risk <= 0 or risk > atr_1h * params.max_stop_multiplier_atr_1h:
            return None
        if not btc_filter_ok("Buy", btc_4h, idx_btc_4h, params.btc_filter_mode):
            return None
        tp = entry + risk * params.rr_target
        return IntradaySetup(
            side="Buy",
            entry=entry,
            stop=stop,
            tp=tp,
            atr_1h=atr_1h,
            planned_rr=params.rr_target,
            meta={
                "trend_4h": trend_4h,
                "ema_fast_1h": round(ema_fast, 8),
                "ema_slow_1h": round(ema_slow, 8),
                "body_fraction_15m": round(body_fraction, 6),
                "expansion_factor_15m": round(expansion_factor, 6),
                "pullback_size_atr_1h": round((breakout_level - pullback_low) / atr_1h, 6),
                "hot_vol_ratio_4h": round(hot_vol, 6) if hot_vol is not None else None,
            },
        )

    if not (ema_fast < ema_slow and last_1h_close <= ema_fast):
        return None
    if not bearish_close(current_15m):
        return None
    breakout_level = min(candle_low(candle) for candle in recent_breakout)
    pullback_high = max(candle_high(candle) for candle in pullback_window)
    if pullback_high - breakout_level < atr_1h * params.min_pullback_atr_1h:
        return None
    entry = candle_close(current_15m)
    if entry >= breakout_level:
        return None
    stop = max(candle_high(candle) for candle in klines_15m[idx_15m - params.pullback_lookback_15m : idx_15m + 1]) + atr_1h * params.stop_buffer_atr_1h
    risk = stop - entry
    if risk <= 0 or risk > atr_1h * params.max_stop_multiplier_atr_1h:
        return None
    if not btc_filter_ok("Sell", btc_4h, idx_btc_4h, params.btc_filter_mode):
        return None
    tp = entry - risk * params.rr_target
    return IntradaySetup(
        side="Sell",
        entry=entry,
        stop=stop,
        tp=tp,
        atr_1h=atr_1h,
        planned_rr=params.rr_target,
        meta={
            "trend_4h": trend_4h,
            "ema_fast_1h": round(ema_fast, 8),
            "ema_slow_1h": round(ema_slow, 8),
            "body_fraction_15m": round(body_fraction, 6),
            "expansion_factor_15m": round(expansion_factor, 6),
            "pullback_size_atr_1h": round((pullback_high - breakout_level) / atr_1h, 6),
            "hot_vol_ratio_4h": round(hot_vol, 6) if hot_vol is not None else None,
        },
    )


def run_intraday_backtest(
    symbol: str,
    klines_15m: List[List[Any]],
    klines_1h: List[List[Any]],
    klines_4h: List[List[Any]],
    btc_4h: List[List[Any]],
    initial_balance: float,
    params: IntradayParams,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    balance = initial_balance
    rows: List[Dict[str, Any]] = []
    times_1h = [int(row[0]) for row in klines_1h]
    times_4h = [int(row[0]) for row in klines_4h]
    btc_times_4h = [int(row[0]) for row in btc_4h]
    idx_15m = max(
        params.breakout_lookback_15m,
        params.pullback_lookback_15m,
        params.ema_slow_period_1h,
        60,
    )

    while idx_15m < len(klines_15m) - 1:
        current_time_ms = int(klines_15m[idx_15m][0])
        idx_1h = latest_idx(times_1h, current_time_ms)
        idx_4h = latest_idx(times_4h, current_time_ms)
        idx_btc_4h = latest_idx(btc_times_4h, current_time_ms)
        if idx_1h < 0 or idx_4h < 0 or idx_btc_4h < 0:
            idx_15m += 1
            continue

        setup = build_setup(
            klines_15m=klines_15m,
            idx_15m=idx_15m,
            klines_1h=klines_1h,
            idx_1h=idx_1h,
            klines_4h=klines_4h,
            idx_4h=idx_4h,
            btc_4h=btc_4h,
            idx_btc_4h=idx_btc_4h,
            params=params,
        )
        if setup is None:
            idx_15m += 1
            continue

        risk_per_unit = abs(setup.entry - setup.stop)
        if risk_per_unit <= 0:
            idx_15m += 1
            continue

        qty_by_risk = calc_qty_by_risk(balance, RISK_PER_TRADE, setup.entry, setup.stop)
        qty_by_margin = calc_qty_by_margin(balance, LEVERAGE, setup.entry)
        qty = min(qty_by_risk, qty_by_margin)
        if qty <= 0:
            idx_15m += 1
            continue

        effective_entry = apply_adverse_slippage(setup.entry, setup.side, "entry")
        exit_price, exit_reason, exit_index = simulate_exit(klines_15m, idx_15m, setup.side, setup.stop, setup.tp)
        effective_exit = apply_adverse_slippage(exit_price, setup.side, "exit")
        pnl_per_unit = effective_exit - effective_entry if setup.side == "Buy" else effective_entry - effective_exit
        entry_fee = qty * effective_entry * BACKTEST_TAKER_FEE_RATE
        exit_fee = qty * effective_exit * BACKTEST_TAKER_FEE_RATE
        realized_pnl = qty * pnl_per_unit - entry_fee - exit_fee
        risk_usdt = qty * risk_per_unit
        balance += realized_pnl

        note_payload = {
            "strategy": "intraday_15m_continuation",
            "effective_entry": round(effective_entry, 8),
            "effective_exit": round(effective_exit, 8),
            "entry_fee": round(entry_fee, 8),
            "exit_fee": round(exit_fee, 8),
            "qty": round(qty, 8),
            "entry_time_ms": current_time_ms,
            "exit_time_ms": int(klines_15m[exit_index][0]),
            "exit_reason": exit_reason,
            "balance_after": round(balance, 8),
            "params": {
                "breakout_lookback_15m": params.breakout_lookback_15m,
                "pullback_lookback_15m": params.pullback_lookback_15m,
                "min_pullback_atr_1h": params.min_pullback_atr_1h,
                "ema_fast_period_1h": params.ema_fast_period_1h,
                "ema_slow_period_1h": params.ema_slow_period_1h,
                "rr_target": params.rr_target,
                "stop_buffer_atr_1h": params.stop_buffer_atr_1h,
                "max_stop_multiplier_atr_1h": params.max_stop_multiplier_atr_1h,
                "min_body_fraction_15m": params.min_body_fraction_15m,
                "min_range_expansion_15m": params.min_range_expansion_15m,
                "btc_filter_mode": params.btc_filter_mode,
                "min_hot_vol_ratio_4h": params.min_hot_vol_ratio_4h,
            },
            **setup.meta,
        }

        rows.append(
            {
                "strategy": "intraday_15m_continuation",
                "timestamp": ms_to_iso(current_time_ms),
                "symbol": symbol,
                "side": setup.side,
                "entry": round(setup.entry, 8),
                "stop": round(setup.stop, 8),
                "tp": round(setup.tp, 8),
                "risk_usdt": round(risk_usdt, 8),
                "planned_rr": round(setup.planned_rr, 4),
                "atr_4h": round(setup.atr_1h, 8),
                "trend": str(setup.meta.get("trend_4h", "")),
                "status": "closed_research",
                "realized_pnl": round(realized_pnl, 8),
                "note": json.dumps(note_payload, ensure_ascii=False, sort_keys=True),
            }
        )

        idx_15m = exit_index + 1

    return rows, summarize_rows(rows, initial_balance)


def parse_ema_pairs(raw: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for part in raw.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        fast_raw, slow_raw = item.split(":", 1)
        fast = int(fast_raw)
        slow = int(slow_raw)
        if fast >= slow:
            continue
        pairs.append((fast, slow))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--btc-symbol", default="BTCUSDT")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_intraday"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--breakout-lookbacks", default="8,12")
    parser.add_argument("--pullback-lookbacks", default="4,6")
    parser.add_argument("--pullback-atr-fractions", default="0.15,0.25")
    parser.add_argument("--ema-pairs", default="10:30,20:50")
    parser.add_argument("--rr-targets", default="2.5,3.0")
    parser.add_argument("--stop-buffers", default="0.05,0.10")
    parser.add_argument("--body-fractions", default="0.55,0.70")
    parser.add_argument("--range-expansions", default="1.2,1.4")
    parser.add_argument("--btc-filter-modes", default="off,allow_flat")
    parser.add_argument("--hot-vol-ratios", default="0.0,0.8")
    parser.add_argument("--top-n", default="1,2,3")
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--min-test-trades", type=int, default=6)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.1)
    parser.add_argument("--top-results", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_symbols = normalize_symbols(args.symbols.split(","))
    loaded_data = load_symbol_data(args.ohlcv_dir, selected_symbols)
    if not loaded_data:
        raise ValueError("Не удалось загрузить multi-timeframe OHLCV для intraday-исследования")

    btc_4h_path = os.path.join(args.ohlcv_dir, f"{args.btc_symbol}_240.csv")
    if not os.path.exists(btc_4h_path):
        raise FileNotFoundError(f"Нет BTC 4H файла: {btc_4h_path}")
    btc_4h = load_ohlcv_csv(btc_4h_path)

    global_start, global_end = find_global_period(loaded_data)
    split_dt = global_start + (global_end - global_start) * args.split_ratio
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    breakout_lookbacks = [int(v) for v in args.breakout_lookbacks.split(",") if v.strip()]
    pullback_lookbacks = [int(v) for v in args.pullback_lookbacks.split(",") if v.strip()]
    pullback_atr_fractions = [float(v) for v in args.pullback_atr_fractions.split(",") if v.strip()]
    ema_pairs = parse_ema_pairs(args.ema_pairs)
    rr_targets = [float(v) for v in args.rr_targets.split(",") if v.strip()]
    stop_buffers = [float(v) for v in args.stop_buffers.split(",") if v.strip()]
    body_fractions = [float(v) for v in args.body_fractions.split(",") if v.strip()]
    range_expansions = [float(v) for v in args.range_expansions.split(",") if v.strip()]
    btc_filter_modes = [v.strip() for v in args.btc_filter_modes.split(",") if v.strip()]
    hot_vol_ratios = [float(v) for v in args.hot_vol_ratios.split(",") if v.strip()]
    top_n_values = [int(v) for v in args.top_n.split(",") if v.strip()]

    grid = list(
        itertools.product(
            breakout_lookbacks,
            pullback_lookbacks,
            pullback_atr_fractions,
            ema_pairs,
            rr_targets,
            stop_buffers,
            body_fractions,
            range_expansions,
            btc_filter_modes,
            hot_vol_ratios,
        )
    )

    search_results: List[Dict[str, Any]] = []

    for (
        breakout_lookback,
        pullback_lookback,
        pullback_atr_fraction,
        ema_pair,
        rr_target,
        stop_buffer,
        body_fraction,
        range_expansion,
        btc_filter_mode,
        hot_vol_ratio,
    ) in grid:
        ema_fast, ema_slow = ema_pair
        params = IntradayParams(
            breakout_lookback_15m=breakout_lookback,
            pullback_lookback_15m=pullback_lookback,
            min_pullback_atr_1h=pullback_atr_fraction,
            ema_fast_period_1h=ema_fast,
            ema_slow_period_1h=ema_slow,
            rr_target=rr_target,
            stop_buffer_atr_1h=stop_buffer,
            min_body_fraction_15m=body_fraction,
            min_range_expansion_15m=range_expansion,
            btc_filter_mode=btc_filter_mode,
            min_hot_vol_ratio_4h=hot_vol_ratio,
        )

        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, payload in loaded_data.items():
            rows, _ = run_intraday_backtest(
                symbol=symbol,
                klines_15m=payload["15m"],
                klines_1h=payload["1h"],
                klines_4h=payload["4h"],
                btc_4h=btc_4h,
                initial_balance=args.initial_balance,
                params=params,
            )
            train_rows, test_rows = split_rows_by_time(rows, split_dt)
            train_metrics = summarize_rows(train_rows, args.initial_balance)
            test_metrics = summarize_rows(test_rows, args.initial_balance)
            per_symbol[symbol] = {"train": train_metrics, "test": test_metrics}

        eligible_symbols = []
        for symbol, metrics in per_symbol.items():
            train_metrics = metrics["train"]
            if train_metrics["trades"] < args.min_train_trades:
                continue
            if train_metrics["profit_factor"] < args.min_train_profit_factor:
                continue
            if train_metrics["net_pnl"] <= 0:
                continue
            train_annualized = annualized_return_pct(
                train_metrics["final_balance"], args.initial_balance, train_days
            )
            eligible_symbols.append((train_annualized, train_metrics["net_pnl"], symbol))

        eligible_symbols.sort(reverse=True)

        for top_n in top_n_values:
            picked_symbols = [symbol for _, _, symbol in eligible_symbols[:top_n]]
            if len(picked_symbols) < top_n:
                continue

            train_symbol_metrics = [per_symbol[symbol]["train"] for symbol in picked_symbols]
            test_symbol_metrics = [
                per_symbol[symbol]["test"]
                for symbol in picked_symbols
                if per_symbol[symbol]["test"]["trades"] >= args.min_test_trades
            ]
            if len(test_symbol_metrics) < top_n:
                continue

            train_portfolio = summarize_portfolio(train_symbol_metrics, args.initial_balance, train_days)
            test_portfolio = summarize_portfolio(test_symbol_metrics, args.initial_balance, test_days)
            search_results.append(
                {
                    "params": {
                        "breakout_lookback_15m": breakout_lookback,
                        "pullback_lookback_15m": pullback_lookback,
                        "min_pullback_atr_1h": pullback_atr_fraction,
                        "ema_fast_period_1h": ema_fast,
                        "ema_slow_period_1h": ema_slow,
                        "rr_target": rr_target,
                        "stop_buffer_atr_1h": stop_buffer,
                        "min_body_fraction_15m": body_fraction,
                        "min_range_expansion_15m": range_expansion,
                        "btc_filter_mode": btc_filter_mode,
                        "min_hot_vol_ratio_4h": hot_vol_ratio,
                        "top_n_symbols": top_n,
                    },
                    "split": {
                        "start": global_start.isoformat(),
                        "split": split_dt.isoformat(),
                        "end": global_end.isoformat(),
                        "train_days": round(train_days, 4),
                        "test_days": round(test_days, 4),
                    },
                    "symbols": picked_symbols,
                    "train": train_portfolio,
                    "test": test_portfolio,
                }
            )

    search_results.sort(
        key=lambda item: (
            item["test"]["average_annualized_return_pct"],
            item["test"]["average_profit_factor"],
            -item["test"]["average_max_drawdown_pct"],
        ),
        reverse=True,
    )

    output_payload = {
        "meta": {
            "grid_size": len(grid),
            "symbols_considered": len(loaded_data),
            "results_count": len(search_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": search_results[: args.top_results],
    }

    summary_path = os.path.join(args.output_dir, "intraday_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
