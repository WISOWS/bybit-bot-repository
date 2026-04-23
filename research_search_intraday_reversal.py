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
from main import (
    BASE_DIR,
    LEVERAGE,
    RISK_PER_TRADE,
    calc_atr_from_klines,
    calc_qty_by_margin,
    calc_qty_by_risk,
    detect_trend_4h,
)
from research_backtest import normalize_symbols, summarize_rows
from research_search import annualized_return_pct, split_rows_by_time, summarize_portfolio
from research_search_intraday import find_global_period
from research_strategies import (
    bearish_close,
    bullish_close,
    candle_body_fraction,
    candle_close,
    candle_high,
    candle_low,
    candle_range_value,
    ema,
)


DEFAULT_SYMBOLS = "NEARUSDT,SOLUSDT,ENAUSDT,LINKUSDT"


@dataclass(frozen=True)
class ReversalParams:
    impulse_lookback_15m: int = 12
    min_range_expansion_15m: float = 2.0
    min_volume_ratio_15m: float = 1.5
    min_extension_atr_1h: float = 0.5
    reversal_close_fraction: float = 0.5
    rr_target: float = 2.0
    stop_buffer_atr_1h: float = 0.05
    max_stop_multiplier_atr_1h: float = 1.2
    ema_fast_period_1h: int = 10
    ema_slow_period_1h: int = 30
    min_hot_vol_ratio_4h: float = 0.0
    regime_mode: str = "all"


@dataclass
class ReversalSetup:
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


def regime_ok(trend_4h: str, side: str, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "flat_only":
        return trend_4h == "FLAT"
    if mode == "countertrend_only":
        if side == "Sell":
            return trend_4h == "UP"
        return trend_4h == "DOWN"
    return True


def avg_volume(klines_15m: List[List[Any]], start_idx: int, end_idx: int) -> float:
    volumes = [float(klines_15m[idx][5]) for idx in range(start_idx, end_idx)]
    return sum(volumes) / len(volumes) if volumes else 0.0


def build_setup(
    klines_15m: List[List[Any]],
    idx_15m: int,
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
    params: ReversalParams,
) -> Optional[ReversalSetup]:
    if idx_15m < params.impulse_lookback_15m + 2:
        return None
    if idx_1h < max(params.ema_slow_period_1h - 1, 14):
        return None
    if idx_4h < 29:
        return None

    impulse_candle = klines_15m[idx_15m - 1]
    reversal_candle = klines_15m[idx_15m]
    history = klines_15m[idx_15m - params.impulse_lookback_15m - 1 : idx_15m - 1]
    if len(history) < params.impulse_lookback_15m:
        return None

    recent_1h_closes = [candle_close(candle) for candle in klines_1h[idx_1h - params.ema_slow_period_1h + 1 : idx_1h + 1]]
    ema_fast = ema(recent_1h_closes, params.ema_fast_period_1h)
    ema_slow = ema(recent_1h_closes, params.ema_slow_period_1h)
    current_1h_close = candle_close(klines_1h[idx_1h])
    atr_1h = calc_atr_from_klines(klines_1h[idx_1h - 14 : idx_1h + 1], period=14)
    if atr_1h <= 0:
        return None

    trend_4h = detect_trend_4h(klines_4h[idx_4h - 29 : idx_4h + 1])
    hot_vol = hot_vol_ratio_4h(klines_4h, idx_4h)
    if params.min_hot_vol_ratio_4h > 0:
        if hot_vol is None or hot_vol < params.min_hot_vol_ratio_4h:
            return None

    avg_range = sum(candle_range_value(candle) for candle in history) / len(history)
    impulse_range = candle_range_value(impulse_candle)
    if avg_range <= 0 or impulse_range < avg_range * params.min_range_expansion_15m:
        return None

    volume_mean = avg_volume(klines_15m, idx_15m - params.impulse_lookback_15m - 1, idx_15m - 1)
    impulse_volume = float(impulse_candle[5])
    if volume_mean <= 0 or impulse_volume < volume_mean * params.min_volume_ratio_15m:
        return None

    extension_up = (current_1h_close - ema_fast) / atr_1h if atr_1h > 0 else 0.0
    extension_down = (ema_fast - current_1h_close) / atr_1h if atr_1h > 0 else 0.0
    prior_high = max(candle_high(candle) for candle in history)
    prior_low = min(candle_low(candle) for candle in history)
    midpoint = candle_low(impulse_candle) + impulse_range * (1 - params.reversal_close_fraction)

    if (
        bullish_close(impulse_candle)
        and candle_high(impulse_candle) > prior_high
        and bearish_close(reversal_candle)
        and candle_close(reversal_candle) <= midpoint
        and extension_up >= params.min_extension_atr_1h
        and ema_fast >= ema_slow
        and regime_ok(trend_4h, "Sell", params.regime_mode)
    ):
        entry = candle_close(reversal_candle)
        stop = max(candle_high(impulse_candle), candle_high(reversal_candle)) + atr_1h * params.stop_buffer_atr_1h
        risk = stop - entry
        if risk <= 0 or risk > atr_1h * params.max_stop_multiplier_atr_1h:
            return None
        tp = entry - risk * params.rr_target
        return ReversalSetup(
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
                "impulse_range_ratio_15m": round(impulse_range / avg_range, 6),
                "impulse_volume_ratio_15m": round(impulse_volume / volume_mean, 6),
                "extension_atr_1h": round(extension_up, 6),
                "hot_vol_ratio_4h": round(hot_vol, 6) if hot_vol is not None else None,
                "signal": "up_thrust_reversal",
            },
        )

    midpoint = candle_high(impulse_candle) - impulse_range * (1 - params.reversal_close_fraction)
    if (
        bearish_close(impulse_candle)
        and candle_low(impulse_candle) < prior_low
        and bullish_close(reversal_candle)
        and candle_close(reversal_candle) >= midpoint
        and extension_down >= params.min_extension_atr_1h
        and ema_fast <= ema_slow
        and regime_ok(trend_4h, "Buy", params.regime_mode)
    ):
        entry = candle_close(reversal_candle)
        stop = min(candle_low(impulse_candle), candle_low(reversal_candle)) - atr_1h * params.stop_buffer_atr_1h
        risk = entry - stop
        if risk <= 0 or risk > atr_1h * params.max_stop_multiplier_atr_1h:
            return None
        tp = entry + risk * params.rr_target
        return ReversalSetup(
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
                "impulse_range_ratio_15m": round(impulse_range / avg_range, 6),
                "impulse_volume_ratio_15m": round(impulse_volume / volume_mean, 6),
                "extension_atr_1h": round(extension_down, 6),
                "hot_vol_ratio_4h": round(hot_vol, 6) if hot_vol is not None else None,
                "signal": "down_flush_reversal",
            },
        )

    return None


def run_reversal_backtest(
    symbol: str,
    klines_15m: List[List[Any]],
    klines_1h: List[List[Any]],
    klines_4h: List[List[Any]],
    initial_balance: float,
    params: ReversalParams,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    balance = initial_balance
    rows: List[Dict[str, Any]] = []
    times_1h = [int(row[0]) for row in klines_1h]
    times_4h = [int(row[0]) for row in klines_4h]
    idx_15m = max(params.impulse_lookback_15m + 2, 80)

    while idx_15m < len(klines_15m) - 1:
        current_time_ms = int(klines_15m[idx_15m][0])
        idx_1h = latest_idx(times_1h, current_time_ms)
        idx_4h = latest_idx(times_4h, current_time_ms)
        if idx_1h < 0 or idx_4h < 0:
            idx_15m += 1
            continue

        setup = build_setup(
            klines_15m=klines_15m,
            idx_15m=idx_15m,
            klines_1h=klines_1h,
            idx_1h=idx_1h,
            klines_4h=klines_4h,
            idx_4h=idx_4h,
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
            "strategy": "intraday_15m_reversal",
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
                "impulse_lookback_15m": params.impulse_lookback_15m,
                "min_range_expansion_15m": params.min_range_expansion_15m,
                "min_volume_ratio_15m": params.min_volume_ratio_15m,
                "min_extension_atr_1h": params.min_extension_atr_1h,
                "reversal_close_fraction": params.reversal_close_fraction,
                "rr_target": params.rr_target,
                "stop_buffer_atr_1h": params.stop_buffer_atr_1h,
                "max_stop_multiplier_atr_1h": params.max_stop_multiplier_atr_1h,
                "ema_fast_period_1h": params.ema_fast_period_1h,
                "ema_slow_period_1h": params.ema_slow_period_1h,
                "min_hot_vol_ratio_4h": params.min_hot_vol_ratio_4h,
                "regime_mode": params.regime_mode,
            },
            **setup.meta,
        }

        rows.append(
            {
                "strategy": "intraday_15m_reversal",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_intraday_reversal"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--impulse-lookbacks", default="12,20")
    parser.add_argument("--range-expansions", default="2.0,2.5")
    parser.add_argument("--volume-ratios", default="1.5,2.0")
    parser.add_argument("--extension-atr-fractions", default="0.5,0.8")
    parser.add_argument("--reversal-close-fractions", default="0.5,0.65")
    parser.add_argument("--rr-targets", default="1.5,2.0")
    parser.add_argument("--stop-buffers", default="0.05,0.10")
    parser.add_argument("--ema-pairs", default="10:30")
    parser.add_argument("--hot-vol-ratios", default="0.0,0.8")
    parser.add_argument("--regime-modes", default="all,flat_only,countertrend_only")
    parser.add_argument("--top-n", default="1,2,3")
    parser.add_argument("--min-train-trades", type=int, default=10)
    parser.add_argument("--min-test-trades", type=int, default=5)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.1)
    parser.add_argument("--top-results", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_symbols = normalize_symbols(args.symbols.split(","))
    loaded_data = load_symbol_data(args.ohlcv_dir, selected_symbols)
    if not loaded_data:
        raise ValueError("Не удалось загрузить OHLCV для reversal-исследования")

    global_start, global_end = find_global_period(loaded_data)
    split_dt = global_start + (global_end - global_start) * args.split_ratio
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    impulse_lookbacks = [int(v) for v in args.impulse_lookbacks.split(",") if v.strip()]
    range_expansions = [float(v) for v in args.range_expansions.split(",") if v.strip()]
    volume_ratios = [float(v) for v in args.volume_ratios.split(",") if v.strip()]
    extension_atr_fractions = [float(v) for v in args.extension_atr_fractions.split(",") if v.strip()]
    reversal_close_fractions = [float(v) for v in args.reversal_close_fractions.split(",") if v.strip()]
    rr_targets = [float(v) for v in args.rr_targets.split(",") if v.strip()]
    stop_buffers = [float(v) for v in args.stop_buffers.split(",") if v.strip()]
    ema_pairs: List[Tuple[int, int]] = []
    for part in args.ema_pairs.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        fast_raw, slow_raw = item.split(":", 1)
        fast = int(fast_raw)
        slow = int(slow_raw)
        if fast >= slow:
            continue
        ema_pairs.append((fast, slow))
    hot_vol_ratios = [float(v) for v in args.hot_vol_ratios.split(",") if v.strip()]
    regime_modes = [v.strip() for v in args.regime_modes.split(",") if v.strip()]
    top_n_values = [int(v) for v in args.top_n.split(",") if v.strip()]

    grid = list(
        itertools.product(
            impulse_lookbacks,
            range_expansions,
            volume_ratios,
            extension_atr_fractions,
            reversal_close_fractions,
            rr_targets,
            stop_buffers,
            ema_pairs,
            hot_vol_ratios,
            regime_modes,
        )
    )

    search_results: List[Dict[str, Any]] = []
    for (
        impulse_lookback,
        range_expansion,
        volume_ratio,
        extension_atr,
        reversal_close_fraction,
        rr_target,
        stop_buffer,
        ema_pair,
        hot_vol_ratio,
        regime_mode,
    ) in grid:
        ema_fast, ema_slow = ema_pair
        params = ReversalParams(
            impulse_lookback_15m=impulse_lookback,
            min_range_expansion_15m=range_expansion,
            min_volume_ratio_15m=volume_ratio,
            min_extension_atr_1h=extension_atr,
            reversal_close_fraction=reversal_close_fraction,
            rr_target=rr_target,
            stop_buffer_atr_1h=stop_buffer,
            ema_fast_period_1h=ema_fast,
            ema_slow_period_1h=ema_slow,
            min_hot_vol_ratio_4h=hot_vol_ratio,
            regime_mode=regime_mode,
        )

        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, payload in loaded_data.items():
            rows, _ = run_reversal_backtest(
                symbol=symbol,
                klines_15m=payload["15m"],
                klines_1h=payload["1h"],
                klines_4h=payload["4h"],
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

            search_results.append(
                {
                    "params": {
                        "impulse_lookback_15m": impulse_lookback,
                        "min_range_expansion_15m": range_expansion,
                        "min_volume_ratio_15m": volume_ratio,
                        "min_extension_atr_1h": extension_atr,
                        "reversal_close_fraction": reversal_close_fraction,
                        "rr_target": rr_target,
                        "stop_buffer_atr_1h": stop_buffer,
                        "ema_fast_period_1h": ema_fast,
                        "ema_slow_period_1h": ema_slow,
                        "min_hot_vol_ratio_4h": hot_vol_ratio,
                        "regime_mode": regime_mode,
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
                    "train": summarize_portfolio(train_symbol_metrics, args.initial_balance, train_days),
                    "test": summarize_portfolio(test_symbol_metrics, args.initial_balance, test_days),
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

    summary_path = os.path.join(args.output_dir, "intraday_reversal_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
