import argparse
import bisect
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from research_search_perp_structure import asof_row, load_series_csv, oi_ratio_at, timestamps_of
from research_strategies import candle_close, candle_high, candle_low, ema


DEFAULT_SYMBOLS = "NEARUSDT,SOLUSDT,ENAUSDT,LINKUSDT"


@dataclass(frozen=True)
class PremiumFadeParams:
    premium_threshold: float = 0.001
    buy_ratio_extreme: float = 0.60
    funding_threshold: float = 0.0
    extension_atr_1h: float = 0.5
    oi_threshold: float = 1.0
    oi_mode: str = "expanding"
    ema_fast_period_1h: int = 10
    ema_slow_period_1h: int = 30
    min_rr_to_mean: float = 1.0
    stop_buffer_atr_1h: float = 0.05
    max_stop_multiplier_atr_1h: float = 1.5
    regime_mode: str = "all"


@dataclass
class PremiumFadeSetup:
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
        file_1h = os.path.join(ohlcv_dir, f"{symbol}_60.csv")
        file_4h = os.path.join(ohlcv_dir, f"{symbol}_240.csv")
        if not os.path.exists(file_1h) or not os.path.exists(file_4h):
            continue
        klines_1h = load_ohlcv_csv(file_1h)
        klines_4h = load_ohlcv_csv(file_4h)
        if len(klines_1h) < 200 or len(klines_4h) < 50:
            continue
        loaded[symbol] = {"1h": klines_1h, "4h": klines_4h}
    return loaded


def find_global_period(loaded_data: Dict[str, Dict[str, List[List[Any]]]]) -> Tuple[datetime, datetime]:
    starts: List[int] = []
    ends: List[int] = []
    for payload in loaded_data.values():
        klines_1h = payload["1h"]
        starts.append(int(klines_1h[0][0]))
        ends.append(int(klines_1h[-1][0]))
    return (
        datetime.fromtimestamp(min(starts) / 1000, tz=timezone.utc),
        datetime.fromtimestamp(max(ends) / 1000, tz=timezone.utc),
    )


def latest_idx(times: List[int], timestamp_ms: int) -> int:
    return bisect.bisect_right(times, timestamp_ms) - 1


def regime_ok(trend_4h: str, side: str, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "flat_only":
        return trend_4h == "FLAT"
    if mode == "countertrend_only":
        if side == "Sell":
            return trend_4h == "UP"
        return trend_4h == "DOWN"
    if mode == "trend_only":
        if side == "Sell":
            return trend_4h == "DOWN"
        return trend_4h == "UP"
    return True


def oi_ok(
    oi_rows: List[Dict[str, Any]],
    oi_ts: List[int],
    timestamp_ms: int,
    threshold: float,
    mode: str,
) -> bool:
    if mode == "off":
        return True
    ratio = oi_ratio_at(oi_rows, oi_ts, timestamp_ms, 6)
    if ratio is None:
        return False
    if mode == "expanding":
        return ratio >= threshold
    if mode == "subdued":
        return ratio <= threshold
    return True


def build_setup(
    klines_1h: List[List[Any]],
    idx_1h: int,
    klines_4h: List[List[Any]],
    idx_4h: int,
    funding_rows: List[Dict[str, Any]],
    funding_ts: List[int],
    account_rows: List[Dict[str, Any]],
    account_ts: List[int],
    premium_rows: List[Dict[str, Any]],
    premium_ts: List[int],
    oi_rows: List[Dict[str, Any]],
    oi_ts: List[int],
    params: PremiumFadeParams,
) -> Optional[PremiumFadeSetup]:
    min_idx = max(params.ema_slow_period_1h - 1, 14, 6)
    if idx_1h < min_idx or idx_4h < 29:
        return None

    timestamp_ms = int(klines_1h[idx_1h][0])
    funding_row = asof_row(funding_rows, funding_ts, timestamp_ms)
    account_row = asof_row(account_rows, account_ts, timestamp_ms)
    premium_row = asof_row(premium_rows, premium_ts, timestamp_ms)
    if not funding_row or not account_row or not premium_row:
        return None

    funding_rate = float(funding_row["funding_rate"])
    buy_ratio = float(account_row["buy_ratio"])
    premium_close = float(premium_row["close"])

    closes_1h = [candle_close(candle) for candle in klines_1h[idx_1h - params.ema_slow_period_1h + 1 : idx_1h + 1]]
    ema_fast = ema(closes_1h, params.ema_fast_period_1h)
    ema_slow = ema(closes_1h, params.ema_slow_period_1h)
    entry = candle_close(klines_1h[idx_1h])
    atr_1h = calc_atr_from_klines(klines_1h[idx_1h - 14 : idx_1h + 1], period=14)
    if atr_1h <= 0:
        return None

    trend_4h = detect_trend_4h(klines_4h[idx_4h - 29 : idx_4h + 1])
    recent_high = max(candle_high(candle) for candle in klines_1h[idx_1h - 5 : idx_1h + 1])
    recent_low = min(candle_low(candle) for candle in klines_1h[idx_1h - 5 : idx_1h + 1])

    extension_up = (entry - ema_fast) / atr_1h
    extension_down = (ema_fast - entry) / atr_1h

    if (
        premium_close >= params.premium_threshold
        and buy_ratio >= params.buy_ratio_extreme
        and funding_rate >= params.funding_threshold
        and extension_up >= params.extension_atr_1h
        and entry > ema_fast
        and entry > ema_slow
        and regime_ok(trend_4h, "Sell", params.regime_mode)
        and oi_ok(oi_rows, oi_ts, timestamp_ms, params.oi_threshold, params.oi_mode)
    ):
        stop = recent_high + atr_1h * params.stop_buffer_atr_1h
        risk = stop - entry
        reward = entry - ema_fast
        if risk <= 0 or risk > atr_1h * params.max_stop_multiplier_atr_1h:
            return None
        if reward <= 0 or reward / risk < params.min_rr_to_mean:
            return None
        return PremiumFadeSetup(
            side="Sell",
            entry=entry,
            stop=stop,
            tp=ema_fast,
            atr_1h=atr_1h,
            planned_rr=reward / risk,
            meta={
                "trend_4h": trend_4h,
                "funding_rate": round(funding_rate, 8),
                "buy_ratio": round(buy_ratio, 6),
                "premium_close": round(premium_close, 8),
                "extension_atr_1h": round(extension_up, 6),
                "signal": "premium_positive_fade",
            },
        )

    if (
        premium_close <= -params.premium_threshold
        and buy_ratio <= 1 - params.buy_ratio_extreme
        and funding_rate <= -params.funding_threshold
        and extension_down >= params.extension_atr_1h
        and entry < ema_fast
        and entry < ema_slow
        and regime_ok(trend_4h, "Buy", params.regime_mode)
        and oi_ok(oi_rows, oi_ts, timestamp_ms, params.oi_threshold, params.oi_mode)
    ):
        stop = recent_low - atr_1h * params.stop_buffer_atr_1h
        risk = entry - stop
        reward = ema_fast - entry
        if risk <= 0 or risk > atr_1h * params.max_stop_multiplier_atr_1h:
            return None
        if reward <= 0 or reward / risk < params.min_rr_to_mean:
            return None
        return PremiumFadeSetup(
            side="Buy",
            entry=entry,
            stop=stop,
            tp=ema_fast,
            atr_1h=atr_1h,
            planned_rr=reward / risk,
            meta={
                "trend_4h": trend_4h,
                "funding_rate": round(funding_rate, 8),
                "buy_ratio": round(buy_ratio, 6),
                "premium_close": round(premium_close, 8),
                "extension_atr_1h": round(extension_down, 6),
                "signal": "premium_negative_fade",
            },
        )

    return None


def run_premium_fade_backtest(
    symbol: str,
    klines_1h: List[List[Any]],
    klines_4h: List[List[Any]],
    funding_rows: List[Dict[str, Any]],
    account_rows: List[Dict[str, Any]],
    premium_rows: List[Dict[str, Any]],
    oi_rows: List[Dict[str, Any]],
    initial_balance: float,
    params: PremiumFadeParams,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    balance = initial_balance
    rows: List[Dict[str, Any]] = []
    times_4h = [int(row[0]) for row in klines_4h]
    funding_ts = timestamps_of(funding_rows)
    account_ts = timestamps_of(account_rows)
    premium_ts = timestamps_of(premium_rows)
    oi_ts = timestamps_of(oi_rows)

    idx_1h = max(params.ema_slow_period_1h, 20)
    while idx_1h < len(klines_1h) - 1:
        current_time_ms = int(klines_1h[idx_1h][0])
        idx_4h = latest_idx(times_4h, current_time_ms)
        if idx_4h < 0:
            idx_1h += 1
            continue

        setup = build_setup(
            klines_1h=klines_1h,
            idx_1h=idx_1h,
            klines_4h=klines_4h,
            idx_4h=idx_4h,
            funding_rows=funding_rows,
            funding_ts=funding_ts,
            account_rows=account_rows,
            account_ts=account_ts,
            premium_rows=premium_rows,
            premium_ts=premium_ts,
            oi_rows=oi_rows,
            oi_ts=oi_ts,
            params=params,
        )
        if setup is None:
            idx_1h += 1
            continue

        risk_per_unit = abs(setup.entry - setup.stop)
        if risk_per_unit <= 0:
            idx_1h += 1
            continue

        qty_by_risk = calc_qty_by_risk(balance, RISK_PER_TRADE, setup.entry, setup.stop)
        qty_by_margin = calc_qty_by_margin(balance, LEVERAGE, setup.entry)
        qty = min(qty_by_risk, qty_by_margin)
        if qty <= 0:
            idx_1h += 1
            continue

        effective_entry = apply_adverse_slippage(setup.entry, setup.side, "entry")
        exit_price, exit_reason, exit_index = simulate_exit(klines_1h, idx_1h, setup.side, setup.stop, setup.tp)
        effective_exit = apply_adverse_slippage(exit_price, setup.side, "exit")
        pnl_per_unit = effective_exit - effective_entry if setup.side == "Buy" else effective_entry - effective_exit
        entry_fee = qty * effective_entry * BACKTEST_TAKER_FEE_RATE
        exit_fee = qty * effective_exit * BACKTEST_TAKER_FEE_RATE
        realized_pnl = qty * pnl_per_unit - entry_fee - exit_fee
        risk_usdt = qty * risk_per_unit
        balance += realized_pnl

        rows.append(
            {
                "strategy": "premium_fade",
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
                "note": json.dumps(
                    {
                        "strategy": "premium_fade",
                        "effective_entry": round(effective_entry, 8),
                        "effective_exit": round(effective_exit, 8),
                        "entry_fee": round(entry_fee, 8),
                        "exit_fee": round(exit_fee, 8),
                        "qty": round(qty, 8),
                        "entry_time_ms": current_time_ms,
                        "exit_time_ms": int(klines_1h[exit_index][0]),
                        "exit_reason": exit_reason,
                        "balance_after": round(balance, 8),
                        "params": {
                            "premium_threshold": params.premium_threshold,
                            "buy_ratio_extreme": params.buy_ratio_extreme,
                            "funding_threshold": params.funding_threshold,
                            "extension_atr_1h": params.extension_atr_1h,
                            "oi_threshold": params.oi_threshold,
                            "oi_mode": params.oi_mode,
                            "ema_fast_period_1h": params.ema_fast_period_1h,
                            "ema_slow_period_1h": params.ema_slow_period_1h,
                            "min_rr_to_mean": params.min_rr_to_mean,
                            "stop_buffer_atr_1h": params.stop_buffer_atr_1h,
                            "max_stop_multiplier_atr_1h": params.max_stop_multiplier_atr_1h,
                            "regime_mode": params.regime_mode,
                        },
                        **setup.meta,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

        idx_1h = exit_index + 1

    return rows, summarize_rows(rows, initial_balance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--feature-dir", default=os.path.join(BASE_DIR, "perp_features"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_premium_fade"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--premium-thresholds", default="0.0005,0.001,0.0015,0.002")
    parser.add_argument("--buy-ratio-extremes", default="0.55,0.58,0.60,0.62")
    parser.add_argument("--funding-thresholds", default="0.0,0.00005,0.0001")
    parser.add_argument("--extension-atr-fractions", default="0.25,0.5,0.75")
    parser.add_argument("--oi-thresholds", default="1.0,1.05")
    parser.add_argument("--oi-modes", default="off,expanding")
    parser.add_argument("--min-rr-to-mean", default="1.0,1.25,1.5")
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
        raise ValueError("Не удалось загрузить OHLCV для premium-fade")

    global_start, global_end = find_global_period(loaded_data)
    split_dt = global_start + (global_end - global_start) * args.split_ratio
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    premium_thresholds = [float(v) for v in args.premium_thresholds.split(",") if v.strip()]
    buy_ratio_extremes = [float(v) for v in args.buy_ratio_extremes.split(",") if v.strip()]
    funding_thresholds = [float(v) for v in args.funding_thresholds.split(",") if v.strip()]
    extension_atr_fractions = [float(v) for v in args.extension_atr_fractions.split(",") if v.strip()]
    oi_thresholds = [float(v) for v in args.oi_thresholds.split(",") if v.strip()]
    oi_modes = [v.strip() for v in args.oi_modes.split(",") if v.strip()]
    min_rr_values = [float(v) for v in args.min_rr_to_mean.split(",") if v.strip()]
    regime_modes = [v.strip() for v in args.regime_modes.split(",") if v.strip()]
    top_n_values = [int(v) for v in args.top_n.split(",") if v.strip()]

    grid = list(
        itertools.product(
            premium_thresholds,
            buy_ratio_extremes,
            funding_thresholds,
            extension_atr_fractions,
            oi_thresholds,
            oi_modes,
            min_rr_values,
            regime_modes,
        )
    )

    search_results: List[Dict[str, Any]] = []
    for (
        premium_threshold,
        buy_ratio_extreme,
        funding_threshold,
        extension_atr,
        oi_threshold,
        oi_mode,
        min_rr_to_mean,
        regime_mode,
    ) in grid:
        params = PremiumFadeParams(
            premium_threshold=premium_threshold,
            buy_ratio_extreme=buy_ratio_extreme,
            funding_threshold=funding_threshold,
            extension_atr_1h=extension_atr,
            oi_threshold=oi_threshold,
            oi_mode=oi_mode,
            min_rr_to_mean=min_rr_to_mean,
            regime_mode=regime_mode,
        )

        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, payload in loaded_data.items():
            funding_path = os.path.join(args.feature_dir, f"{symbol}_funding.csv")
            account_path = os.path.join(args.feature_dir, f"{symbol}_account_ratio_1h.csv")
            premium_path = os.path.join(args.feature_dir, f"{symbol}_premium_60.csv")
            oi_path = os.path.join(args.feature_dir, f"{symbol}_open_interest_1h.csv")
            if not all(os.path.exists(path) for path in [funding_path, account_path, premium_path, oi_path]):
                continue
            rows, _ = run_premium_fade_backtest(
                symbol=symbol,
                klines_1h=payload["1h"],
                klines_4h=payload["4h"],
                funding_rows=load_series_csv(funding_path),
                account_rows=load_series_csv(account_path),
                premium_rows=load_series_csv(premium_path),
                oi_rows=load_series_csv(oi_path),
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
                        "premium_threshold": premium_threshold,
                        "buy_ratio_extreme": buy_ratio_extreme,
                        "funding_threshold": funding_threshold,
                        "extension_atr_1h": extension_atr,
                        "oi_threshold": oi_threshold,
                        "oi_mode": oi_mode,
                        "min_rr_to_mean": min_rr_to_mean,
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

    summary_path = os.path.join(args.output_dir, "premium_fade_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
