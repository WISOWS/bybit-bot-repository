import argparse
import bisect
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from backtest import load_ohlcv_csv
from main import BASE_DIR, calc_atr_from_klines, detect_trend_4h
from research_backtest import run_strategy_backtest_klines, summarize_rows
from research_search import annualized_return_pct, find_global_period, split_rows_by_time
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy


DEFAULT_BASELINE_PARAMS = RegimeSwitchParams(
    trend_threshold=0.65,
    trend_min_distance_score=0.5,
    trend_rr_target=3.5,
    trend_stop_buffer_atr=0.2,
    range_zone_fraction=0.12,
    range_target_fraction=0.6,
)


def load_series_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return sorted(
        rows,
        key=lambda row: int(row.get("timestamp_ms") or row.get("start_time_ms") or 0),
    )


def timestamps_of(rows: List[Dict[str, Any]]) -> List[int]:
    return [int(row.get("timestamp_ms") or row.get("start_time_ms") or 0) for row in rows]


def asof_row(rows: List[Dict[str, Any]], timestamps: List[int], target_ms: int) -> Optional[Dict[str, Any]]:
    idx = bisect.bisect_right(timestamps, target_ms) - 1
    if idx < 0 or idx >= len(rows):
        return None
    return rows[idx]


def build_4h_index(klines_4h: List[List[Any]]) -> List[int]:
    return [int(row[0]) for row in klines_4h]


def latest_4h_idx(times_4h: List[int], target_ms: int) -> int:
    return bisect.bisect_right(times_4h, target_ms) - 1


def btc_allow_flat_ok(side: str, btc_4h: List[List[Any]], btc_times_4h: List[int], timestamp_ms: int) -> bool:
    idx_4h = latest_4h_idx(btc_times_4h, timestamp_ms)
    if idx_4h < 29:
        return False
    recent = btc_4h[idx_4h - 29 : idx_4h + 1]
    trend = detect_trend_4h(recent)
    if side == "Buy":
        return trend in {"UP", "FLAT"}
    return trend in {"DOWN", "FLAT"}


def hot_vol_ratio_at(klines_4h: List[List[Any]], times_4h: List[int], timestamp_ms: int) -> Optional[float]:
    idx_4h = latest_4h_idx(times_4h, timestamp_ms)
    if idx_4h < 14:
        return None
    short = calc_atr_from_klines(klines_4h[idx_4h - 6 : idx_4h + 1], period=6)
    long = calc_atr_from_klines(klines_4h[idx_4h - 14 : idx_4h + 1], period=14)
    if short <= 0 or long <= 0:
        return None
    return short / long


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def oi_ratio_at(
    oi_rows: List[Dict[str, Any]],
    oi_timestamps: List[int],
    timestamp_ms: int,
    lookback: int,
) -> Optional[float]:
    idx = bisect.bisect_right(oi_timestamps, timestamp_ms) - 1
    if idx < lookback or idx >= len(oi_rows):
        return None
    current_oi = float(oi_rows[idx]["open_interest"])
    recent = [float(oi_rows[j]["open_interest"]) for j in range(idx - lookback, idx)]
    avg_oi = mean(recent)
    if avg_oi <= 0:
        return None
    return current_oi / avg_oi


def load_baseline_rows(
    symbol: str,
    klines_1h: List[List[Any]],
    klines_4h: List[List[Any]],
    initial_balance: float,
    params: RegimeSwitchParams,
) -> List[Dict[str, Any]]:
    strategy_fn = make_regime_switch_strategy(params)
    rows, _ = run_strategy_backtest_klines(
        strategy_name="perp_structure_regime_switch",
        strategy_fn=strategy_fn,
        symbol=symbol,
        klines_1h=klines_1h,
        klines_4h=klines_4h,
        initial_balance=initial_balance,
    )
    parsed_rows: List[Dict[str, Any]] = []
    for row in rows:
        note = json.loads(row["note"])
        parsed = dict(row)
        parsed["_note"] = note
        parsed["_timestamp_ms"] = int(note["entry_time_ms"])
        parsed["_side"] = str(row["side"])
        parsed["_sub_strategy"] = str(note.get("sub_strategy", ""))
        parsed_rows.append(parsed)
    return parsed_rows


def build_enriched_rows(
    rows: List[Dict[str, Any]],
    symbol_4h: List[List[Any]],
    btc_4h: List[List[Any]],
    funding_rows: List[Dict[str, Any]],
    oi_rows: List[Dict[str, Any]],
    account_rows: List[Dict[str, Any]],
    premium_rows: List[Dict[str, Any]],
    min_hot_vol_ratio: float,
) -> List[Dict[str, Any]]:
    symbol_times_4h = build_4h_index(symbol_4h)
    btc_times_4h = build_4h_index(btc_4h)
    funding_ts = timestamps_of(funding_rows)
    oi_ts = timestamps_of(oi_rows)
    account_ts = timestamps_of(account_rows)
    premium_ts = timestamps_of(premium_rows)

    enriched_rows: List[Dict[str, Any]] = []
    for row in rows:
        timestamp_ms = int(row["_timestamp_ms"])
        side = str(row["_side"])

        if not btc_allow_flat_ok(side, btc_4h, btc_times_4h, timestamp_ms):
            continue

        hot_vol_ratio = hot_vol_ratio_at(symbol_4h, symbol_times_4h, timestamp_ms)
        if hot_vol_ratio is None or hot_vol_ratio < min_hot_vol_ratio:
            continue

        funding_row = asof_row(funding_rows, funding_ts, timestamp_ms)
        oi_row = asof_row(oi_rows, oi_ts, timestamp_ms)
        account_row = asof_row(account_rows, account_ts, timestamp_ms)
        premium_row = asof_row(premium_rows, premium_ts, timestamp_ms)
        if not funding_row or not oi_row or not account_row or not premium_row:
            continue

        item = dict(row)
        item["_funding_rate"] = float(funding_row["funding_rate"])
        item["_buy_ratio"] = float(account_row["buy_ratio"])
        item["_sell_ratio"] = float(account_row["sell_ratio"])
        item["_premium_close"] = float(premium_row["close"])
        item["_hot_vol_ratio"] = hot_vol_ratio
        enriched_rows.append(item)

    return enriched_rows


def passes_perp_filter(
    row: Dict[str, Any],
    oi_rows: List[Dict[str, Any]],
    oi_ts: List[int],
    long_max_funding: float,
    short_min_funding: float,
    long_max_buy_ratio: float,
    short_min_buy_ratio: float,
    long_max_premium: float,
    short_min_premium: float,
    oi_mode: str,
    oi_lookback: int,
    oi_threshold: float,
) -> bool:
    side = str(row["_side"])
    funding_rate = float(row["_funding_rate"])
    buy_ratio = float(row["_buy_ratio"])
    premium_close = float(row["_premium_close"])
    oi_ratio = oi_ratio_at(oi_rows, oi_ts, int(row["_timestamp_ms"]), oi_lookback)
    if oi_ratio is None:
        return False

    if oi_mode == "expanding" and oi_ratio < oi_threshold:
        return False
    if oi_mode == "subdued" and oi_ratio > oi_threshold:
        return False

    if side == "Buy":
        if funding_rate > long_max_funding:
            return False
        if buy_ratio > long_max_buy_ratio:
            return False
        if premium_close > long_max_premium:
            return False
    else:
        if funding_rate < short_min_funding:
            return False
        if buy_ratio < short_min_buy_ratio:
            return False
        if premium_close < short_min_premium:
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NEARUSDT")
    parser.add_argument("--btc-symbol", default="BTCUSDT")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--feature-dir", default=os.path.join(BASE_DIR, "perp_features"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_perp_structure"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--min-hot-vol-ratio", type=float, default=0.8)
    parser.add_argument("--trend-threshold", type=float, default=DEFAULT_BASELINE_PARAMS.trend_threshold)
    parser.add_argument("--trend-distance", type=float, default=DEFAULT_BASELINE_PARAMS.trend_min_distance_score)
    parser.add_argument("--trend-rr-target", type=float, default=DEFAULT_BASELINE_PARAMS.trend_rr_target)
    parser.add_argument("--trend-stop-buffer", type=float, default=DEFAULT_BASELINE_PARAMS.trend_stop_buffer_atr)
    parser.add_argument("--range-zone-fraction", type=float, default=DEFAULT_BASELINE_PARAMS.range_zone_fraction)
    parser.add_argument("--range-target-fraction", type=float, default=DEFAULT_BASELINE_PARAMS.range_target_fraction)
    parser.add_argument("--long-max-funding", default="1.0,0.0002,0.0001,0.0")
    parser.add_argument("--short-min-funding", default="-1.0,0.0,0.0001,0.0002")
    parser.add_argument("--long-max-buy-ratio", default="1.0,0.56,0.54,0.52")
    parser.add_argument("--short-min-buy-ratio", default="0.0,0.50,0.52,0.54")
    parser.add_argument("--long-max-premium", default="1.0,0.002,0.001,0.0005")
    parser.add_argument("--short-min-premium", default="-1.0,0.0,0.0005,0.001")
    parser.add_argument("--oi-lookbacks", default="6,12,24")
    parser.add_argument("--oi-modes", default="any,expanding,subdued")
    parser.add_argument("--oi-thresholds", default="0.95,1.0,1.05")
    parser.add_argument("--min-train-trades", type=int, default=8)
    parser.add_argument("--min-test-trades", type=int, default=5)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.15)
    parser.add_argument("--top-results", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    symbol_1h_path = os.path.join(args.ohlcv_dir, f"{args.symbol}_60.csv")
    symbol_4h_path = os.path.join(args.ohlcv_dir, f"{args.symbol}_240.csv")
    btc_4h_path = os.path.join(args.ohlcv_dir, f"{args.btc_symbol}_240.csv")
    funding_path = os.path.join(args.feature_dir, f"{args.symbol}_funding.csv")
    oi_path = os.path.join(args.feature_dir, f"{args.symbol}_open_interest_1h.csv")
    account_path = os.path.join(args.feature_dir, f"{args.symbol}_account_ratio_1h.csv")
    premium_path = os.path.join(args.feature_dir, f"{args.symbol}_premium_60.csv")

    required_paths = [
        symbol_1h_path,
        symbol_4h_path,
        btc_4h_path,
        funding_path,
        oi_path,
        account_path,
        premium_path,
    ]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Не хватает файлов для perp-search: {missing}")

    klines_1h = load_ohlcv_csv(symbol_1h_path)
    klines_4h = load_ohlcv_csv(symbol_4h_path)
    btc_4h = load_ohlcv_csv(btc_4h_path)
    funding_rows = load_series_csv(funding_path)
    oi_rows = load_series_csv(oi_path)
    account_rows = load_series_csv(account_path)
    premium_rows = load_series_csv(premium_path)

    baseline_params = RegimeSwitchParams(
        trend_threshold=args.trend_threshold,
        trend_min_distance_score=args.trend_distance,
        trend_rr_target=args.trend_rr_target,
        trend_stop_buffer_atr=args.trend_stop_buffer,
        range_zone_fraction=args.range_zone_fraction,
        range_target_fraction=args.range_target_fraction,
    )

    global_start, global_end = find_global_period(
        {
            args.symbol: {"1h": klines_1h, "4h": klines_4h},
            args.btc_symbol: {"1h": klines_1h, "4h": btc_4h},
        }
    )
    split_dt = global_start + timedelta(seconds=(global_end - global_start).total_seconds() * args.split_ratio)
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    baseline_rows = load_baseline_rows(
        args.symbol,
        klines_1h,
        klines_4h,
        args.initial_balance,
        baseline_params,
    )
    enriched_rows = build_enriched_rows(
        rows=baseline_rows,
        symbol_4h=klines_4h,
        btc_4h=btc_4h,
        funding_rows=funding_rows,
        oi_rows=oi_rows,
        account_rows=account_rows,
        premium_rows=premium_rows,
        min_hot_vol_ratio=args.min_hot_vol_ratio,
    )

    oi_timestamps = timestamps_of(oi_rows)
    long_max_funding_values = [float(v) for v in args.long_max_funding.split(",") if v.strip()]
    short_min_funding_values = [float(v) for v in args.short_min_funding.split(",") if v.strip()]
    long_max_buy_ratio_values = [float(v) for v in args.long_max_buy_ratio.split(",") if v.strip()]
    short_min_buy_ratio_values = [float(v) for v in args.short_min_buy_ratio.split(",") if v.strip()]
    long_max_premium_values = [float(v) for v in args.long_max_premium.split(",") if v.strip()]
    short_min_premium_values = [float(v) for v in args.short_min_premium.split(",") if v.strip()]
    oi_lookback_values = [int(v) for v in args.oi_lookbacks.split(",") if v.strip()]
    oi_modes = [v.strip() for v in args.oi_modes.split(",") if v.strip()]
    oi_threshold_values = [float(v) for v in args.oi_thresholds.split(",") if v.strip()]

    search_results: List[Dict[str, Any]] = []
    for long_max_funding in long_max_funding_values:
        for short_min_funding in short_min_funding_values:
            for long_max_buy_ratio in long_max_buy_ratio_values:
                for short_min_buy_ratio in short_min_buy_ratio_values:
                    for long_max_premium in long_max_premium_values:
                        for short_min_premium in short_min_premium_values:
                            for oi_lookback in oi_lookback_values:
                                for oi_mode in oi_modes:
                                    active_oi_thresholds = [999.0] if oi_mode == "any" else oi_threshold_values
                                    for oi_threshold in active_oi_thresholds:
                                        filtered_rows = [
                                            row
                                            for row in enriched_rows
                                            if passes_perp_filter(
                                                row=row,
                                                oi_rows=oi_rows,
                                                oi_ts=oi_timestamps,
                                                long_max_funding=long_max_funding,
                                                short_min_funding=short_min_funding,
                                                long_max_buy_ratio=long_max_buy_ratio,
                                                short_min_buy_ratio=short_min_buy_ratio,
                                                long_max_premium=long_max_premium,
                                                short_min_premium=short_min_premium,
                                                oi_mode=oi_mode,
                                                oi_lookback=oi_lookback,
                                                oi_threshold=oi_threshold,
                                            )
                                        ]
                                        train_rows, test_rows = split_rows_by_time(filtered_rows, split_dt)
                                        train_metrics = summarize_rows(train_rows, args.initial_balance)
                                        test_metrics = summarize_rows(test_rows, args.initial_balance)

                                        if train_metrics["trades"] < args.min_train_trades:
                                            continue
                                        if test_metrics["trades"] < args.min_test_trades:
                                            continue
                                        if train_metrics["profit_factor"] < args.min_train_profit_factor:
                                            continue
                                        if train_metrics["net_pnl"] <= 0:
                                            continue

                                        search_results.append(
                                            {
                                                "symbol": args.symbol,
                                                "baseline": {
                                                    "strategy": "regime_switch_hybrid",
                                                    "params": {
                                                        "trend_threshold": baseline_params.trend_threshold,
                                                        "trend_min_distance_score": baseline_params.trend_min_distance_score,
                                                        "trend_rr_target": baseline_params.trend_rr_target,
                                                        "trend_stop_buffer_atr": baseline_params.trend_stop_buffer_atr,
                                                        "range_zone_fraction": baseline_params.range_zone_fraction,
                                                        "range_target_fraction": baseline_params.range_target_fraction,
                                                    },
                                                    "btc_filter": "allow_flat",
                                                    "min_hot_vol_ratio": args.min_hot_vol_ratio,
                                                },
                                                "params": {
                                                    "long_max_funding": long_max_funding,
                                                    "short_min_funding": short_min_funding,
                                                    "long_max_buy_ratio": long_max_buy_ratio,
                                                    "short_min_buy_ratio": short_min_buy_ratio,
                                                    "long_max_premium": long_max_premium,
                                                    "short_min_premium": short_min_premium,
                                                    "oi_lookback": oi_lookback,
                                                    "oi_mode": oi_mode,
                                                    "oi_threshold": None if oi_mode == "any" else oi_threshold,
                                                },
                                                "split": {
                                                    "start": global_start.isoformat(),
                                                    "split": split_dt.isoformat(),
                                                    "end": global_end.isoformat(),
                                                    "train_days": round(train_days, 4),
                                                    "test_days": round(test_days, 4),
                                                },
                                                "train": {
                                                    **train_metrics,
                                                    "annualized_return_pct": round(
                                                        annualized_return_pct(
                                                            train_metrics["final_balance"],
                                                            args.initial_balance,
                                                            train_days,
                                                        ),
                                                        4,
                                                    ),
                                                },
                                                "test": {
                                                    **test_metrics,
                                                    "annualized_return_pct": round(
                                                        annualized_return_pct(
                                                            test_metrics["final_balance"],
                                                            args.initial_balance,
                                                            test_days,
                                                        ),
                                                        4,
                                                    ),
                                                },
                                            }
                                        )

    search_results.sort(
        key=lambda item: (
            item["test"]["annualized_return_pct"],
            item["test"]["profit_factor"],
            -item["test"]["max_drawdown_pct"],
        ),
        reverse=True,
    )

    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol": args.symbol,
            "baseline_rows": len(baseline_rows),
            "rows_after_existing_filters": len(enriched_rows),
            "results_count": len(search_results),
        },
        "results": search_results[: args.top_results],
    }

    summary_path = os.path.join(args.output_dir, "perp_structure_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"baseline_rows={len(baseline_rows)}")
    print(f"rows_after_existing_filters={len(enriched_rows)}")
    print(f"results={len(search_results)}")
    for item in output["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
