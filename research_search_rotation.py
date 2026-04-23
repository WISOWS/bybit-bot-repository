import argparse
import bisect
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from main import BASE_DIR, SYMBOLS
from research_backtest import normalize_symbols, run_strategy_backtest_klines, summarize_rows
from research_search import (
    annualized_return_pct,
    find_global_period,
    load_symbol_data,
    split_rows_by_time,
    summarize_portfolio,
)
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy


def parse_bool_list(raw: str) -> List[bool]:
    return [value.strip().lower() == "true" for value in raw.split(",") if value.strip()]


def load_regime_baseline_rows(
    loaded_data: Dict[str, Dict[str, List[List[Any]]]],
    initial_balance: float,
    params: RegimeSwitchParams,
) -> Dict[str, List[Dict[str, Any]]]:
    strategy_fn = make_regime_switch_strategy(params)
    rows_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, payload in loaded_data.items():
        rows, _ = run_strategy_backtest_klines(
            strategy_name="regime_switch_hybrid_rotation",
            strategy_fn=strategy_fn,
            symbol=symbol,
            klines_1h=payload["1h"],
            klines_4h=payload["4h"],
            initial_balance=initial_balance,
        )
        parsed_rows: List[Dict[str, Any]] = []
        for row in rows:
            note = json.loads(row["note"])
            parsed = dict(row)
            parsed["_timestamp_ms"] = int(note["entry_time_ms"])
            parsed["_sub_strategy"] = str(note.get("sub_strategy", ""))
            parsed["_regime"] = str(note.get("regime", note.get("trend", "")))
            parsed_rows.append(parsed)
        rows_by_symbol[symbol] = parsed_rows
    return rows_by_symbol


def build_symbol_closes(
    loaded_data: Dict[str, Dict[str, List[List[Any]]]],
) -> Dict[str, Tuple[List[int], List[float]]]:
    closes: Dict[str, Tuple[List[int], List[float]]] = {}
    for symbol, payload in loaded_data.items():
        klines_4h = payload["4h"]
        closes[symbol] = (
            [int(row[0]) for row in klines_4h],
            [float(row[4]) for row in klines_4h],
        )
    return closes


def strength_score_at(
    times: List[int],
    closes: List[float],
    timestamp_ms: int,
    lookback_4h: int,
) -> Optional[float]:
    idx = bisect.bisect_right(times, timestamp_ms) - 1
    if idx < lookback_4h:
        return None
    past_close = closes[idx - lookback_4h]
    current_close = closes[idx]
    if past_close <= 0:
        return None
    return current_close / past_close - 1.0


def build_rotation_cache(
    symbol_closes: Dict[str, Tuple[List[int], List[float]]],
    timestamps_ms: List[int],
    lookback_4h: int,
) -> Dict[int, Dict[str, Any]]:
    cache: Dict[int, Dict[str, Any]] = {}
    for timestamp_ms in timestamps_ms:
        scores: List[Tuple[str, float]] = []
        for symbol, (times, closes) in symbol_closes.items():
            score = strength_score_at(times, closes, timestamp_ms, lookback_4h)
            if score is None:
                continue
            scores.append((symbol, score))

        long_sorted = sorted(scores, key=lambda item: item[1], reverse=True)
        short_sorted = sorted(scores, key=lambda item: item[1])
        cache[timestamp_ms] = {
            "scores": {symbol: score for symbol, score in scores},
            "long_rank": {symbol: idx + 1 for idx, (symbol, _) in enumerate(long_sorted)},
            "short_rank": {symbol: idx + 1 for idx, (symbol, _) in enumerate(short_sorted)},
        }
    return cache


def filter_rows_by_rotation(
    rows_by_symbol: Dict[str, List[Dict[str, Any]]],
    rotation_cache: Dict[int, Dict[str, Any]],
    top_n_rotation: int,
    min_abs_strength: float,
    filter_range_trades: bool,
    max_range_strength: float,
) -> Dict[str, List[Dict[str, Any]]]:
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, rows in rows_by_symbol.items():
        kept_rows: List[Dict[str, Any]] = []
        for row in rows:
            timestamp_ms = int(row["_timestamp_ms"])
            rotation = rotation_cache.get(timestamp_ms)
            if not rotation:
                continue

            score = rotation["scores"].get(symbol)
            if score is None:
                continue

            side = str(row["side"])
            sub_strategy = str(row["_sub_strategy"])

            if sub_strategy == "simple_trend_pullback":
                if side == "Buy":
                    rank = rotation["long_rank"].get(symbol, 999999)
                    if rank > top_n_rotation or score < min_abs_strength:
                        continue
                else:
                    rank = rotation["short_rank"].get(symbol, 999999)
                    if rank > top_n_rotation or score > -min_abs_strength:
                        continue
            elif filter_range_trades and abs(score) > max_range_strength:
                continue

            kept_rows.append(row)
        filtered[symbol] = kept_rows
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_rotation"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--strength-lookbacks", default="6,10,14")
    parser.add_argument("--rotation-top-n", default="1,2,3")
    parser.add_argument("--min-abs-strengths", default="0.00,0.02")
    parser.add_argument("--filter-range-trades", default="false,true")
    parser.add_argument("--max-range-strengths", default="0.03,0.06")
    parser.add_argument("--pool-sizes", default="1,2,3,5")
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--min-test-trades", type=int, default=6)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.1)
    parser.add_argument("--top-results", type=int, default=20)
    parser.add_argument("--trend-threshold", type=float, default=0.7)
    parser.add_argument("--trend-distance", type=float, default=0.5)
    parser.add_argument("--trend-rr-target", type=float, default=3.2)
    parser.add_argument("--trend-stop-buffer", type=float, default=0.25)
    parser.add_argument("--range-zone-fraction", type=float, default=0.12)
    parser.add_argument("--range-target-fraction", type=float, default=0.60)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_symbols = normalize_symbols(SYMBOLS) if args.symbols == "all" else normalize_symbols(args.symbols.split(","))
    loaded_data = load_symbol_data(args.ohlcv_dir, selected_symbols)
    if not loaded_data:
        raise ValueError("Не удалось загрузить OHLCV для rotation-исследования")

    global_start, global_end = find_global_period(loaded_data)
    split_dt = global_start + (global_end - global_start) * args.split_ratio
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    baseline_params = RegimeSwitchParams(
        trend_threshold=args.trend_threshold,
        trend_min_distance_score=args.trend_distance,
        trend_rr_target=args.trend_rr_target,
        trend_stop_buffer_atr=args.trend_stop_buffer,
        range_zone_fraction=args.range_zone_fraction,
        range_target_fraction=args.range_target_fraction,
    )
    rows_by_symbol = load_regime_baseline_rows(loaded_data, args.initial_balance, baseline_params)
    symbol_closes = build_symbol_closes(loaded_data)
    all_timestamps = sorted(
        {
            int(row["_timestamp_ms"])
            for rows in rows_by_symbol.values()
            for row in rows
        }
    )

    strength_lookbacks = [int(value) for value in args.strength_lookbacks.split(",") if value.strip()]
    rotation_top_n_values = [int(value) for value in args.rotation_top_n.split(",") if value.strip()]
    min_abs_strengths = [float(value) for value in args.min_abs_strengths.split(",") if value.strip()]
    filter_range_values = parse_bool_list(args.filter_range_trades)
    max_range_strengths = [float(value) for value in args.max_range_strengths.split(",") if value.strip()]
    pool_sizes = [int(value) for value in args.pool_sizes.split(",") if value.strip()]

    search_results: List[Dict[str, Any]] = []

    for strength_lookback in strength_lookbacks:
        rotation_cache = build_rotation_cache(symbol_closes, all_timestamps, strength_lookback)

        for top_n_rotation in rotation_top_n_values:
            for min_abs_strength in min_abs_strengths:
                for filter_range_trades in filter_range_values:
                    range_strength_values = max_range_strengths if filter_range_trades else [999.0]

                    for max_range_strength in range_strength_values:
                        filtered_rows = filter_rows_by_rotation(
                            rows_by_symbol=rows_by_symbol,
                            rotation_cache=rotation_cache,
                            top_n_rotation=top_n_rotation,
                            min_abs_strength=min_abs_strength,
                            filter_range_trades=filter_range_trades,
                            max_range_strength=max_range_strength,
                        )

                        per_symbol: Dict[str, Dict[str, Any]] = {}
                        eligible_symbols: List[Tuple[float, float, str]] = []

                        for symbol, rows in filtered_rows.items():
                            train_rows, test_rows = split_rows_by_time(rows, split_dt)
                            train_metrics = summarize_rows(train_rows, args.initial_balance)
                            test_metrics = summarize_rows(test_rows, args.initial_balance)
                            per_symbol[symbol] = {
                                "train": train_metrics,
                                "test": test_metrics,
                            }

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

                        for pool_size in pool_sizes:
                            picked_symbols = [symbol for _, _, symbol in eligible_symbols[:pool_size]]
                            if len(picked_symbols) < pool_size:
                                continue

                            train_symbol_metrics = [per_symbol[symbol]["train"] for symbol in picked_symbols]
                            test_symbol_metrics = [
                                per_symbol[symbol]["test"]
                                for symbol in picked_symbols
                                if per_symbol[symbol]["test"]["trades"] >= args.min_test_trades
                            ]
                            if len(test_symbol_metrics) < pool_size:
                                continue

                            train_portfolio = summarize_portfolio(train_symbol_metrics, args.initial_balance, train_days)
                            test_portfolio = summarize_portfolio(test_symbol_metrics, args.initial_balance, test_days)
                            search_results.append(
                                {
                                    "baseline": {
                                        "trend_threshold": args.trend_threshold,
                                        "trend_distance": args.trend_distance,
                                        "trend_rr_target": args.trend_rr_target,
                                        "trend_stop_buffer": args.trend_stop_buffer,
                                        "range_zone_fraction": args.range_zone_fraction,
                                        "range_target_fraction": args.range_target_fraction,
                                    },
                                    "params": {
                                        "strength_lookback_4h": strength_lookback,
                                        "rotation_top_n": top_n_rotation,
                                        "min_abs_strength": min_abs_strength,
                                        "filter_range_trades": filter_range_trades,
                                        "max_range_strength": max_range_strength if filter_range_trades else None,
                                        "pool_size": pool_size,
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
            "symbols_considered": len(loaded_data),
            "results_count": len(search_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": search_results[: args.top_results],
    }

    summary_path = os.path.join(args.output_dir, "rotation_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
