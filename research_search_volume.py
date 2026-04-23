import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backtest import load_ohlcv_csv
from main import BASE_DIR, SYMBOLS
from research_backtest import normalize_symbols, run_strategy_backtest_klines, summarize_rows
from research_search import annualized_return_pct, find_global_period, split_rows_by_time, summarize_portfolio
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy


def parse_float_list(raw: str) -> List[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def parse_bool_list(raw: str) -> List[bool]:
    return [value.strip().lower() == "true" for value in raw.split(",") if value.strip()]


def build_volume_index(klines_1h: List[List[str]]) -> Dict[int, int]:
    return {int(row[0]): idx for idx, row in enumerate(klines_1h)}


def volume_ratio_at(
    klines_1h: List[List[str]],
    volume_index: Dict[int, int],
    timestamp_ms: int,
    lookback: int,
) -> Optional[float]:
    idx = volume_index.get(timestamp_ms)
    if idx is None or idx < lookback:
        return None
    current_volume = float(klines_1h[idx][5])
    recent = [float(klines_1h[j][5]) for j in range(idx - lookback, idx)]
    avg_volume = sum(recent) / len(recent) if recent else 0.0
    if avg_volume <= 0:
        return None
    return current_volume / avg_volume


def build_loaded_data(symbols: List[str], ohlcv_dir: str) -> Dict[str, Dict[str, List[List[str]]]]:
    loaded: Dict[str, Dict[str, List[List[str]]]] = {}
    for symbol in symbols:
        file_1h = os.path.join(ohlcv_dir, f"{symbol}_60.csv")
        file_4h = os.path.join(ohlcv_dir, f"{symbol}_240.csv")
        if not os.path.exists(file_1h) or not os.path.exists(file_4h):
            continue
        klines_1h = load_ohlcv_csv(file_1h)
        klines_4h = load_ohlcv_csv(file_4h)
        if len(klines_1h) < 100 or len(klines_4h) < 50:
            continue
        loaded[symbol] = {"1h": klines_1h, "4h": klines_4h}
    return loaded


def load_baseline_rows(
    loaded_data: Dict[str, Dict[str, List[List[str]]]],
    initial_balance: float,
    params: RegimeSwitchParams,
) -> Dict[str, List[Dict[str, Any]]]:
    strategy_fn = make_regime_switch_strategy(params)
    rows_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, payload in loaded_data.items():
        rows, _ = run_strategy_backtest_klines(
            strategy_name="volume_filter_regime_switch",
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
            parsed["_note"] = note
            parsed["_timestamp_ms"] = int(note["entry_time_ms"])
            parsed["_sub_strategy"] = str(note.get("sub_strategy", ""))
            parsed_rows.append(parsed)
        rows_by_symbol[symbol] = parsed_rows
    return rows_by_symbol


def filter_rows_by_volume(
    rows: List[Dict[str, Any]],
    klines_1h: List[List[str]],
    volume_index: Dict[int, int],
    lookback: int,
    trend_min_ratio: float,
    filter_range_trades: bool,
    range_max_ratio: float,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        ratio = volume_ratio_at(klines_1h, volume_index, int(row["_timestamp_ms"]), lookback)
        if ratio is None:
            continue

        sub_strategy = str(row["_sub_strategy"])
        if sub_strategy == "simple_trend_pullback":
            if ratio < trend_min_ratio:
                continue
        elif filter_range_trades and sub_strategy == "range_mean_reversion":
            if ratio > range_max_ratio:
                continue

        enriched = dict(row)
        enriched["_volume_ratio"] = ratio
        filtered.append(enriched)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="NEARUSDT")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_volume"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--volume-lookbacks", default="12,24,36")
    parser.add_argument("--trend-min-ratios", default="0.8,1.0,1.2,1.5")
    parser.add_argument("--filter-range-trades", default="false,true")
    parser.add_argument("--range-max-ratios", default="0.8,1.0,1.2")
    parser.add_argument("--pool-sizes", default="1,2,3")
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--min-test-trades", type=int, default=6)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.1)
    parser.add_argument("--top-results", type=int, default=20)
    parser.add_argument("--trend-threshold", type=float, default=0.65)
    parser.add_argument("--trend-distance", type=float, default=0.5)
    parser.add_argument("--trend-rr-target", type=float, default=3.5)
    parser.add_argument("--trend-stop-buffer", type=float, default=0.2)
    parser.add_argument("--range-zone-fraction", type=float, default=0.12)
    parser.add_argument("--range-target-fraction", type=float, default=0.6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_symbols = normalize_symbols(SYMBOLS) if args.symbols == "all" else normalize_symbols(args.symbols.split(","))
    loaded_data = build_loaded_data(selected_symbols, args.ohlcv_dir)
    if not loaded_data:
        raise ValueError("Не удалось загрузить OHLCV для volume-исследования")

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
    rows_by_symbol = load_baseline_rows(loaded_data, args.initial_balance, baseline_params)

    volume_lookbacks = parse_int_list(args.volume_lookbacks)
    trend_min_ratios = parse_float_list(args.trend_min_ratios)
    filter_range_values = parse_bool_list(args.filter_range_trades)
    range_max_ratios = parse_float_list(args.range_max_ratios)
    pool_sizes = parse_int_list(args.pool_sizes)

    search_results: List[Dict[str, Any]] = []
    volume_indexes = {
        symbol: build_volume_index(payload["1h"])
        for symbol, payload in loaded_data.items()
    }

    for volume_lookback in volume_lookbacks:
        for trend_min_ratio in trend_min_ratios:
            for filter_range_trades in filter_range_values:
                active_range_ratios = range_max_ratios if filter_range_trades else [999.0]
                for range_max_ratio in active_range_ratios:
                    per_symbol: Dict[str, Dict[str, Any]] = {}
                    eligible_symbols: List[tuple[float, float, str]] = []

                    for symbol, rows in rows_by_symbol.items():
                        filtered_rows = filter_rows_by_volume(
                            rows=rows,
                            klines_1h=loaded_data[symbol]["1h"],
                            volume_index=volume_indexes[symbol],
                            lookback=volume_lookback,
                            trend_min_ratio=trend_min_ratio,
                            filter_range_trades=filter_range_trades,
                            range_max_ratio=range_max_ratio,
                        )
                        train_rows, test_rows = split_rows_by_time(filtered_rows, split_dt)
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
                                    "volume_lookback": volume_lookback,
                                    "trend_min_ratio": trend_min_ratio,
                                    "filter_range_trades": filter_range_trades,
                                    "range_max_ratio": range_max_ratio if filter_range_trades else None,
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
            "symbols_considered": len(loaded_data),
            "results_count": len(search_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": search_results[: args.top_results],
    }

    summary_path = os.path.join(args.output_dir, "volume_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
