import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_regime"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--trend-thresholds", default="0.70,0.75")
    parser.add_argument("--trend-distances", default="0.40,0.50")
    parser.add_argument("--trend-rr-targets", default="3.0,3.2")
    parser.add_argument("--trend-stop-buffers", default="0.20,0.25")
    parser.add_argument("--range-zone-fractions", default="0.08,0.12")
    parser.add_argument("--range-target-fractions", default="0.50,0.60")
    parser.add_argument("--top-n", default="1,2,3,5")
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--min-test-trades", type=int, default=6)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.1)
    parser.add_argument("--top-results", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_symbols = normalize_symbols(SYMBOLS) if args.symbols == "all" else normalize_symbols(args.symbols.split(","))
    loaded_data = load_symbol_data(args.ohlcv_dir, selected_symbols)
    if not loaded_data:
        raise ValueError("Не удалось загрузить OHLCV для regime-исследования")

    global_start, global_end = find_global_period(loaded_data)
    split_dt = global_start + (global_end - global_start) * args.split_ratio
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    trend_thresholds = [float(value) for value in args.trend_thresholds.split(",") if value.strip()]
    trend_distances = [float(value) for value in args.trend_distances.split(",") if value.strip()]
    trend_rr_targets = [float(value) for value in args.trend_rr_targets.split(",") if value.strip()]
    trend_stop_buffers = [float(value) for value in args.trend_stop_buffers.split(",") if value.strip()]
    range_zone_fractions = [float(value) for value in args.range_zone_fractions.split(",") if value.strip()]
    range_target_fractions = [float(value) for value in args.range_target_fractions.split(",") if value.strip()]
    top_n_values = [int(value) for value in args.top_n.split(",") if value.strip()]

    grid = list(
        itertools.product(
            trend_thresholds,
            trend_distances,
            trend_rr_targets,
            trend_stop_buffers,
            range_zone_fractions,
            range_target_fractions,
        )
    )

    search_results: List[Dict[str, Any]] = []

    for (
        trend_threshold,
        trend_distance,
        trend_rr_target,
        trend_stop_buffer,
        range_zone_fraction,
        range_target_fraction,
    ) in grid:
        params = RegimeSwitchParams(
            trend_threshold=trend_threshold,
            trend_min_distance_score=trend_distance,
            trend_rr_target=trend_rr_target,
            trend_stop_buffer_atr=trend_stop_buffer,
            range_zone_fraction=range_zone_fraction,
            range_target_fraction=range_target_fraction,
        )
        strategy_fn = make_regime_switch_strategy(params)

        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, payload in loaded_data.items():
            rows, _ = run_strategy_backtest_klines(
                strategy_name="regime_switch_hybrid_search",
                strategy_fn=strategy_fn,
                symbol=symbol,
                klines_1h=payload["1h"],
                klines_4h=payload["4h"],
                initial_balance=args.initial_balance,
            )
            train_rows, test_rows = split_rows_by_time(rows, split_dt)
            train_metrics = summarize_rows(train_rows, args.initial_balance)
            test_metrics = summarize_rows(test_rows, args.initial_balance)
            per_symbol[symbol] = {
                "train": train_metrics,
                "test": test_metrics,
            }

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
                        "trend_threshold": trend_threshold,
                        "trend_min_distance_score": trend_distance,
                        "trend_rr_target": trend_rr_target,
                        "trend_stop_buffer_atr": trend_stop_buffer,
                        "range_zone_fraction": range_zone_fraction,
                        "range_target_fraction": range_target_fraction,
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

    summary_path = os.path.join(args.output_dir, "regime_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
