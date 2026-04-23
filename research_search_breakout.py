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
from research_strategies import BreakoutRetestParams, make_breakout_retest_strategy


def parse_ema_pairs(raw: str) -> List[tuple[int, int]]:
    pairs: List[tuple[int, int]] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        fast_str, slow_str = token.split(":")
        fast = int(fast_str)
        slow = int(slow_str)
        if fast <= 0 or slow <= 0 or fast >= slow:
            continue
        pairs.append((fast, slow))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_breakout"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--lookbacks", default="12,20,30")
    parser.add_argument("--rr-targets", default="2.8,3.2,4.0")
    parser.add_argument("--stop-buffers", default="0.10,0.15,0.20")
    parser.add_argument("--max-stop-multipliers", default="1.0,1.3")
    parser.add_argument("--retest-buffers", default="0.001,0.002,0.003")
    parser.add_argument("--stop-lookbacks", default="3,5,8")
    parser.add_argument("--ema-pairs", default="20:50,10:30")
    parser.add_argument("--directional-close", default="true")
    parser.add_argument("--top-n", default="1,2,3,5")
    parser.add_argument("--min-train-trades", type=int, default=8)
    parser.add_argument("--min-test-trades", type=int, default=4)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.1)
    parser.add_argument("--top-results", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_symbols = normalize_symbols(SYMBOLS) if args.symbols == "all" else normalize_symbols(args.symbols.split(","))
    loaded_data = load_symbol_data(args.ohlcv_dir, selected_symbols)
    if not loaded_data:
        raise ValueError("Не удалось загрузить OHLCV для breakout-исследования")

    global_start, global_end = find_global_period(loaded_data)
    total_seconds = (global_end - global_start).total_seconds()
    split_dt = global_start + (global_end - global_start) * args.split_ratio
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    lookbacks = [int(value) for value in args.lookbacks.split(",") if value.strip()]
    rr_targets = [float(value) for value in args.rr_targets.split(",") if value.strip()]
    stop_buffers = [float(value) for value in args.stop_buffers.split(",") if value.strip()]
    max_stop_multipliers = [float(value) for value in args.max_stop_multipliers.split(",") if value.strip()]
    retest_buffers = [float(value) for value in args.retest_buffers.split(",") if value.strip()]
    stop_lookbacks = [int(value) for value in args.stop_lookbacks.split(",") if value.strip()]
    ema_pairs = parse_ema_pairs(args.ema_pairs)
    directional_close_values = [value.strip().lower() == "true" for value in args.directional_close.split(",") if value.strip()]
    top_n_values = [int(value) for value in args.top_n.split(",") if value.strip()]

    grid = list(
        itertools.product(
            lookbacks,
            rr_targets,
            stop_buffers,
            max_stop_multipliers,
            retest_buffers,
            stop_lookbacks,
            ema_pairs,
            directional_close_values,
        )
    )

    search_results: List[Dict[str, Any]] = []

    for (
        lookback,
        rr_target,
        stop_buffer,
        max_stop_multiplier,
        retest_buffer,
        stop_lookback,
        ema_pair,
        require_directional_close,
    ) in grid:
        ema_fast, ema_slow = ema_pair
        params = BreakoutRetestParams(
            breakout_lookback_4h=lookback,
            rr_target=rr_target,
            stop_buffer_atr=stop_buffer,
            max_stop_multiplier=max_stop_multiplier,
            retest_buffer_pct=retest_buffer,
            stop_lookback_1h=stop_lookback,
            ema_fast_period=ema_fast,
            ema_slow_period=ema_slow,
            require_directional_close=require_directional_close,
        )
        strategy_fn = make_breakout_retest_strategy(params)

        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, payload in loaded_data.items():
            rows, _ = run_strategy_backtest_klines(
                strategy_name="breakout_retest_search",
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
                        "breakout_lookback_4h": lookback,
                        "rr_target": rr_target,
                        "stop_buffer_atr": stop_buffer,
                        "max_stop_multiplier": max_stop_multiplier,
                        "retest_buffer_pct": retest_buffer,
                        "stop_lookback_1h": stop_lookback,
                        "ema_fast_period": ema_fast,
                        "ema_slow_period": ema_slow,
                        "require_directional_close": require_directional_close,
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

    summary_path = os.path.join(args.output_dir, "breakout_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
