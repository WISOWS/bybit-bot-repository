import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from main import BASE_DIR, SYMBOLS
from research_backtest import normalize_symbols, run_strategy_backtest_klines, summarize_rows
from research_search import (
    annualized_return_pct,
    find_global_period,
    load_symbol_data,
    split_rows_by_time,
    summarize_portfolio,
)
from research_strategies import (
    BreakoutRetestParams,
    MomentumExpansionParams,
    RangeMeanReversionParams,
    RegimeSwitchParams,
    TrendPullbackParams,
    make_breakout_retest_strategy,
    make_momentum_expansion_strategy,
    make_range_mean_reversion_strategy,
    make_regime_switch_strategy,
    make_simple_trend_pullback_strategy,
)


def build_candidates() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    trend_candidates = [
        TrendPullbackParams(threshold=0.7, min_distance_score=0.5, rr_target=3.2, stop_buffer_atr=0.25),
        TrendPullbackParams(threshold=0.7, min_distance_score=0.4, rr_target=3.2, stop_buffer_atr=0.25),
        TrendPullbackParams(threshold=0.75, min_distance_score=0.5, rr_target=3.2, stop_buffer_atr=0.25),
    ]
    for idx, params in enumerate(trend_candidates, start=1):
        candidates.append(
            {
                "name": f"trend_pullback_{idx}",
                "family": "trend_pullback",
                "params": asdict(params),
                "strategy_name": "symbol_lab_trend_pullback",
                "strategy_fn": make_simple_trend_pullback_strategy(params),
            }
        )

    regime_candidates = [
        RegimeSwitchParams(
            trend_threshold=0.7,
            trend_min_distance_score=0.5,
            trend_rr_target=3.2,
            trend_stop_buffer_atr=0.25,
            range_zone_fraction=0.12,
            range_target_fraction=0.60,
        ),
        RegimeSwitchParams(
            trend_threshold=0.7,
            trend_min_distance_score=0.5,
            trend_rr_target=3.2,
            trend_stop_buffer_atr=0.25,
            range_zone_fraction=0.08,
            range_target_fraction=0.60,
        ),
        RegimeSwitchParams(
            trend_threshold=0.7,
            trend_min_distance_score=0.5,
            trend_rr_target=3.2,
            trend_stop_buffer_atr=0.25,
            range_zone_fraction=0.12,
            range_target_fraction=0.50,
        ),
        RegimeSwitchParams(
            trend_threshold=0.7,
            trend_min_distance_score=0.4,
            trend_rr_target=3.2,
            trend_stop_buffer_atr=0.25,
            range_zone_fraction=0.12,
            range_target_fraction=0.60,
        ),
    ]
    for idx, params in enumerate(regime_candidates, start=1):
        candidates.append(
            {
                "name": f"regime_switch_{idx}",
                "family": "regime_switch",
                "params": asdict(params),
                "strategy_name": "symbol_lab_regime_switch",
                "strategy_fn": make_regime_switch_strategy(params),
            }
        )

    range_candidates = [
        RangeMeanReversionParams(
            flat_lookback_4h=24,
            range_lookback_4h=20,
            zone_fraction=0.08,
            stop_buffer_atr=0.15,
            target_fraction=0.50,
            min_rr=1.3,
            max_stop_multiplier=1.0,
            require_directional_close=True,
        ),
        RangeMeanReversionParams(
            flat_lookback_4h=24,
            range_lookback_4h=20,
            zone_fraction=0.12,
            stop_buffer_atr=0.15,
            target_fraction=0.50,
            min_rr=1.3,
            max_stop_multiplier=1.0,
            require_directional_close=True,
        ),
    ]
    for idx, params in enumerate(range_candidates, start=1):
        candidates.append(
            {
                "name": f"range_mean_reversion_{idx}",
                "family": "range_mean_reversion",
                "params": asdict(params),
                "strategy_name": "symbol_lab_range_mean_reversion",
                "strategy_fn": make_range_mean_reversion_strategy(params),
            }
        )

    breakout_params = BreakoutRetestParams(
        breakout_lookback_4h=12,
        rr_target=2.8,
        stop_buffer_atr=0.15,
        max_stop_multiplier=1.3,
        retest_buffer_pct=0.002,
        stop_lookback_1h=5,
        ema_fast_period=20,
        ema_slow_period=50,
        require_directional_close=True,
    )
    candidates.append(
        {
            "name": "breakout_retest_best",
            "family": "breakout_retest",
            "params": asdict(breakout_params),
            "strategy_name": "symbol_lab_breakout_retest",
            "strategy_fn": make_breakout_retest_strategy(breakout_params),
        }
    )

    momentum_params = MomentumExpansionParams(
        breakout_lookback_1h=12,
        compression_lookback_4h=6,
        max_atr_ratio=1.05,
        rr_target=3.0,
        stop_buffer_atr=0.1,
        max_stop_multiplier=1.0,
        ema_fast_period=10,
        ema_slow_period=30,
        min_body_fraction=0.55,
        min_range_expansion=1.0,
        swing_lookback_1h=4,
    )
    candidates.append(
        {
            "name": "momentum_expansion_best",
            "family": "momentum_expansion",
            "params": asdict(momentum_params),
            "strategy_name": "symbol_lab_momentum_expansion",
            "strategy_fn": make_momentum_expansion_strategy(momentum_params),
        }
    )

    return candidates


def train_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float]:
    train = item["train"]
    return (
        float(train["annualized_return_pct"]),
        float(train["profit_factor"]),
        -float(train["max_drawdown_pct"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_symbol_lab"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--min-test-trades", type=int, default=6)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.1)
    parser.add_argument("--pool-sizes", default="1,2,3,5")
    parser.add_argument("--top-results", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_symbols = normalize_symbols(SYMBOLS) if args.symbols == "all" else normalize_symbols(args.symbols.split(","))
    loaded_data = load_symbol_data(args.ohlcv_dir, selected_symbols)
    if not loaded_data:
        raise ValueError("Не удалось загрузить OHLCV для symbol-lab")

    global_start, global_end = find_global_period(loaded_data)
    split_dt = global_start + (global_end - global_start) * args.split_ratio
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)
    pool_sizes = [int(value) for value in args.pool_sizes.split(",") if value.strip()]

    candidates = build_candidates()
    symbol_results: List[Dict[str, Any]] = []

    for symbol, payload in loaded_data.items():
        evaluated: List[Dict[str, Any]] = []
        for candidate in candidates:
            rows, _ = run_strategy_backtest_klines(
                strategy_name=candidate["strategy_name"],
                strategy_fn=candidate["strategy_fn"],
                symbol=symbol,
                klines_1h=payload["1h"],
                klines_4h=payload["4h"],
                initial_balance=args.initial_balance,
            )
            train_rows, test_rows = split_rows_by_time(rows, split_dt)
            train_metrics = summarize_rows(train_rows, args.initial_balance)
            test_metrics = summarize_rows(test_rows, args.initial_balance)
            train_annualized = annualized_return_pct(train_metrics["final_balance"], args.initial_balance, train_days)
            test_annualized = annualized_return_pct(test_metrics["final_balance"], args.initial_balance, test_days)

            evaluated.append(
                {
                    "name": candidate["name"],
                    "family": candidate["family"],
                    "params": candidate["params"],
                    "train": {
                        **train_metrics,
                        "annualized_return_pct": round(train_annualized, 4),
                    },
                    "test": {
                        **test_metrics,
                        "annualized_return_pct": round(test_annualized, 4),
                    },
                }
            )

        eligible = [
            item
            for item in evaluated
            if item["train"]["trades"] >= args.min_train_trades
            and item["train"]["profit_factor"] >= args.min_train_profit_factor
            and item["train"]["net_pnl"] > 0
        ]
        if not eligible:
            continue

        eligible.sort(key=train_sort_key, reverse=True)
        selected = eligible[0]
        if selected["test"]["trades"] < args.min_test_trades:
            continue

        symbol_results.append(
            {
                "symbol": symbol,
                "selected_model": selected["name"],
                "family": selected["family"],
                "params": selected["params"],
                "train": selected["train"],
                "test": selected["test"],
            }
        )

    symbol_results.sort(
        key=lambda item: (
            item["test"]["annualized_return_pct"],
            item["test"]["profit_factor"],
            -item["test"]["max_drawdown_pct"],
        ),
        reverse=True,
    )

    portfolio_results: List[Dict[str, Any]] = []
    train_sorted_symbols = sorted(
        symbol_results,
        key=lambda item: (
            item["train"]["annualized_return_pct"],
            item["train"]["profit_factor"],
            -item["train"]["max_drawdown_pct"],
        ),
        reverse=True,
    )

    for pool_size in pool_sizes:
        selected_pool = train_sorted_symbols[:pool_size]
        if len(selected_pool) < pool_size:
            continue
        train_metrics = [item["train"] for item in selected_pool]
        test_metrics = [item["test"] for item in selected_pool]
        portfolio_results.append(
            {
                "pool_size": pool_size,
                "symbols": [item["symbol"] for item in selected_pool],
                "models": [item["selected_model"] for item in selected_pool],
                "train": summarize_portfolio(train_metrics, args.initial_balance, train_days),
                "test": summarize_portfolio(test_metrics, args.initial_balance, test_days),
            }
        )

    output_payload = {
        "meta": {
            "candidates": [candidate["name"] for candidate in candidates],
            "symbols_considered": len(loaded_data),
            "selected_symbols": len(symbol_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "best_symbols": symbol_results[: args.top_results],
        "portfolio_results": portfolio_results,
        "split": {
            "start": global_start.isoformat(),
            "split": split_dt.isoformat(),
            "end": global_end.isoformat(),
            "train_days": round(train_days, 4),
            "test_days": round(test_days, 4),
        },
    }

    summary_path = os.path.join(args.output_dir, "symbol_lab.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"selected_symbols={len(symbol_results)}")
    for item in symbol_results[:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))
    for item in portfolio_results[:5]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
