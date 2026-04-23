import argparse
import itertools
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from backtest import load_ohlcv_csv
from main import BASE_DIR, SYMBOLS
from research_backtest import normalize_symbols, run_strategy_backtest_klines, summarize_rows
from research_strategies import TrendPullbackParams, make_simple_trend_pullback_strategy


def annualized_return_pct(final_balance: float, initial_balance: float, days: float) -> float:
    if initial_balance <= 0 or final_balance <= 0 or days <= 0:
        return 0.0
    return ((final_balance / initial_balance) ** (365.0 / days) - 1.0) * 100.0


def total_return_pct(final_balance: float, initial_balance: float) -> float:
    if initial_balance <= 0:
        return 0.0
    return (final_balance / initial_balance - 1.0) * 100.0


def find_global_period(loaded_data: Dict[str, Dict[str, List[List[Any]]]]) -> Tuple[datetime, datetime]:
    starts: List[int] = []
    ends: List[int] = []
    for payload in loaded_data.values():
        klines_1h = payload["1h"]
        if not klines_1h:
            continue
        starts.append(int(klines_1h[0][0]))
        ends.append(int(klines_1h[-1][0]))
    if not starts or not ends:
        raise ValueError("Нет данных для поиска периода")
    start = datetime.fromtimestamp(min(starts) / 1000, tz=timezone.utc)
    end = datetime.fromtimestamp(max(ends) / 1000, tz=timezone.utc)
    return start, end


def split_rows_by_time(rows: List[Dict[str, Any]], split_dt: datetime) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_dt = datetime.fromisoformat(row["timestamp"])
        if row_dt <= split_dt:
            train_rows.append(row)
        else:
            test_rows.append(row)
    return train_rows, test_rows


def summarize_portfolio(symbol_metrics: List[Dict[str, Any]], initial_balance: float, days: float) -> Dict[str, Any]:
    if not symbol_metrics:
        return {
            "symbols": 0,
            "average_final_balance": initial_balance,
            "average_total_return_pct": 0.0,
            "average_annualized_return_pct": 0.0,
            "average_profit_factor": 0.0,
            "average_max_drawdown_pct": 0.0,
            "total_trades": 0,
            "trades_per_month": 0.0,
        }

    avg_final_balance = sum(item["final_balance"] for item in symbol_metrics) / len(symbol_metrics)
    return {
        "symbols": len(symbol_metrics),
        "average_final_balance": round(avg_final_balance, 4),
        "average_total_return_pct": round(total_return_pct(avg_final_balance, initial_balance), 4),
        "average_annualized_return_pct": round(
            annualized_return_pct(avg_final_balance, initial_balance, days), 4
        ),
        "average_profit_factor": round(
            sum(item["profit_factor"] for item in symbol_metrics) / len(symbol_metrics), 4
        ),
        "average_max_drawdown_pct": round(
            sum(item["max_drawdown_pct"] for item in symbol_metrics) / len(symbol_metrics), 4
        ),
        "total_trades": int(sum(item["trades"] for item in symbol_metrics)),
        "trades_per_month": round(sum(item["trades"] for item in symbol_metrics) / (days / 30.4375), 4),
    }


def load_symbol_data(ohlcv_dir: str, symbols: List[str]) -> Dict[str, Dict[str, List[List[Any]]]]:
    loaded: Dict[str, Dict[str, List[List[Any]]]] = {}
    for symbol in symbols:
        file_1h = os.path.join(ohlcv_dir, f"{symbol}_60.csv")
        file_4h = os.path.join(ohlcv_dir, f"{symbol}_240.csv")
        if not os.path.exists(file_1h) or not os.path.exists(file_4h):
            continue
        klines_1h = load_ohlcv_csv(file_1h)
        klines_4h = load_ohlcv_csv(file_4h)
        if len(klines_1h) < 50 or len(klines_4h) < 50:
            continue
        loaded[symbol] = {"1h": klines_1h, "4h": klines_4h}
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--thresholds", default="0.70,0.75,0.80")
    parser.add_argument("--distances", default="0.30,0.40,0.50")
    parser.add_argument("--rr-targets", default="2.8,3.2")
    parser.add_argument("--stop-buffers", default="0.20,0.25")
    parser.add_argument("--strict-trend", default="false,true")
    parser.add_argument("--top-n", default="5,8")
    parser.add_argument("--min-train-trades", type=int, default=12)
    parser.add_argument("--min-test-trades", type=int, default=6)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.2)
    parser.add_argument("--top-results", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_symbols = normalize_symbols(SYMBOLS) if args.symbols == "all" else normalize_symbols(args.symbols.split(","))
    loaded_data = load_symbol_data(args.ohlcv_dir, selected_symbols)
    if not loaded_data:
        raise ValueError("Не удалось загрузить OHLCV для исследования")

    global_start, global_end = find_global_period(loaded_data)
    total_seconds = (global_end - global_start).total_seconds()
    split_dt = global_start + timedelta(seconds=total_seconds * args.split_ratio)
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    distances = [float(value) for value in args.distances.split(",") if value.strip()]
    rr_targets = [float(value) for value in args.rr_targets.split(",") if value.strip()]
    stop_buffers = [float(value) for value in args.stop_buffers.split(",") if value.strip()]
    strict_trend_values = [value.strip().lower() == "true" for value in args.strict_trend.split(",") if value.strip()]
    top_n_values = [int(value) for value in args.top_n.split(",") if value.strip()]

    search_results: List[Dict[str, Any]] = []
    symbol_cache: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = {}

    grid = list(
        itertools.product(
            thresholds,
            distances,
            rr_targets,
            stop_buffers,
            strict_trend_values,
        )
    )

    for threshold, min_distance, rr_target, stop_buffer, strict_trend in grid:
        params = TrendPullbackParams(
            threshold=threshold,
            min_distance_score=min_distance,
            rr_target=rr_target,
            stop_buffer_atr=stop_buffer,
            strict_trend_alignment=strict_trend,
        )
        strategy_fn = make_simple_trend_pullback_strategy(params)
        cache_key = (
            threshold,
            min_distance,
            rr_target,
            stop_buffer,
            strict_trend,
        )

        per_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, payload in loaded_data.items():
            rows, _ = run_strategy_backtest_klines(
                strategy_name="simple_trend_pullback_search",
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
        symbol_cache[cache_key] = per_symbol

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
                        "threshold": threshold,
                        "min_distance_score": min_distance,
                        "rr_target": rr_target,
                        "stop_buffer_atr": stop_buffer,
                        "strict_trend_alignment": strict_trend,
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

    summary_path = os.path.join(args.output_dir, "trend_pullback_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
