import argparse
import bisect
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from backtest import load_ohlcv_csv
from main import BASE_DIR, RISK_PER_TRADE, calc_atr_from_klines, detect_trend_4h
from research_backtest import max_drawdown, normalize_symbols, profit_factor, run_strategy_backtest_klines
from research_search import annualized_return_pct, find_global_period
from research_strategies import RegimeSwitchParams, make_regime_switch_strategy


@dataclass(frozen=True)
class ModelSpec:
    name: str
    symbol: str
    params: RegimeSwitchParams


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="near_regime_x",
        symbol="NEARUSDT",
        params=RegimeSwitchParams(
            trend_threshold=0.65,
            trend_min_distance_score=0.5,
            trend_rr_target=3.5,
            trend_stop_buffer_atr=0.2,
            range_zone_fraction=0.12,
            range_target_fraction=0.6,
        ),
    ),
    ModelSpec(
        name="sol_regime",
        symbol="SOLUSDT",
        params=RegimeSwitchParams(
            trend_threshold=0.65,
            trend_min_distance_score=0.4,
            trend_rr_target=3.5,
            trend_stop_buffer_atr=0.25,
            range_zone_fraction=0.16,
            range_target_fraction=0.7,
        ),
    ),
    ModelSpec(
        name="link_regime",
        symbol="LINKUSDT",
        params=RegimeSwitchParams(
            trend_threshold=0.65,
            trend_min_distance_score=0.4,
            trend_rr_target=3.5,
            trend_stop_buffer_atr=0.2,
            range_zone_fraction=0.08,
            range_target_fraction=0.5,
        ),
    ),
    ModelSpec(
        name="ena_regime",
        symbol="ENAUSDT",
        params=RegimeSwitchParams(
            trend_threshold=0.65,
            trend_min_distance_score=0.4,
            trend_rr_target=3.5,
            trend_stop_buffer_atr=0.25,
            range_zone_fraction=0.12,
            range_target_fraction=0.7,
        ),
    ),
]


def build_4h_index(klines_4h: List[List[Any]]) -> List[int]:
    return [int(row[0]) for row in klines_4h]


def latest_4h_idx(times_4h: List[int], target_ms: int) -> int:
    return bisect.bisect_right(times_4h, target_ms) - 1


def btc_mode_ok(mode: str, side: str, btc_4h: List[List[Any]], btc_times_4h: List[int], timestamp_ms: int) -> bool:
    if mode == "none":
        return True
    idx_4h = latest_4h_idx(btc_times_4h, timestamp_ms)
    if idx_4h < 29:
        return False
    trend = detect_trend_4h(btc_4h[idx_4h - 29 : idx_4h + 1])
    if side == "Buy":
        if mode == "allow_flat":
            return trend in {"UP", "FLAT"}
        return trend == "UP"
    if mode == "allow_flat":
        return trend in {"DOWN", "FLAT"}
    return trend == "DOWN"


def hot_vol_ratio_at(klines_4h: List[List[Any]], times_4h: List[int], timestamp_ms: int) -> Optional[float]:
    idx_4h = latest_4h_idx(times_4h, timestamp_ms)
    if idx_4h < 14:
        return None
    short = calc_atr_from_klines(klines_4h[idx_4h - 6 : idx_4h + 1], period=6)
    long = calc_atr_from_klines(klines_4h[idx_4h - 14 : idx_4h + 1], period=14)
    if short <= 0 or long <= 0:
        return None
    return short / long


def load_trade_stream(
    model: ModelSpec,
    klines_1h: List[List[Any]],
    klines_4h: List[List[Any]],
    initial_balance: float,
) -> List[Dict[str, Any]]:
    rows, _ = run_strategy_backtest_klines(
        strategy_name=model.name,
        strategy_fn=make_regime_switch_strategy(model.params),
        symbol=model.symbol,
        klines_1h=klines_1h,
        klines_4h=klines_4h,
        initial_balance=initial_balance,
    )
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        note = json.loads(row["note"])
        risk_usdt = float(row["risk_usdt"])
        realized_pnl = float(row["realized_pnl"])
        if risk_usdt <= 0:
            continue
        parsed.append(
            {
                "model": model.name,
                "symbol": model.symbol,
                "side": str(row["side"]),
                "entry_time_ms": int(note["entry_time_ms"]),
                "exit_time_ms": int(note["exit_time_ms"]),
                "r_multiple": realized_pnl / risk_usdt,
                "planned_rr": float(row["planned_rr"]),
                "note": note,
            }
        )
    return parsed


def filter_trade_stream(
    trades: List[Dict[str, Any]],
    symbol_4h: List[List[Any]],
    btc_4h: List[List[Any]],
    btc_mode: str,
    min_hot_vol_ratio: float,
) -> List[Dict[str, Any]]:
    symbol_times_4h = build_4h_index(symbol_4h)
    btc_times_4h = build_4h_index(btc_4h)
    filtered: List[Dict[str, Any]] = []
    for trade in trades:
        ts = int(trade["entry_time_ms"])
        if not btc_mode_ok(btc_mode, str(trade["side"]), btc_4h, btc_times_4h, ts):
            continue
        if min_hot_vol_ratio > 0:
            ratio = hot_vol_ratio_at(symbol_4h, symbol_times_4h, ts)
            if ratio is None or ratio < min_hot_vol_ratio:
                continue
        filtered.append(dict(trade))
    return filtered


def split_trades_by_time(
    trades: List[Dict[str, Any]],
    split_dt: datetime,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_ms = int(split_dt.timestamp() * 1000)
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for trade in trades:
        if int(trade["entry_time_ms"]) <= split_ms:
            train.append(trade)
        else:
            test.append(trade)
    return train, test


def summarize_trade_stream(
    trades: List[Dict[str, Any]],
    initial_balance: float,
    days: float,
    max_concurrent: int,
) -> Dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "executed_trades": 0,
            "skipped_trades": 0,
            "symbols": 0,
            "net_pnl": 0.0,
            "avg_r": 0.0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "final_balance": initial_balance,
            "annualized_return_pct": 0.0,
            "trades_per_month": 0.0,
        }

    events: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for idx, trade in enumerate(trades):
        events.append((int(trade["exit_time_ms"]), 0, f"x{idx}", trade))
        events.append((int(trade["entry_time_ms"]), 1, f"e{idx}", trade))
    events.sort()

    balance = initial_balance
    balance_curve = [initial_balance]
    realized_pnls: List[float] = []
    executed_rs: List[float] = []
    active: Dict[str, Dict[str, Any]] = {}
    skipped_trades = 0
    executed_symbols = set()

    for _, event_type, event_id, trade in events:
        if event_type == 0:
            active_trade = active.pop(event_id.replace("x", "e"), None)
            if active_trade is None:
                continue
            pnl = float(active_trade["pnl"])
            balance += pnl
            realized_pnls.append(pnl)
            executed_rs.append(float(active_trade["r_multiple"]))
            executed_symbols.add(str(active_trade["symbol"]))
            balance_curve.append(balance)
            continue

        if len(active) >= max_concurrent:
            skipped_trades += 1
            continue

        risk_amount = balance * RISK_PER_TRADE
        pnl = risk_amount * float(trade["r_multiple"])
        active[event_id] = {
            "symbol": str(trade["symbol"]),
            "pnl": pnl,
            "r_multiple": float(trade["r_multiple"]),
        }

    avg_r = sum(executed_rs) / len(executed_rs) if executed_rs else 0.0
    winrate = sum(1 for value in executed_rs if value > 0) / len(executed_rs) if executed_rs else 0.0
    final_balance = balance_curve[-1]

    return {
        "trades": len(trades),
        "executed_trades": len(realized_pnls),
        "skipped_trades": skipped_trades,
        "symbols": len(executed_symbols),
        "net_pnl": round(sum(realized_pnls), 4),
        "avg_r": round(avg_r, 4),
        "winrate": round(winrate, 4),
        "profit_factor": round(profit_factor(realized_pnls), 4),
        "max_drawdown_pct": round(max_drawdown(balance_curve) * 100, 4),
        "final_balance": round(final_balance, 4),
        "annualized_return_pct": round(annualized_return_pct(final_balance, initial_balance, days), 4),
        "trades_per_month": round(len(realized_pnls) / (days / 30.4375), 4) if days > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="NEARUSDT,SOLUSDT,LINKUSDT,ENAUSDT")
    parser.add_argument("--btc-symbol", default="BTCUSDT")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_search_meta_portfolio"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--split-ratio", type=float, default=0.7)
    parser.add_argument("--btc-modes", default="none,allow_flat,strict")
    parser.add_argument("--hot-vol-ratios", default="0.0,0.75,0.8,0.85")
    parser.add_argument("--pool-sizes", default="1,2,3,4")
    parser.add_argument("--max-concurrent-values", default="1,2,3,4")
    parser.add_argument("--min-train-trades", type=int, default=20)
    parser.add_argument("--min-test-trades", type=int, default=8)
    parser.add_argument("--min-train-profit-factor", type=float, default=1.15)
    parser.add_argument("--top-results", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    selected_symbols = set(normalize_symbols(args.symbols.split(",")))
    selected_models = [model for model in MODEL_SPECS if model.symbol in selected_symbols]
    if not selected_models:
        raise ValueError("Не выбрано ни одной модели для meta-portfolio")

    btc_4h_path = os.path.join(args.ohlcv_dir, f"{args.btc_symbol}_240.csv")
    if not os.path.exists(btc_4h_path):
        raise FileNotFoundError(f"Нет BTC 4H файла: {btc_4h_path}")
    btc_4h = load_ohlcv_csv(btc_4h_path)

    loaded_data: Dict[str, Dict[str, List[List[Any]]]] = {}
    for model in selected_models:
        file_1h = os.path.join(args.ohlcv_dir, f"{model.symbol}_60.csv")
        file_4h = os.path.join(args.ohlcv_dir, f"{model.symbol}_240.csv")
        if not os.path.exists(file_1h) or not os.path.exists(file_4h):
            continue
        loaded_data[model.symbol] = {
            "1h": load_ohlcv_csv(file_1h),
            "4h": load_ohlcv_csv(file_4h),
        }
    if not loaded_data:
        raise ValueError("Не удалось загрузить OHLCV для meta-portfolio")

    global_start, global_end = find_global_period(
        {
            **loaded_data,
            args.btc_symbol: {"1h": next(iter(loaded_data.values()))["1h"], "4h": btc_4h},
        }
    )
    split_dt = global_start + timedelta(seconds=(global_end - global_start).total_seconds() * args.split_ratio)
    train_days = max((split_dt - global_start).total_seconds() / 86400.0, 1.0)
    test_days = max((global_end - split_dt).total_seconds() / 86400.0, 1.0)

    base_streams: Dict[str, List[Dict[str, Any]]] = {}
    for model in selected_models:
        payload = loaded_data.get(model.symbol)
        if payload is None:
            continue
        base_streams[model.name] = load_trade_stream(
            model=model,
            klines_1h=payload["1h"],
            klines_4h=payload["4h"],
            initial_balance=args.initial_balance,
        )

    btc_modes = [value.strip() for value in args.btc_modes.split(",") if value.strip()]
    hot_vol_ratios = [float(value) for value in args.hot_vol_ratios.split(",") if value.strip()]
    pool_sizes = [int(value) for value in args.pool_sizes.split(",") if value.strip()]
    max_concurrent_values = [int(value) for value in args.max_concurrent_values.split(",") if value.strip()]

    filtered_cache: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = {}
    search_results: List[Dict[str, Any]] = []

    for btc_mode, hot_vol_ratio in itertools.product(btc_modes, hot_vol_ratios):
        for model in selected_models:
            cache_key = (model.name, hot_vol_ratio, btc_mode)
            filtered_cache[cache_key] = filter_trade_stream(
                trades=base_streams.get(model.name, []),
                symbol_4h=loaded_data[model.symbol]["4h"],
                btc_4h=btc_4h,
                btc_mode=btc_mode,
                min_hot_vol_ratio=hot_vol_ratio,
            )

        for pool_size in pool_sizes:
            for combo in itertools.combinations(selected_models, pool_size):
                model_names = [model.name for model in combo]
                combined_trades: List[Dict[str, Any]] = []
                for model in combo:
                    combined_trades.extend(filtered_cache[(model.name, hot_vol_ratio, btc_mode)])
                combined_trades.sort(key=lambda item: (int(item["entry_time_ms"]), int(item["exit_time_ms"])))

                train_trades, test_trades = split_trades_by_time(combined_trades, split_dt)
                for max_concurrent in max_concurrent_values:
                    train_summary = summarize_trade_stream(
                        trades=train_trades,
                        initial_balance=args.initial_balance,
                        days=train_days,
                        max_concurrent=max_concurrent,
                    )
                    if train_summary["executed_trades"] < args.min_train_trades:
                        continue
                    if train_summary["profit_factor"] < args.min_train_profit_factor:
                        continue
                    if train_summary["net_pnl"] <= 0:
                        continue

                    test_summary = summarize_trade_stream(
                        trades=test_trades,
                        initial_balance=args.initial_balance,
                        days=test_days,
                        max_concurrent=max_concurrent,
                    )
                    if test_summary["executed_trades"] < args.min_test_trades:
                        continue

                    search_results.append(
                        {
                            "params": {
                                "btc_mode": btc_mode,
                                "min_hot_vol_ratio": hot_vol_ratio,
                                "pool_size": pool_size,
                                "max_concurrent": max_concurrent,
                            },
                            "models": model_names,
                            "symbols": [model.symbol for model in combo],
                            "split": {
                                "start": global_start.isoformat(),
                                "split": split_dt.isoformat(),
                                "end": global_end.isoformat(),
                                "train_days": round(train_days, 4),
                                "test_days": round(test_days, 4),
                            },
                            "train": train_summary,
                            "test": test_summary,
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

    output_payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models_considered": len(selected_models),
            "results_count": len(search_results),
        },
        "results": search_results[: args.top_results],
    }

    summary_path = os.path.join(args.output_dir, "meta_portfolio_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"summary_output={summary_path}")
    print(f"results={len(search_results)}")
    for item in output_payload["results"][:10]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
