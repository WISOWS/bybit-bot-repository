import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from backtest import (
    BACKTEST_SLIPPAGE_RATE,
    BACKTEST_TAKER_FEE_RATE,
    apply_adverse_slippage,
    load_ohlcv_csv,
    simulate_exit,
)
from main import BASE_DIR, LEVERAGE, RISK_PER_TRADE, SYMBOLS, calc_qty_by_margin, calc_qty_by_risk
from research_strategies import STRATEGIES, STRATEGY_DESCRIPTIONS, StrategySetup

RESEARCH_FIELDNAMES = [
    "strategy",
    "timestamp",
    "symbol",
    "side",
    "entry",
    "stop",
    "tp",
    "risk_usdt",
    "planned_rr",
    "atr_4h",
    "trend",
    "status",
    "realized_pnl",
    "note",
]


def ms_to_iso(ms_value: int) -> str:
    return datetime.fromtimestamp(ms_value / 1000, tz=timezone.utc).isoformat()


def normalize_symbols(symbols: List[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def normalize_strategy_names(values: List[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for raw in values:
        name = str(raw).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def profit_factor(values: List[float]) -> float:
    gross_profit = sum(value for value in values if value > 0)
    gross_loss = abs(sum(value for value in values if value < 0))
    if gross_loss <= 0:
        return gross_profit if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(balance_points: List[float]) -> float:
    peak = 0.0
    drawdown = 0.0
    for value in balance_points:
        peak = max(peak, value)
        if peak > 0:
            drawdown = min(drawdown, (value - peak) / peak)
    return abs(drawdown)


def summarize_rows(rows: List[Dict[str, Any]], initial_balance: float) -> Dict[str, Any]:
    pnls = [float(row["realized_pnl"]) for row in rows]
    risks = [float(row["risk_usdt"]) for row in rows if float(row["risk_usdt"]) > 0]
    r_values = [
        float(row["realized_pnl"]) / float(row["risk_usdt"])
        for row in rows
        if float(row["risk_usdt"]) > 0
    ]
    balance_curve = [initial_balance]
    balance = initial_balance
    for pnl in pnls:
        balance += pnl
        balance_curve.append(balance)

    return {
        "trades": len(rows),
        "net_pnl": round(sum(pnls), 4),
        "avg_r": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
        "winrate": round(sum(1 for value in r_values if value > 0) / len(r_values), 4) if r_values else 0.0,
        "profit_factor": round(profit_factor(pnls), 4),
        "max_drawdown_pct": round(max_drawdown(balance_curve) * 100, 4),
        "final_balance": round(balance_curve[-1], 4),
    }


def run_strategy_backtest(
    strategy_name: str,
    symbol: str,
    file_1h: str,
    file_4h: str,
    initial_balance: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    klines_1h = load_ohlcv_csv(file_1h)
    klines_4h = load_ohlcv_csv(file_4h)

    if len(klines_1h) < 50:
        raise ValueError("Недостаточно 1H данных")
    if len(klines_4h) < 50:
        raise ValueError("Недостаточно 4H данных")

    strategy_fn = STRATEGIES[strategy_name]
    balance = initial_balance
    rows: List[Dict[str, Any]] = []
    times_4h = [int(row[0]) for row in klines_4h]
    idx_4h = -1
    idx_1h = 1

    while idx_1h < len(klines_1h) - 1:
        current_time_ms = int(klines_1h[idx_1h][0])
        while idx_4h + 1 < len(times_4h) and times_4h[idx_4h + 1] <= current_time_ms:
            idx_4h += 1

        if idx_4h < 0:
            idx_1h += 1
            continue

        setup: StrategySetup | None = strategy_fn(klines_1h, idx_1h, klines_4h, idx_4h)
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
        note_payload = {
            "message": "research_backtest",
            "strategy": strategy_name,
            "strategy_description": STRATEGY_DESCRIPTIONS[strategy_name],
            "effective_entry": round(effective_entry, 8),
            "effective_exit": round(effective_exit, 8),
            "entry_fee": round(entry_fee, 8),
            "exit_fee": round(exit_fee, 8),
            "slippage_rate": BACKTEST_SLIPPAGE_RATE,
            "taker_fee_rate": BACKTEST_TAKER_FEE_RATE,
            "qty": round(qty, 8),
            "exit_reason": exit_reason,
            "entry_time_ms": current_time_ms,
            "exit_time_ms": int(klines_1h[exit_index][0]),
            "balance_after": round(balance, 8),
            **setup.meta,
        }

        rows.append(
            {
                "strategy": strategy_name,
                "timestamp": ms_to_iso(current_time_ms),
                "symbol": symbol,
                "side": setup.side,
                "entry": round(setup.entry, 8),
                "stop": round(setup.stop, 8),
                "tp": round(setup.tp, 8),
                "risk_usdt": round(risk_usdt, 8),
                "planned_rr": round(setup.planned_rr, 4),
                "atr_4h": round(setup.atr_4h, 8),
                "trend": str(setup.meta.get("trend", "")),
                "status": "closed_research",
                "realized_pnl": round(realized_pnl, 8),
                "note": json.dumps(note_payload, ensure_ascii=False, sort_keys=True),
            }
        )

        idx_1h = exit_index + 1

    return rows, summarize_rows(rows, initial_balance)


def write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESEARCH_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", default="all")
    parser.add_argument("--symbols", default="all")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "research_results"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_strategies = list(STRATEGIES.keys()) if args.strategies == "all" else normalize_strategy_names(args.strategies.split(","))
    selected_strategies = [name for name in selected_strategies if name in STRATEGIES]
    if not selected_strategies:
        raise ValueError("Не выбрано ни одной стратегии")

    selected_symbols = normalize_symbols(SYMBOLS) if args.symbols == "all" else normalize_symbols(args.symbols.split(","))
    if not selected_symbols:
        raise ValueError("Не выбрано ни одного символа")

    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"strategies": {}}

    for strategy_name in selected_strategies:
        strategy_rows: List[Dict[str, Any]] = []
        symbol_summaries: Dict[str, Any] = {}
        ok_symbol_metrics: List[Dict[str, Any]] = []
        for symbol in selected_symbols:
            file_1h = os.path.join(args.ohlcv_dir, f"{symbol}_60.csv")
            file_4h = os.path.join(args.ohlcv_dir, f"{symbol}_240.csv")
            if not os.path.exists(file_1h) or not os.path.exists(file_4h):
                symbol_summaries[symbol] = {"status": "missing_ohlcv"}
                continue
            try:
                rows, metrics = run_strategy_backtest(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    file_1h=file_1h,
                    file_4h=file_4h,
                    initial_balance=args.initial_balance,
                )
                strategy_rows.extend(rows)
                symbol_summaries[symbol] = {"status": "ok", **metrics}
                ok_symbol_metrics.append(metrics)
            except Exception as exc:
                symbol_summaries[symbol] = {"status": str(exc)}

        strategy_summary = summarize_rows(strategy_rows, args.initial_balance)
        strategy_summary["active_symbols"] = len(ok_symbol_metrics)
        if ok_symbol_metrics:
            strategy_summary["average_symbol_final_balance"] = round(
                sum(item["final_balance"] for item in ok_symbol_metrics) / len(ok_symbol_metrics),
                4,
            )
            strategy_summary["average_symbol_drawdown_pct"] = round(
                sum(item["max_drawdown_pct"] for item in ok_symbol_metrics) / len(ok_symbol_metrics),
                4,
            )
        strategy_summary["description"] = STRATEGY_DESCRIPTIONS[strategy_name]
        strategy_summary["symbols"] = symbol_summaries
        summary["strategies"][strategy_name] = strategy_summary
        all_rows.extend(strategy_rows)

    summary_path = os.path.join(args.output_dir, "summary.json")
    trades_path = os.path.join(args.output_dir, "trades.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    write_rows(trades_path, all_rows)

    print(f"strategies={len(summary['strategies'])}")
    print(f"trades_output={trades_path}")
    print(f"summary_output={summary_path}")
    for strategy_name, strategy_summary in summary["strategies"].items():
        print(json.dumps({"strategy": strategy_name, **{k: v for k, v in strategy_summary.items() if k != 'symbols'}}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
