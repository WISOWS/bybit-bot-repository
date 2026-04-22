import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from main import (
    JOURNAL_FIELDNAMES,
    LEVERAGE,
    LIVE_SL_BUFFER,
    MAX_LEVEL_DISTANCE_ATR,
    MAX_STOP_ATR,
    MIN_RR,
    RISK_PER_TRADE,
    SIGNAL_SCORE_THRESHOLD,
    SL_ATR_BUFFER,
    calc_adaptive_edge_score,
    calc_atr_from_klines,
    calc_qty_by_margin,
    calc_qty_by_risk,
    detect_trend_4h,
    find_level_break_trend,
    get_adaptive_threshold,
    impulse_filter_ok,
    is_range_dirty_around_level,
    level_touched,
)

BACKTEST_TAKER_FEE_RATE = 0.00055
BACKTEST_SLIPPAGE_RATE = 0.0005


def load_ohlcv_csv(path: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    str(row["start_time_ms"]),
                    str(row["open"]),
                    str(row["high"]),
                    str(row["low"]),
                    str(row["close"]),
                    str(row["volume"]),
                    str(row["turnover"]),
                ]
            )

    return sorted(rows, key=lambda item: int(item[0]))


def ms_to_iso(ms_value: int) -> str:
    return datetime.fromtimestamp(ms_value / 1000, tz=timezone.utc).isoformat()


def simulate_exit(
    klines_1h: List[List[str]],
    entry_index: int,
    side: str,
    stop: float,
    tp: float,
) -> Tuple[float, str, int]:
    for idx in range(entry_index + 1, len(klines_1h)):
        candle = klines_1h[idx]
        high = float(candle[2])
        low = float(candle[3])

        if side == "Buy":
            if low <= stop:
                return stop, "stop_loss", idx
            if high >= tp:
                return tp, "take_profit", idx
        else:
            if high >= stop:
                return stop, "stop_loss", idx
            if low <= tp:
                return tp, "take_profit", idx

    last_index = len(klines_1h) - 1
    return float(klines_1h[last_index][4]), "end_of_data", last_index


def apply_adverse_slippage(price: float, side: str, action: str) -> float:
    if price <= 0:
        return price

    if action == "entry":
        if side == "Buy":
            return price * (1 + BACKTEST_SLIPPAGE_RATE)
        return price * (1 - BACKTEST_SLIPPAGE_RATE)

    if side == "Buy":
        return price * (1 - BACKTEST_SLIPPAGE_RATE)
    return price * (1 + BACKTEST_SLIPPAGE_RATE)


def run_backtest(
    symbol: str,
    file_1h: str,
    file_4h: str,
    initial_balance: float,
) -> Tuple[List[Dict[str, Any]], float]:
    klines_1h = load_ohlcv_csv(file_1h)
    klines_4h = load_ohlcv_csv(file_4h)

    if len(klines_1h) < 12:
        raise ValueError("Недостаточно 1H данных")
    if len(klines_4h) < 30:
        raise ValueError("Недостаточно 4H данных")

    balance = initial_balance
    backtest_rows: List[Dict[str, Any]] = []
    times_4h = [int(row[0]) for row in klines_4h]
    idx_4h = -1
    idx_1h = 11

    while idx_1h < len(klines_1h) - 1:
        current_1h = klines_1h[idx_1h]
        current_time_ms = int(current_1h[0])

        while idx_4h + 1 < len(times_4h) and times_4h[idx_4h + 1] <= current_time_ms:
            idx_4h += 1

        if idx_4h < 29:
            idx_1h += 1
            continue

        recent_4h_30 = klines_4h[idx_4h - 29 : idx_4h + 1]
        recent_4h_20 = klines_4h[idx_4h - 19 : idx_4h + 1]
        recent_4h_15 = klines_4h[idx_4h - 14 : idx_4h + 1]
        trend = detect_trend_4h(recent_4h_30)
        if trend == "FLAT":
            idx_1h += 1
            continue

        level_pair = find_level_break_trend(recent_4h_30)
        if level_pair is None:
            idx_1h += 1
            continue

        atr_4h = calc_atr_from_klines(recent_4h_15, period=14)
        if atr_4h <= 0:
            idx_1h += 1
            continue

        level_low, level_high = level_pair
        level = level_low if trend == "UP" else level_high
        side = "Buy" if trend == "UP" else "Sell"
        entry = float(current_1h[4])
        max_level_distance = atr_4h * MAX_LEVEL_DISTANCE_ATR

        trend_score = 1.0 if (
            (trend == "UP" and entry >= level)
            or (trend == "DOWN" and entry <= level)
        ) else 0.3

        if max_level_distance > 0:
            distance_score = max(0.0, 1 - abs(entry - level) / max_level_distance)
        else:
            distance_score = 0.0

        recent_1h_12 = klines_1h[idx_1h - 11 : idx_1h + 1]
        impulse_ok_value = impulse_filter_ok(recent_1h_12, len(recent_1h_12) - 1, side)
        impulse_score = 1.0 if impulse_ok_value else 0.3

        clean_level_ok = not is_range_dirty_around_level(recent_4h_20, level)
        level_score = 1.0 if clean_level_ok else 0.2

        structure_touch = level_touched(current_1h, level, atr_4h * 0.2)
        structure_score = 1.0 if structure_touch else 0.0

        edge_score = calc_adaptive_edge_score(
            trend_score,
            level_score,
            distance_score,
            impulse_score,
            structure_score,
        )
        score_threshold = get_adaptive_threshold(SIGNAL_SCORE_THRESHOLD / 3.0)

        if edge_score < score_threshold:
            idx_1h += 1
            continue

        if side == "Buy":
            stop = level - atr_4h * SL_ATR_BUFFER
            tp = entry + (entry - stop) * MIN_RR
            risk_per_unit = entry - stop
            reward_per_unit = tp - entry
        else:
            stop = level + atr_4h * SL_ATR_BUFFER
            tp = entry - (stop - entry) * MIN_RR
            risk_per_unit = stop - entry
            reward_per_unit = entry - tp

        if risk_per_unit <= 0 or reward_per_unit <= 0:
            idx_1h += 1
            continue

        rr = reward_per_unit / risk_per_unit
        if rr < MIN_RR:
            idx_1h += 1
            continue

        if risk_per_unit > atr_4h * MAX_STOP_ATR:
            idx_1h += 1
            continue

        if risk_per_unit < atr_4h * LIVE_SL_BUFFER:
            idx_1h += 1
            continue

        qty_by_risk = calc_qty_by_risk(balance, RISK_PER_TRADE, entry, stop)
        qty_by_margin = calc_qty_by_margin(balance, LEVERAGE, entry)
        qty = min(qty_by_risk, qty_by_margin)
        if qty <= 0:
            idx_1h += 1
            continue

        effective_entry = apply_adverse_slippage(entry, side, "entry")
        exit_price, exit_reason, exit_index = simulate_exit(klines_1h, idx_1h, side, stop, tp)
        effective_exit = apply_adverse_slippage(exit_price, side, "exit")
        pnl_per_unit = effective_exit - effective_entry if side == "Buy" else effective_entry - effective_exit
        entry_fee = qty * effective_entry * BACKTEST_TAKER_FEE_RATE
        exit_fee = qty * effective_exit * BACKTEST_TAKER_FEE_RATE
        realized_pnl = qty * pnl_per_unit - entry_fee - exit_fee
        risk_usdt = qty * risk_per_unit

        balance += realized_pnl

        note_payload = {
            "message": "backtest",
            "edge_score": round(edge_score, 6),
            "trend_score": round(trend_score, 6),
            "level_score": round(level_score, 6),
            "distance_score": round(distance_score, 6),
            "impulse_score": round(impulse_score, 6),
            "structure_score": round(structure_score, 6),
            "entry_time_ms": current_time_ms,
            "exit_time_ms": int(klines_1h[exit_index][0]),
            "exit_price": round(exit_price, 8),
            "effective_entry": round(effective_entry, 8),
            "effective_exit": round(effective_exit, 8),
            "exit_reason": exit_reason,
            "entry_fee": round(entry_fee, 8),
            "exit_fee": round(exit_fee, 8),
            "slippage_rate": BACKTEST_SLIPPAGE_RATE,
            "taker_fee_rate": BACKTEST_TAKER_FEE_RATE,
            "qty": round(qty, 8),
            "balance_after": round(balance, 8),
        }

        backtest_rows.append(
            {
                "timestamp": ms_to_iso(current_time_ms),
                "symbol": symbol,
                "side": side,
                "entry": round(entry, 8),
                "stop": round(stop, 8),
                "tp": round(tp, 8),
                "risk_usdt": round(risk_usdt, 8),
                "planned_rr": round(rr, 4),
                "atr_4h": round(atr_4h, 8),
                "trend": trend,
                "level": round(level, 8),
                "impulse_ok": impulse_ok_value,
                "status": "closed_backtest",
                "realized_pnl": round(realized_pnl, 8),
                "note": json.dumps(note_payload, ensure_ascii=False, sort_keys=True),
            }
        )

        idx_1h = exit_index + 1

    return backtest_rows, balance


def write_backtest_rows(output_path: str, rows: List[Dict[str, Any]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--file-1h", default=os.path.join(base_dir, "BTCUSDT_60.csv"))
    parser.add_argument("--file-4h", default=os.path.join(base_dir, "BTCUSDT_240.csv"))
    parser.add_argument("--output", default=os.path.join(base_dir, "backtest_journal.csv"))
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    args = parser.parse_args()

    rows, final_balance = run_backtest(
        symbol=args.symbol.upper(),
        file_1h=args.file_1h,
        file_4h=args.file_4h,
        initial_balance=args.initial_balance,
    )
    write_backtest_rows(args.output, rows)
    print(f"trades={len(rows)} final_balance={final_balance:.2f} output={args.output}")


if __name__ == "__main__":
    main()
