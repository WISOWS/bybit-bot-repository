import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from backtest import run_backtest, write_backtest_rows
from download_ohlcv import download_ohlcv, parse_datetime_to_ms
from main import BACKTEST_JOURNAL_PATH, BASE_DIR, SYMBOLS, update_adaptive_model


def load_symbols() -> List[str]:
    seen = set()
    normalized: List[str] = []
    for raw in SYMBOLS:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def row_timestamp_ms(row: Dict[str, Any]) -> int:
    timestamp = str(row.get("timestamp", "")).strip()
    if not timestamp:
        return 0
    try:
        return int(datetime.fromisoformat(timestamp).timestamp() * 1000)
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--category", default="linear")
    parser.add_argument("--ohlcv-dir", default=os.path.join(BASE_DIR, "ohlcv"))
    parser.add_argument("--backtest-output", default=BACKTEST_JOURNAL_PATH)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    args = parser.parse_args()

    start_ms = parse_datetime_to_ms(args.start)
    end_ms = parse_datetime_to_ms(args.end) if args.end else int(time.time() * 1000)
    if end_ms < start_ms:
        raise ValueError("end должен быть >= start")

    os.makedirs(args.ohlcv_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    symbol_summaries: List[Dict[str, Any]] = []

    for symbol in load_symbols():
        file_1h = os.path.join(args.ohlcv_dir, f"{symbol}_60.csv")
        file_4h = os.path.join(args.ohlcv_dir, f"{symbol}_240.csv")

        try:
            candles_1h = download_ohlcv(
                category=args.category,
                symbol=symbol,
                interval="60",
                start_ms=start_ms,
                end_ms=end_ms,
                include_open_candle=False,
                output_path=file_1h,
            )
            candles_4h = download_ohlcv(
                category=args.category,
                symbol=symbol,
                interval="240",
                start_ms=start_ms,
                end_ms=end_ms,
                include_open_candle=False,
                output_path=file_4h,
            )
        except Exception as exc:
            symbol_summaries.append(
                {
                    "symbol": symbol,
                    "candles_1h": 0,
                    "candles_4h": 0,
                    "trades": 0,
                    "final_balance": None,
                    "status": f"download_error: {exc}",
                }
            )
            continue

        try:
            rows, final_balance = run_backtest(
                symbol=symbol,
                file_1h=file_1h,
                file_4h=file_4h,
                initial_balance=args.initial_balance,
            )
        except ValueError as exc:
            symbol_summaries.append(
                {
                    "symbol": symbol,
                    "candles_1h": candles_1h,
                    "candles_4h": candles_4h,
                    "trades": 0,
                    "final_balance": None,
                    "status": str(exc),
                }
            )
            continue

        all_rows.extend(rows)
        symbol_summaries.append(
            {
                "symbol": symbol,
                "candles_1h": candles_1h,
                "candles_4h": candles_4h,
                "trades": len(rows),
                "final_balance": round(final_balance, 2),
                "status": "ok",
            }
        )

    all_rows.sort(key=row_timestamp_ms)
    write_backtest_rows(args.backtest_output, all_rows)

    adaptive_state = update_adaptive_model()

    print(f"symbols={len(symbol_summaries)}")
    print(f"backtest_rows={len(all_rows)}")
    print(f"backtest_output={args.backtest_output}")
    if adaptive_state:
        print(
            "adaptive_state="
            + json.dumps(
                {
                    "threshold": adaptive_state["threshold"],
                    "trade_count": adaptive_state["trade_count"],
                    "live_trade_count": adaptive_state["live_trade_count"],
                    "backtest_trade_count": adaptive_state["backtest_trade_count"],
                    "weights": adaptive_state["weights"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    for item in symbol_summaries:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
