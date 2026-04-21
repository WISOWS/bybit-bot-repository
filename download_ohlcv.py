import argparse
import csv
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

import requests


BASE_URL = "https://api.bybit.com"
DEFAULT_CATEGORY = "linear"
DEFAULT_LIMIT = 1000
DEFAULT_INTERVALS = ["60", "240"]
DEFAULT_START = "2020-01-01"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5

INTERVAL_TO_MS: Dict[str, int] = {
    "1": 60_000,
    "3": 180_000,
    "5": 300_000,
    "15": 900_000,
    "30": 1_800_000,
    "60": 3_600_000,
    "120": 7_200_000,
    "240": 14_400_000,
    "360": 21_600_000,
    "720": 43_200_000,
    "D": 86_400_000,
    "W": 604_800_000,
}


def parse_datetime_to_ms(value: str) -> int:
    text = value.strip()
    if text.isdigit():
        return int(text)

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue

    raise ValueError(f"Не удалось распарсить дату: {value}")


def ms_to_utc_text(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fetch_kline_batch(
    session: requests.Session,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> List[List[str]]:
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "end": end_ms,
        "limit": limit,
    }
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit error: {data}")

            rows = data.get("result", {}).get("list", []) or []
            return sorted(rows, key=lambda row: int(row[0]))
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"Не удалось получить kline для {symbol} {interval}: {last_error}")


def download_ohlcv(
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    include_open_candle: bool,
    output_path: str,
) -> int:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Неподдерживаемый interval: {interval}")

    interval_ms = INTERVAL_TO_MS[interval]
    chunk_span_ms = interval_ms * (DEFAULT_LIMIT - 1)
    now_ms = int(time.time() * 1000)
    candles: Dict[int, List[str]] = {}

    session = requests.Session()
    current_start = start_ms

    while current_start <= end_ms:
        current_end = min(end_ms, current_start + chunk_span_ms)
        batch = fetch_kline_batch(
            session=session,
            category=category,
            symbol=symbol,
            interval=interval,
            start_ms=current_start,
            end_ms=current_end,
            limit=DEFAULT_LIMIT,
        )

        for row in batch:
            start_time_ms = int(row[0])
            if not include_open_candle and start_time_ms + interval_ms > now_ms:
                continue
            candles[start_time_ms] = row

        current_start = current_end + interval_ms
        time.sleep(0.05)

    ordered = [candles[key] for key in sorted(candles)]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "start_time_ms",
                "datetime_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ]
        )
        for row in ordered:
            start_time_ms = int(row[0])
            writer.writerow(
                [
                    start_time_ms,
                    ms_to_utc_text(start_time_ms),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                ]
            )

    return len(ordered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default="")
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--include-open-candle", action="store_true")
    args = parser.parse_args()

    start_ms = parse_datetime_to_ms(args.start)
    end_ms = parse_datetime_to_ms(args.end) if args.end else int(time.time() * 1000)

    if end_ms < start_ms:
        raise ValueError("end должен быть >= start")

    os.makedirs(args.output_dir, exist_ok=True)

    for interval in args.intervals:
        output_path = os.path.join(args.output_dir, f"{args.symbol}_{interval}.csv")
        row_count = download_ohlcv(
            category=args.category,
            symbol=args.symbol.upper(),
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            include_open_candle=args.include_open_candle,
            output_path=output_path,
        )
        print(f"{output_path}: {row_count} candles")


if __name__ == "__main__":
    main()
