import argparse
import csv
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


BASE_URL = "https://api.bybit.com"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
DEFAULT_CATEGORY = "linear"
DEFAULT_START = "2025-01-01"
DEFAULT_END = ""
DEFAULT_SYMBOLS = "NEARUSDT"
DEFAULT_OUTPUT_DIR = "perp_features"

PREMIUM_INTERVAL_TO_MS: Dict[str, int] = {
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

OI_INTERVAL_TO_MS: Dict[str, int] = {
    "5min": 300_000,
    "15min": 900_000,
    "30min": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

ACCOUNT_RATIO_INTERVAL_TO_MS = OI_INTERVAL_TO_MS


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


def normalize_symbols(raw: str) -> List[str]:
    seen = set()
    result: List[str] = []
    for part in raw.split(","):
        symbol = part.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def bybit_get(session: requests.Session, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(f"{BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit error {path}: {data}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Не удалось получить {path}: {last_error}")


def get_instrument_info(session: requests.Session, category: str, symbol: str) -> Dict[str, Any]:
    data = bybit_get(
        session,
        "/v5/market/instruments-info",
        {"category": category, "symbol": symbol},
    )
    rows = data.get("result", {}).get("list", []) or []
    if not rows:
        raise RuntimeError(f"Пустой instruments-info для {symbol}")
    return rows[0]


def fetch_funding_history(
    session: requests.Session,
    category: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    funding_interval_minutes: int,
) -> List[Dict[str, Any]]:
    interval_ms = max(funding_interval_minutes, 1) * 60_000
    chunk_ms = interval_ms * 199
    rows_by_ts: Dict[int, Dict[str, Any]] = {}
    cursor_start = start_ms

    while cursor_start <= end_ms:
        cursor_end = min(end_ms, cursor_start + chunk_ms)
        data = bybit_get(
            session,
            "/v5/market/funding/history",
            {
                "category": category,
                "symbol": symbol,
                "startTime": cursor_start,
                "endTime": cursor_end,
                "limit": 200,
            },
        )
        payload = data.get("result", {}).get("list", []) or []
        for row in payload:
            ts = int(row["fundingRateTimestamp"])
            rows_by_ts[ts] = {
                "timestamp_ms": ts,
                "datetime_utc": ms_to_utc_text(ts),
                "funding_rate": row["fundingRate"],
            }
        cursor_start = cursor_end + interval_ms
        time.sleep(0.05)

    return [rows_by_ts[key] for key in sorted(rows_by_ts)]


def fetch_cursor_series(
    session: requests.Session,
    path: str,
    params: Dict[str, Any],
    timestamp_key: str,
    row_mapper,
) -> List[Dict[str, Any]]:
    rows_by_ts: Dict[int, Dict[str, Any]] = {}
    cursor = ""

    while True:
        request_params = dict(params)
        if cursor:
            request_params["cursor"] = cursor
        data = bybit_get(session, path, request_params)
        result = data.get("result", {}) or {}
        payload = result.get("list", []) or []
        for row in payload:
            ts = int(row[timestamp_key])
            rows_by_ts[ts] = row_mapper(row, ts)
        cursor = result.get("nextPageCursor", "") or ""
        if not cursor:
            break
        time.sleep(0.05)

    return [rows_by_ts[key] for key in sorted(rows_by_ts)]


def fetch_open_interest(
    session: requests.Session,
    category: str,
    symbol: str,
    interval_time: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    return fetch_cursor_series(
        session=session,
        path="/v5/market/open-interest",
        params={
            "category": category,
            "symbol": symbol,
            "intervalTime": interval_time,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200,
        },
        timestamp_key="timestamp",
        row_mapper=lambda row, ts: {
            "timestamp_ms": ts,
            "datetime_utc": ms_to_utc_text(ts),
            "open_interest": row["openInterest"],
        },
    )


def fetch_account_ratio(
    session: requests.Session,
    category: str,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    return fetch_cursor_series(
        session=session,
        path="/v5/market/account-ratio",
        params={
            "category": category,
            "symbol": symbol,
            "period": period,
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": 500,
        },
        timestamp_key="timestamp",
        row_mapper=lambda row, ts: {
            "timestamp_ms": ts,
            "datetime_utc": ms_to_utc_text(ts),
            "buy_ratio": row["buyRatio"],
            "sell_ratio": row["sellRatio"],
        },
    )


def fetch_premium_klines(
    session: requests.Session,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    if interval not in PREMIUM_INTERVAL_TO_MS:
        raise ValueError(f"Неподдерживаемый premium interval: {interval}")

    interval_ms = PREMIUM_INTERVAL_TO_MS[interval]
    chunk_ms = interval_ms * 999
    rows_by_ts: Dict[int, Dict[str, Any]] = {}
    cursor_start = start_ms

    while cursor_start <= end_ms:
        cursor_end = min(end_ms, cursor_start + chunk_ms)
        data = bybit_get(
            session,
            "/v5/market/premium-index-price-kline",
            {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": cursor_start,
                "end": cursor_end,
                "limit": 1000,
            },
        )
        payload = data.get("result", {}).get("list", []) or []
        for row in payload:
            ts = int(row[0])
            rows_by_ts[ts] = {
                "start_time_ms": ts,
                "datetime_utc": ms_to_utc_text(ts),
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
            }
        cursor_start = cursor_end + interval_ms
        time.sleep(0.05)

    return [rows_by_ts[key] for key in sorted(rows_by_ts)]


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--oi-interval", default="1h")
    parser.add_argument("--account-ratio-period", default="1h")
    parser.add_argument("--premium-interval", default="60")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    start_ms = parse_datetime_to_ms(args.start)
    end_ms = parse_datetime_to_ms(args.end) if args.end else int(time.time() * 1000)
    if end_ms < start_ms:
        raise ValueError("end должен быть >= start")

    os.makedirs(args.output_dir, exist_ok=True)
    symbols = normalize_symbols(args.symbols)
    if not symbols:
        raise ValueError("Не выбрано ни одного symbol")

    session = requests.Session()

    for symbol in symbols:
        info = get_instrument_info(session, args.category, symbol)
        funding_interval_minutes = int(info.get("fundingInterval") or 480)

        funding_rows = fetch_funding_history(
            session=session,
            category=args.category,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            funding_interval_minutes=funding_interval_minutes,
        )
        funding_path = os.path.join(args.output_dir, f"{symbol}_funding.csv")
        write_csv(
            funding_path,
            ["timestamp_ms", "datetime_utc", "funding_rate"],
            funding_rows,
        )

        oi_rows = fetch_open_interest(
            session=session,
            category=args.category,
            symbol=symbol,
            interval_time=args.oi_interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        oi_path = os.path.join(args.output_dir, f"{symbol}_open_interest_{args.oi_interval}.csv")
        write_csv(
            oi_path,
            ["timestamp_ms", "datetime_utc", "open_interest"],
            oi_rows,
        )

        account_rows = fetch_account_ratio(
            session=session,
            category=args.category,
            symbol=symbol,
            period=args.account_ratio_period,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        account_path = os.path.join(args.output_dir, f"{symbol}_account_ratio_{args.account_ratio_period}.csv")
        write_csv(
            account_path,
            ["timestamp_ms", "datetime_utc", "buy_ratio", "sell_ratio"],
            account_rows,
        )

        premium_rows = fetch_premium_klines(
            session=session,
            category=args.category,
            symbol=symbol,
            interval=args.premium_interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        premium_path = os.path.join(args.output_dir, f"{symbol}_premium_{args.premium_interval}.csv")
        write_csv(
            premium_path,
            ["start_time_ms", "datetime_utc", "open", "high", "low", "close"],
            premium_rows,
        )

        print(
            f"{symbol}: funding={len(funding_rows)} oi={len(oi_rows)} "
            f"account_ratio={len(account_rows)} premium={len(premium_rows)}"
        )


if __name__ == "__main__":
    main()
