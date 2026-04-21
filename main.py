import csv
import hashlib
import hmac
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

try:
    import pandas as pd
    from sklearn.linear_model import LinearRegression
except ImportError:
    pd = None
    LinearRegression = None

# -------------------- Пути и конфиг --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
CONFIG_EXAMPLE_PATH = os.path.join(BASE_DIR, "config.example.json")

load_dotenv(ENV_PATH)


def load_config() -> Tuple[Dict[str, Any], str]:
    for path in (CONFIG_PATH, CONFIG_EXAMPLE_PATH):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
    raise FileNotFoundError(
        "Не найден config.json. Создай его рядом со скриптом, можно начать с config.example.json."
    )


CONFIG, CONFIG_SOURCE = load_config()

MODE = os.getenv("MODE", "DEMO").upper()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

RISK_PER_TRADE = float(CONFIG.get("risk_per_trade", 0.0025))
LEVERAGE = int(CONFIG.get("leverage", 5))
POSITION_IDX = int(CONFIG.get("position_idx", 0))
TIMEFRAME_SIGNAL = str(CONFIG.get("timeframe_signal", "1h"))
TIMEFRAME_TREND = str(CONFIG.get("timeframe_trend", "4h"))
MIN_RR = float(CONFIG.get("min_rr", 3.0))
SYMBOLS = list(CONFIG.get("symbols", []))
TRIGGER_BY = str(CONFIG.get("trigger_by", "LastPrice"))

SL_ATR_BUFFER = float(CONFIG.get("sl_atr_buffer", 0.2))
LIVE_SL_BUFFER = float(CONFIG.get("live_sl_buffer", 0.15))
API_SL_ATR_BUFFER = float(CONFIG.get("api_sl_atr_buffer", 0.3))
MAX_STOP_ATR = float(CONFIG.get("max_stop_atr", 0.7))
MAX_LEVEL_DISTANCE_ATR = float(CONFIG.get("max_level_distance_atr", 0.5))
POSITION_POLL_TIMEOUT = float(CONFIG.get("position_poll_timeout_sec", 10))
POSITION_POLL_INTERVAL = float(CONFIG.get("position_poll_interval_sec", 0.5))
MARGIN_USAGE_BUFFER = float(CONFIG.get("margin_usage_buffer", 0.95))
MAX_DAILY_LOSS_PCT = float(CONFIG.get("max_daily_loss_pct", 0.10))
MAX_CONSECUTIVE_LOSSES = int(CONFIG.get("max_consecutive_losses", 3))
SYMBOL_VALIDATION_ON_STARTUP = bool(CONFIG.get("symbol_validation_on_startup", True))
SLEEP_SECONDS = int(CONFIG.get("sleep_seconds", 900))
DAILY_PNL_SYNC_LIMIT = max(1, min(int(CONFIG.get("daily_pnl_sync_limit", 100)), 100))
SIGNAL_SCORE_THRESHOLD = int(CONFIG.get("signal_score_threshold", 2))
MAX_DIRECTIONAL_RISK_PCT = float(CONFIG.get("max_directional_risk_pct", 0.02))
MAX_TOTAL_OPEN_RISK_PCT = float(CONFIG.get("max_total_open_risk_pct", 0.04))
MAX_POSITIONS_PER_DIRECTION = int(CONFIG.get("max_positions_per_direction", 5))
MIN_LIVE_TRADES_FOR_ADAPTIVE_WEIGHTS = int(CONFIG.get("min_live_trades_for_adaptive_weights", 30))
REQUEST_RECV_WINDOW_MS = int(CONFIG.get("recv_window_ms", 10000))

LOG_PATH = os.path.join(BASE_DIR, "trades.log")
JOURNAL_PATH = os.path.join(BASE_DIR, "journal.csv")
BACKTEST_JOURNAL_PATH = os.path.join(BASE_DIR, "backtest_journal.csv")
ADAPTIVE_STATE_PATH = os.path.join(BASE_DIR, "adaptive_state.json")
MIN_TRADES_FOR_ADAPTIVE = 50
LIVE_TRAINING_WEIGHT = 5.0
BACKTEST_TRAINING_WEIGHT = 1.0
ADAPTIVE_HALF_LIFE_DAYS = float(CONFIG.get("adaptive_half_life_days", 60.0))
FEATURE_COLUMNS = [
    "trend_score",
    "level_score",
    "distance_score",
    "impulse_score",
    "structure_score",
]
JOURNAL_FIELDNAMES = [
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
    "level",
    "impulse_ok",
    "status",
    "realized_pnl",
    "note",
]

if MODE not in {"DEMO", "LIVE"}:
    raise ValueError("MODE должен быть DEMO или LIVE")

BASE_URL = "https://api-demo.bybit.com" if MODE == "DEMO" else "https://api.bybit.com"
REQUEST_RECV_WINDOW = str(REQUEST_RECV_WINDOW_MS)
REQUEST_TIMEOUT = 15

EDGE_WEIGHTS = {
    "trend_alignment": 0.25,
    "level_quality": 0.25,
    "distance_to_level": 0.15,
    "impulse": 0.20,
    "market_structure": 0.15,
}

TIMEFRAME_TO_BYBIT = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}

TIMEFRAME_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}

# -------------------- Логи --------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

if CONFIG_SOURCE.endswith("config.example.json"):
    logger.warning("Используется config.example.json. Для реальной работы создай отдельный config.json.")

# -------------------- Утилиты Decimal --------------------


def decimal_places_from_step(step_str: str) -> int:
    step = Decimal(step_str)
    return max(0, -step.as_tuple().exponent)


def format_decimal_str(value: Decimal, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def parse_decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)


def quantize_to_step(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    if step <= 0:
        return value
    steps = (value / step).to_integral_value(rounding=rounding)
    return steps * step


@dataclass
class DailyRiskState:
    day: str
    gross_closed_pnl: float
    net_closed_pnl: float
    total_fees: float
    trade_count: int
    consecutive_losses: int
    estimated_start_balance: float
    current_wallet_balance: float
    current_available_balance: float
    current_equity: float
    loss_limit_usdt: float
    blocked_reason: Optional[str] = None


# -------------------- Bybit API клиент --------------------


class BybitClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    @staticmethod
    def _clean_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not params:
            return {}
        return {k: v for k, v in params.items() if v is not None}

    @staticmethod
    def _query_items(params: Dict[str, Any]) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for key in sorted(params):
            value = params[key]
            if isinstance(value, bool):
                items.append((key, "true" if value else "false"))
            else:
                items.append((key, str(value)))
        return items

    def _build_auth_headers(self, payload_to_sign: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        raw = f"{timestamp}{self.api_key}{REQUEST_RECV_WINDOW}{payload_to_sign}"
        signature = hmac.new(self.api_secret, raw.encode("utf-8"), hashlib.sha256).hexdigest()
        return {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": REQUEST_RECV_WINDOW,
            "X-BAPI-SIGN": signature,
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        clean_params = self._clean_params(params)

        try:
            if method == "GET":
                query_items = self._query_items(clean_params)
                query_string = urlencode(query_items)
                headers = {"Content-Type": "application/json"}
                if auth:
                    headers = self._build_auth_headers(query_string)
                resp = self.session.get(
                    url,
                    params=query_items,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
            else:
                body = json.dumps(clean_params, separators=(",", ":"), ensure_ascii=False)
                headers = {"Content-Type": "application/json"}
                if auth:
                    headers = self._build_auth_headers(body)
                resp = self.session.post(
                    url,
                    data=body.encode("utf-8"),
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
        except requests.RequestException as exc:
            logger.exception("HTTP error on %s %s: %s", method, path, exc)
            return {"retCode": -1, "retMsg": str(exc), "result": {}}

        try:
            data = resp.json()
        except ValueError:
            logger.error("Non-JSON response on %s %s: status=%s body=%s", method, path, resp.status_code, resp.text)
            return {"retCode": -1, "retMsg": f"HTTP {resp.status_code}", "result": {}}

        if not resp.ok:
            logger.warning("HTTP %s on %s: %s", resp.status_code, path, data)
            return data

        if data.get("retCode") != 0:
            logger.warning("Bybit API error on %s: %s", path, data)

        return data

    # ---- Маркет данные ----
    def get_kline(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        closed_only: bool = False,
    ) -> List[List[Any]]:
        bybit_interval = TIMEFRAME_TO_BYBIT.get(interval, interval)
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": limit,
        }
        data = self._request("GET", "/v5/market/kline", params)
        klines = data.get("result", {}).get("list", []) or []
        ordered = sorted(klines, key=lambda row: int(row[0]))
        if closed_only:
            return filter_closed_klines(ordered, interval)
        return ordered

    def get_last_price(self, symbol: str) -> float:
        params = {"category": "linear", "symbol": symbol}
        data = self._request("GET", "/v5/market/tickers", params)
        items = data.get("result", {}).get("list", []) or []
        if not items:
            return 0.0
        return float(items[0].get("lastPrice", 0.0) or 0.0)

    def get_instrument_info(self, symbol: str) -> Dict[str, Any]:
        params = {"category": "linear", "symbol": symbol}
        data = self._request("GET", "/v5/market/instruments-info", params)
        items = data.get("result", {}).get("list", []) or []
        return items[0] if items else {}

    # ---- Аккаунт / баланс ----
    def get_um_wallet_balance(self, coin: str = "USDT") -> float:
        return float(self.get_usdt_balance_info(coin).get("walletBalance", Decimal("0")))

    def get_usdt_balance_info(self, coin: str = "USDT") -> Dict[str, Decimal]:
        params = {"accountType": "UNIFIED", "coin": coin}
        data = self._request("GET", "/v5/account/wallet-balance", params, auth=True)
        items = data.get("result", {}).get("list", []) or []
        for account in items:
            account_available = parse_decimal(account.get("totalAvailableBalance"))
            total_wallet_balance = parse_decimal(account.get("totalWalletBalance"))
            total_equity = parse_decimal(account.get("totalEquity"))
            for item in account.get("coin", []) or []:
                if item.get("coin") == coin:
                    return {
                        "equity": parse_decimal(item.get("equity")),
                        "walletBalance": parse_decimal(item.get("walletBalance")),
                        "availableBalance": account_available,
                        "totalWalletBalance": total_wallet_balance,
                        "totalEquity": total_equity,
                    }
        return {
            "equity": Decimal("0"),
            "walletBalance": Decimal("0"),
            "availableBalance": Decimal("0"),
            "totalWalletBalance": Decimal("0"),
            "totalEquity": Decimal("0"),
        }

    def get_closed_pnl(
        self,
        start_time_ms: int,
        end_time_ms: int,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        while len(rows) < limit:
            page_limit = min(100, limit - len(rows))
            params = {
                "category": "linear",
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": page_limit,
                "cursor": cursor,
                "symbol": symbol,
            }
            data = self._request("GET", "/v5/position/closed-pnl", params, auth=True)
            result = data.get("result", {}) or {}
            page_rows = result.get("list", []) or []
            rows.extend(page_rows)
            cursor = result.get("nextPageCursor")
            if not page_rows or not cursor:
                break

        return rows[:limit]

    # ---- Ордеры и позиции ----
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        params = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        return self._request("POST", "/v5/position/set-leverage", params, auth=True)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        order_type: str = "Market",
        entry_price: Optional[str] = None,
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        position_idx: Optional[int] = None,
        order_link_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "positionIdx": position_idx,
            "orderLinkId": order_link_id,
        }

        if order_type == "Limit":
            params["price"] = entry_price
            params["timeInForce"] = "GTC"

        if reduce_only:
            params["reduceOnly"] = True
        if close_on_trigger:
            params["closeOnTrigger"] = True

        return self._request("POST", "/v5/order/create", params, auth=True)

    def close_position_market(
        self,
        symbol: str,
        open_side: str,
        qty: str,
        position_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        close_side = "Sell" if open_side == "Buy" else "Buy"
        return self.place_order(
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_type="Market",
            reduce_only=True,
            close_on_trigger=True,
            position_idx=position_idx,
            order_link_id=build_order_link_id(symbol, close_side, "close"),
        )

    def get_position(self, symbol: str, side: Optional[str] = None) -> Optional[Dict[str, Any]]:
        params = {"category": "linear", "symbol": symbol, "settleCoin": "USDT"}
        data = self._request("GET", "/v5/position/list", params, auth=True)
        items = data.get("result", {}).get("list", []) or []
        if not items:
            return None

        non_empty = []
        for item in items:
            size = parse_decimal(item.get("size", "0"))
            pos_side = str(item.get("side", "") or "")
            if size > 0 and pos_side in {"Buy", "Sell"}:
                non_empty.append(item)

        if side:
            for item in non_empty:
                if item.get("side") == side:
                    return item
            return None

        return non_empty[0] if non_empty else None

    def set_trading_stop(
        self,
        symbol: str,
        position_idx: int,
        stop_loss: str,
        take_profit: str,
        trigger_by: str = "LastPrice",
    ) -> Dict[str, Any]:
        params = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx,
            "tpslMode": "Full",
            "stopLoss": stop_loss,
            "takeProfit": take_profit,
            "slTriggerBy": trigger_by,
            "tpTriggerBy": trigger_by,
        }
        return self._request("POST", "/v5/position/trading-stop", params, auth=True)


# -------------------- Параметры инструмента --------------------

INSTRUMENT_CACHE: Dict[str, Dict[str, Any]] = {}


def get_symbol_constraints(client: BybitClient, symbol: str) -> Optional[Dict[str, Any]]:
    if symbol in INSTRUMENT_CACHE:
        return INSTRUMENT_CACHE[symbol]

    info = client.get_instrument_info(symbol)
    if not info:
        logger.warning("%s: не удалось получить instruments-info", symbol)
        return None

    lot_size = info.get("lotSizeFilter", {}) or {}
    price_filter = info.get("priceFilter", {}) or {}

    min_qty = parse_decimal(lot_size.get("minOrderQty"))
    qty_step = parse_decimal(lot_size.get("qtyStep"))
    max_order_qty = parse_decimal(lot_size.get("maxOrderQty"))
    max_market_qty = parse_decimal(lot_size.get("maxMktOrderQty"), str(max_order_qty))
    min_notional = parse_decimal(lot_size.get("minNotionalValue"))

    tick_size = parse_decimal(price_filter.get("tickSize"))
    min_price = parse_decimal(price_filter.get("minPrice"))
    max_price = parse_decimal(price_filter.get("maxPrice"))

    if min_qty <= 0 or qty_step <= 0 or tick_size <= 0:
        logger.warning("%s: некорректные фильтры инструмента: %s", symbol, info)
        return None

    constraints = {
        "minOrderQty": min_qty,
        "qtyStep": qty_step,
        "maxOrderQty": max_order_qty,
        "maxMktOrderQty": max_market_qty,
        "minNotionalValue": min_notional,
        "tickSize": tick_size,
        "minPrice": min_price,
        "maxPrice": max_price,
        "qtyDecimals": decimal_places_from_step(str(qty_step)),
        "priceDecimals": decimal_places_from_step(str(tick_size)),
    }
    INSTRUMENT_CACHE[symbol] = constraints

    logger.info(
        "%s: qtyStep=%s, minQty=%s, maxMktQty=%s, tickSize=%s",
        symbol,
        format_decimal_str(qty_step, constraints["qtyDecimals"]),
        format_decimal_str(min_qty, constraints["qtyDecimals"]),
        format_decimal_str(max_market_qty, constraints["qtyDecimals"]),
        format_decimal_str(tick_size, constraints["priceDecimals"]),
    )
    return constraints


def normalize_qty(qty_raw: float, constraints: Dict[str, Any], order_type: str = "Market") -> Tuple[Decimal, str]:
    raw = Decimal(str(qty_raw))
    if raw <= 0:
        return Decimal("0"), "0"

    min_qty: Decimal = constraints["minOrderQty"]
    qty_step: Decimal = constraints["qtyStep"]
    max_qty: Decimal = constraints["maxMktOrderQty"] if order_type == "Market" else constraints["maxOrderQty"]
    decimals: int = constraints["qtyDecimals"]

    normalized = quantize_to_step(raw, qty_step, ROUND_DOWN)
    if max_qty > 0 and normalized > max_qty:
        normalized = quantize_to_step(max_qty, qty_step, ROUND_DOWN)
    if normalized < min_qty:
        return Decimal("0"), "0"
    return normalized, format_decimal_str(normalized, decimals)


def normalize_price(price_raw: float, constraints: Dict[str, Any], rounding: str) -> Tuple[Decimal, str]:
    raw = Decimal(str(price_raw))
    if raw <= 0:
        return Decimal("0"), "0"

    tick_size: Decimal = constraints["tickSize"]
    min_price: Decimal = constraints["minPrice"]
    max_price: Decimal = constraints["maxPrice"]
    decimals: int = constraints["priceDecimals"]

    normalized = quantize_to_step(raw, tick_size, rounding)
    if min_price > 0 and normalized < min_price:
        normalized = min_price
    if max_price > 0 and normalized > max_price:
        normalized = max_price

    return normalized, format_decimal_str(normalized, decimals)


# -------------------- Утилиты свечей --------------------


def is_kline_closed(kline: List[Any], interval: str) -> bool:
    interval_ms = TIMEFRAME_TO_MS.get(interval)
    if interval_ms is None:
        return True
    open_time = int(kline[0])
    now_ms = int(time.time() * 1000)
    return open_time + interval_ms <= now_ms


def filter_closed_klines(klines: List[List[Any]], interval: str) -> List[List[Any]]:
    return [k for k in klines if is_kline_closed(k, interval)]


# -------------------- Риск и ATR --------------------


def calc_atr_from_klines(klines: List[List[Any]], period: int = 14) -> float:
    recent = klines[-(period + 1) :]
    if len(recent) < period + 1:
        return 0.0

    trs: List[float] = []
    prev_close = float(recent[0][4])
    for candle in recent[1:]:
        high = float(candle[2])
        low = float(candle[3])
        close = float(candle[4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close

    return sum(trs) / len(trs) if trs else 0.0


def calc_qty_by_risk(balance: float, risk_per_trade: float, entry: float, stop: float) -> float:
    stop_size = abs(entry - stop)
    if balance <= 0 or stop_size <= 0:
        return 0.0
    return max((balance * risk_per_trade) / stop_size, 0.0)


def calc_qty_by_margin(balance: float, leverage: int, entry: float) -> float:
    if balance <= 0 or leverage <= 0 or entry <= 0:
        return 0.0
    return max((balance * leverage * MARGIN_USAGE_BUFFER) / entry, 0.0)


# -------------------- Журнал сделок --------------------


def append_journal_row(row: Dict[str, Any]) -> None:
    ensure_journal_schema()
    file_exists = os.path.exists(JOURNAL_PATH)

    with open(JOURNAL_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def count_opened_trades_for_day(day_str: str) -> int:
    if not os.path.exists(JOURNAL_PATH):
        return 0

    count = 0
    with open(JOURNAL_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = str(row.get("timestamp", ""))
            status = str(row.get("status", ""))
            if timestamp.startswith(day_str) and status == "opened_with_sl_tp":
                count += 1
    return count


def count_closed_trades() -> int:
    if not os.path.exists(JOURNAL_PATH):
        return 0

    count = 0
    with open(JOURNAL_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("realized_pnl", "")).strip():
                count += 1
    return count


def get_open_portfolio_risk_state() -> Dict[str, Any]:
    state = {
        "total_risk_usdt": 0.0,
        "directional_risk": {"Buy": 0.0, "Sell": 0.0},
        "directional_count": {"Buy": 0, "Sell": 0},
    }

    if not os.path.exists(JOURNAL_PATH):
        return state

    with open(JOURNAL_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")) != "opened_with_sl_tp":
                continue
            if str(row.get("realized_pnl", "")).strip():
                continue

            side = str(row.get("side", "")).strip()
            if side not in {"Buy", "Sell"}:
                continue

            try:
                risk_usdt = float(row.get("risk_usdt", 0.0) or 0.0)
            except (TypeError, ValueError):
                risk_usdt = 0.0

            if risk_usdt < 0:
                risk_usdt = 0.0

            state["total_risk_usdt"] += risk_usdt
            state["directional_risk"][side] += risk_usdt
            state["directional_count"][side] += 1

    return state


def ensure_journal_schema() -> None:
    if not os.path.exists(JOURNAL_PATH):
        return

    with open(JOURNAL_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_fieldnames = reader.fieldnames or []
        rows = list(reader)

    if current_fieldnames == JOURNAL_FIELDNAMES:
        return

    with open(JOURNAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in JOURNAL_FIELDNAMES})


def parse_note_payload(note: Any) -> Dict[str, Any]:
    if not isinstance(note, str) or not note.strip():
        return {}
    try:
        payload = json.loads(note)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_journal_df(path: str = JOURNAL_PATH):
    if pd is None or not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if "note" in df.columns:
        meta = df["note"].apply(parse_note_payload)
        meta_df = pd.json_normalize(meta)
        df = pd.concat([df, meta_df], axis=1)
    return df


def compute_journal_r(df):
    if pd is None or df is None or "realized_pnl" not in df.columns:
        return None

    model_df = df.copy()
    model_df["realized_pnl"] = pd.to_numeric(model_df["realized_pnl"], errors="coerce")
    model_df["risk_usdt"] = pd.to_numeric(model_df["risk_usdt"], errors="coerce")
    model_df = model_df.dropna(subset=["realized_pnl", "risk_usdt"])
    model_df = model_df[model_df["risk_usdt"] > 0]
    if len(model_df) == 0:
        return model_df

    model_df["R"] = model_df["realized_pnl"] / model_df["risk_usdt"]
    return model_df


def load_journal_rows(path: str = JOURNAL_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            payload = dict(row)
            payload.update(parse_note_payload(row.get("note")))
            rows.append(payload)
    return rows


def compute_journal_r_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for row in rows:
        try:
            realized_pnl = float(row.get("realized_pnl", ""))
            risk_usdt = float(row.get("risk_usdt", ""))
        except (TypeError, ValueError):
            continue

        if risk_usdt <= 0:
            continue

        payload = dict(row)
        payload["realized_pnl"] = realized_pnl
        payload["risk_usdt"] = risk_usdt
        payload["R"] = realized_pnl / risk_usdt
        processed.append(payload)

    return processed


def load_adaptive_training_df():
    if pd is None:
        return None, {"live": 0, "backtest": 0}

    frames = []
    stats = {"live": 0, "backtest": 0}

    live_df = compute_journal_r(load_journal_df(JOURNAL_PATH))
    if live_df is not None and len(live_df) > 0:
        stats["live"] = int(len(live_df))
        live_df = live_df.copy()
        live_df["training_source"] = "live"
        frames.append(live_df)

    backtest_df = compute_journal_r(load_journal_df(BACKTEST_JOURNAL_PATH))
    if backtest_df is not None and len(backtest_df) > 0:
        stats["backtest"] = int(len(backtest_df))
        backtest_df = backtest_df.copy()
        backtest_df["training_source"] = "backtest"
        frames.append(backtest_df)

    if not frames:
        return None, stats

    return pd.concat(frames, ignore_index=True), stats


def load_adaptive_training_rows() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats = {"live": 0, "backtest": 0}

    live_rows = compute_journal_r_rows(load_journal_rows(JOURNAL_PATH))
    if live_rows:
        stats["live"] = len(live_rows)
        for row in live_rows:
            payload = dict(row)
            payload["training_source"] = "live"
            rows.append(payload)

    backtest_rows = compute_journal_r_rows(load_journal_rows(BACKTEST_JOURNAL_PATH))
    if backtest_rows:
        stats["backtest"] = len(backtest_rows)
        for row in backtest_rows:
            payload = dict(row)
            payload["training_source"] = "backtest"
            rows.append(payload)

    return rows, stats


def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> Optional[List[float]]:
    size = len(vector)
    augmented = [list(matrix[row_idx]) + [vector[row_idx]] for row_idx in range(size)]

    for pivot_idx in range(size):
        pivot_row = max(range(pivot_idx, size), key=lambda row_idx: abs(augmented[row_idx][pivot_idx]))
        pivot_value = augmented[pivot_row][pivot_idx]
        if abs(pivot_value) < 1e-12:
            return None

        if pivot_row != pivot_idx:
            augmented[pivot_idx], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_idx]

        pivot_value = augmented[pivot_idx][pivot_idx]
        for col_idx in range(pivot_idx, size + 1):
            augmented[pivot_idx][col_idx] /= pivot_value

        for row_idx in range(size):
            if row_idx == pivot_idx:
                continue
            factor = augmented[row_idx][pivot_idx]
            for col_idx in range(pivot_idx, size + 1):
                augmented[row_idx][col_idx] -= factor * augmented[pivot_idx][col_idx]

    return [augmented[row_idx][size] for row_idx in range(size)]


def get_time_decay_weight(timestamp_value: Any) -> float:
    if ADAPTIVE_HALF_LIFE_DAYS <= 0:
        return 1.0

    timestamp_text = str(timestamp_value or "").strip()
    if not timestamp_text:
        return 1.0

    try:
        trade_time = datetime.fromisoformat(timestamp_text)
    except ValueError:
        return 1.0

    if trade_time.tzinfo is None:
        trade_time = trade_time.replace(tzinfo=timezone.utc)

    age_seconds = max((datetime.now(timezone.utc) - trade_time).total_seconds(), 0.0)
    age_days = age_seconds / 86400.0
    return math.pow(0.5, age_days / ADAPTIVE_HALF_LIFE_DAYS)


def get_training_row_weight(training_source: Any, timestamp_value: Any = None) -> float:
    source_weight = BACKTEST_TRAINING_WEIGHT
    if str(training_source).strip().lower() == "live":
        source_weight = LIVE_TRAINING_WEIGHT
    return source_weight * get_time_decay_weight(timestamp_value)


def fit_adaptive_weights(df) -> Optional[Dict[str, float]]:
    if pd is None or LinearRegression is None or df is None:
        return None

    model_df = df.copy()
    for column in FEATURE_COLUMNS + ["R"]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df.dropna(subset=FEATURE_COLUMNS + ["R"])
    if len(model_df) < MIN_TRADES_FOR_ADAPTIVE:
        return None

    x_values = model_df[FEATURE_COLUMNS]
    y_values = model_df["R"]
    sample_weights = model_df.apply(
        lambda row: get_training_row_weight(row.get("training_source"), row.get("timestamp")),
        axis=1,
    )

    model = LinearRegression()
    model.fit(x_values, y_values, sample_weight=sample_weights)

    weights = dict(zip(FEATURE_COLUMNS, model.coef_))
    total = sum(abs(value) for value in weights.values())
    if total <= 0:
        return None

    return {key: round(value / total, 6) for key, value in weights.items()}


def fit_adaptive_weights_rows(rows: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if len(rows) < MIN_TRADES_FOR_ADAPTIVE:
        return None

    dataset: List[Tuple[List[float], float, float]] = []
    for row in rows:
        try:
            features = [float(row[column]) for column in FEATURE_COLUMNS]
            result = float(row["R"])
        except (KeyError, TypeError, ValueError):
            continue
        weight = get_training_row_weight(row.get("training_source"), row.get("timestamp"))
        dataset.append((features, result, weight))

    if len(dataset) < MIN_TRADES_FOR_ADAPTIVE:
        return None

    feature_count = len(FEATURE_COLUMNS)
    total_weight = sum(item[2] for item in dataset)
    if total_weight <= 0:
        return None

    x_means = [
        sum(item[0][idx] * item[2] for item in dataset) / total_weight for idx in range(feature_count)
    ]
    y_mean = sum(item[1] * item[2] for item in dataset) / total_weight

    xtx = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    xty = [0.0 for _ in range(feature_count)]

    for features, result, weight in dataset:
        centered_x = [features[idx] - x_means[idx] for idx in range(feature_count)]
        centered_y = result - y_mean
        for i in range(feature_count):
            xty[i] += weight * centered_x[i] * centered_y
            for j in range(feature_count):
                xtx[i][j] += weight * centered_x[i] * centered_x[j]

    for idx in range(feature_count):
        xtx[idx][idx] += 1e-9

    coefficients = solve_linear_system(xtx, xty)
    if coefficients is None:
        return None

    weights = dict(zip(FEATURE_COLUMNS, coefficients))
    total = sum(abs(value) for value in weights.values())
    if total <= 0:
        return None

    return {key: round(value / total, 6) for key, value in weights.items()}


def find_best_adaptive_threshold(df) -> float:
    if pd is None or df is None:
        return SIGNAL_SCORE_THRESHOLD / 3.0

    threshold_df = df.copy()
    threshold_df["edge_score"] = pd.to_numeric(threshold_df["edge_score"], errors="coerce")
    threshold_df = threshold_df.dropna(subset=["edge_score", "R"])

    best_threshold = SIGNAL_SCORE_THRESHOLD / 3.0
    best_expectancy = float("-inf")
    threshold_df["training_weight"] = threshold_df.apply(
        lambda row: get_training_row_weight(row.get("training_source"), row.get("timestamp")),
        axis=1,
    )

    for threshold in [i / 100 for i in range(30, 90, 5)]:
        subset = threshold_df[threshold_df["edge_score"] >= threshold]
        if len(subset) < 30:
            continue

        weight_sum = float(subset["training_weight"].sum())
        if weight_sum <= 0:
            continue

        expectancy = float((subset["R"] * subset["training_weight"]).sum() / weight_sum)
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_threshold = threshold

    return round(best_threshold, 6)


def find_best_adaptive_threshold_rows(rows: List[Dict[str, Any]]) -> float:
    best_threshold = SIGNAL_SCORE_THRESHOLD / 3.0
    best_expectancy = float("-inf")

    for threshold in [i / 100 for i in range(30, 90, 5)]:
        weighted_results: List[Tuple[float, float]] = []
        for row in rows:
            try:
                edge_score = float(row["edge_score"])
                result = float(row["R"])
            except (KeyError, TypeError, ValueError):
                continue
            if edge_score >= threshold:
                weighted_results.append(
                    (result, get_training_row_weight(row.get("training_source"), row.get("timestamp")))
                )

        if len(weighted_results) < 30:
            continue

        total_weight = sum(item[1] for item in weighted_results)
        if total_weight <= 0:
            continue

        expectancy = sum(result * weight for result, weight in weighted_results) / total_weight
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_threshold = threshold

    return round(best_threshold, 6)


def load_adaptive_state() -> Dict[str, Any]:
    if not os.path.exists(ADAPTIVE_STATE_PATH):
        return {}

    try:
        with open(ADAPTIVE_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return {}

    return state if isinstance(state, dict) else {}


def save_adaptive_state(state: Dict[str, Any]) -> None:
    with open(ADAPTIVE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)


def update_adaptive_model() -> Optional[Dict[str, Any]]:
    if pd is not None and LinearRegression is not None:
        df, training_stats = load_adaptive_training_df()
        if df is None or len(df) < MIN_TRADES_FOR_ADAPTIVE:
            return None

        weights = fit_adaptive_weights(df)
        if weights is None:
            return None

        state = {
            "weights": weights,
            "threshold": find_best_adaptive_threshold(df),
            "trade_count": int(len(df)),
            "live_trade_count": training_stats["live"],
            "backtest_trade_count": training_stats["backtest"],
            "updated_at": int(time.time()),
        }
    else:
        rows, training_stats = load_adaptive_training_rows()
        if len(rows) < MIN_TRADES_FOR_ADAPTIVE:
            return None

        weights = fit_adaptive_weights_rows(rows)
        if weights is None:
            return None

        state = {
            "weights": weights,
            "threshold": find_best_adaptive_threshold_rows(rows),
            "trade_count": int(len(rows)),
            "live_trade_count": training_stats["live"],
            "backtest_trade_count": training_stats["backtest"],
            "updated_at": int(time.time()),
        }

    current_state = load_adaptive_state()
    if (
        current_state.get("weights") == state["weights"]
        and current_state.get("threshold") == state["threshold"]
        and current_state.get("trade_count") == state["trade_count"]
        and current_state.get("live_trade_count") == state["live_trade_count"]
        and current_state.get("backtest_trade_count") == state["backtest_trade_count"]
    ):
        return None

    save_adaptive_state(state)
    return state


def get_adaptive_weights() -> Dict[str, float]:
    state = load_adaptive_state()
    try:
        live_trade_count = int(state.get("live_trade_count", 0))
    except (TypeError, ValueError):
        live_trade_count = 0
    if live_trade_count < MIN_LIVE_TRADES_FOR_ADAPTIVE_WEIGHTS:
        return {}

    weights = state.get("weights", {})
    if not isinstance(weights, dict):
        return {}

    adaptive_weights: Dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        try:
            adaptive_weights[column] = float(weights[column])
        except (KeyError, TypeError, ValueError):
            return {}
    return adaptive_weights


def get_adaptive_threshold(default_threshold: float) -> float:
    state = load_adaptive_state()
    try:
        threshold = float(state.get("threshold", default_threshold))
    except (TypeError, ValueError):
        return default_threshold

    if threshold < 0 or threshold > 1:
        return default_threshold
    return threshold


def journal_timestamp_to_ms(timestamp_str: str) -> int:
    try:
        return int(datetime.fromisoformat(timestamp_str).timestamp() * 1000)
    except ValueError:
        return 0


def sync_closed_trades_to_journal(client: BybitClient) -> int:
    ensure_journal_schema()
    if not os.path.exists(JOURNAL_PATH):
        return 0

    with open(JOURNAL_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pending_rows = []
    start_time_ms = 0

    for idx, row in enumerate(rows):
        if str(row.get("status", "")) != "opened_with_sl_tp":
            continue
        if str(row.get("realized_pnl", "")).strip():
            continue

        open_time_ms = journal_timestamp_to_ms(str(row.get("timestamp", "")))
        if open_time_ms <= 0:
            continue

        note_payload = parse_note_payload(row.get("note"))
        entry = float(row.get("entry", 0.0) or 0.0)

        pending_rows.append(
            {
                "row_idx": idx,
                "symbol": str(row.get("symbol", "")),
                "side": str(row.get("side", "")),
                "entry": entry,
                "open_time_ms": open_time_ms,
                "note_payload": note_payload,
            }
        )
        if start_time_ms == 0 or open_time_ms < start_time_ms:
            start_time_ms = open_time_ms

    if not pending_rows or start_time_ms <= 0:
        return 0

    closed_rows = client.get_closed_pnl(start_time_ms, int(time.time() * 1000), limit=DAILY_PNL_SYNC_LIMIT)
    closed_rows = sorted(
        closed_rows,
        key=lambda row: int(row.get("updatedTime") or row.get("createdTime") or 0),
    )

    used_closed_indexes = set()
    updated_rows = 0

    for pending in pending_rows:
        matched_index = None

        for idx, closed in enumerate(closed_rows):
            if idx in used_closed_indexes:
                continue
            if str(closed.get("symbol", "")) != pending["symbol"]:
                continue

            close_side = str(closed.get("side", ""))
            if pending["side"] == "Buy" and close_side != "Sell":
                continue
            if pending["side"] == "Sell" and close_side != "Buy":
                continue

            close_time_ms = int(closed.get("updatedTime") or closed.get("createdTime") or 0)
            if close_time_ms < pending["open_time_ms"]:
                continue

            avg_entry_price = float(closed.get("avgEntryPrice", 0.0) or 0.0)
            entry_tolerance = max(pending["entry"] * 0.003, 0.0001) if pending["entry"] > 0 else 0.0001
            if avg_entry_price > 0 and pending["entry"] > 0 and abs(avg_entry_price - pending["entry"]) > entry_tolerance:
                continue

            matched_index = idx
            break

        if matched_index is None:
            continue

        used_closed_indexes.add(matched_index)
        closed = closed_rows[matched_index]
        realized_pnl = closed_pnl_net_value(closed)

        row = rows[pending["row_idx"]]
        row["realized_pnl"] = f"{realized_pnl:.8f}"

        note_payload = pending["note_payload"]
        note_payload.update(
            {
                "closed_pnl": float(closed.get("closedPnl", 0.0) or 0.0),
                "open_fee": float(closed.get("openFee", 0.0) or 0.0),
                "close_fee": float(closed.get("closeFee", 0.0) or 0.0),
                "close_time_ms": int(closed.get("updatedTime") or closed.get("createdTime") or 0),
                "close_order_id": str(closed.get("orderId", "") or ""),
            }
        )
        row["note"] = json.dumps(note_payload, ensure_ascii=False, sort_keys=True)
        updated_rows += 1

    if updated_rows == 0:
        return 0

    with open(JOURNAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in JOURNAL_FIELDNAMES})

    return updated_rows


def normalize_symbols(symbols: List[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()

    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)

    return normalized


def validate_symbols(client: BybitClient, symbols: List[str]) -> List[str]:
    valid: List[str] = []
    invalid: List[str] = []

    for symbol in normalize_symbols(symbols):
        info = client.get_instrument_info(symbol)
        status = str(info.get("status", "") or "").upper()
        if not info:
            invalid.append(symbol)
            continue
        if status and status != "TRADING":
            logger.warning("%s: статус инструмента %s, пропускаем", symbol, status)
            invalid.append(symbol)
            continue
        constraints = get_symbol_constraints(client, symbol)
        if constraints:
            valid.append(symbol)
        else:
            invalid.append(symbol)

    if invalid:
        logger.warning("Пропускаем невалидные/недоступные symbols: %s", ", ".join(invalid))
    logger.info("После валидации symbols: %s из %s", len(valid), len(normalize_symbols(symbols)))

    return valid


def utc_day_bounds_ms(day_str: str) -> Tuple[int, int]:
    day_start = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    day_end = day_start.replace(hour=23, minute=59, second=59, microsecond=999000)
    return int(day_start.timestamp() * 1000), int(day_end.timestamp() * 1000)


def closed_pnl_net_value(row: Dict[str, Any]) -> float:
    gross = float(row.get("closedPnl", 0.0) or 0.0)
    open_fee = float(row.get("openFee", 0.0) or 0.0)
    close_fee = float(row.get("closeFee", 0.0) or 0.0)
    return gross - open_fee - close_fee


def sync_daily_risk_state(client: BybitClient, day_str: str) -> DailyRiskState:
    start_ms, end_ms = utc_day_bounds_ms(day_str)
    rows = client.get_closed_pnl(start_ms, end_ms, limit=DAILY_PNL_SYNC_LIMIT)
    sorted_rows = sorted(
        rows,
        key=lambda row: int(row.get("updatedTime") or row.get("createdTime") or 0),
    )

    gross_closed_pnl = 0.0
    total_fees = 0.0
    net_closed_pnl = 0.0

    for row in sorted_rows:
        gross = float(row.get("closedPnl", 0.0) or 0.0)
        fees = float(row.get("openFee", 0.0) or 0.0) + float(row.get("closeFee", 0.0) or 0.0)
        gross_closed_pnl += gross
        total_fees += fees
        net_closed_pnl += gross - fees

    consecutive_losses = 0
    for row in reversed(sorted_rows):
        if closed_pnl_net_value(row) < 0:
            consecutive_losses += 1
        else:
            break

    total_closed_trades = count_closed_trades()
    ignore_losses_stop = total_closed_trades < 30

    balance_info = client.get_usdt_balance_info("USDT")
    current_wallet_balance = float(balance_info.get("walletBalance", Decimal("0")))
    current_available_balance = float(balance_info.get("availableBalance", Decimal("0")))
    current_equity = float(balance_info.get("equity", Decimal("0")))

    estimated_start_balance = current_wallet_balance - net_closed_pnl
    if estimated_start_balance <= 0:
        positive_candidates = [
            value
            for value in (current_wallet_balance, current_available_balance, current_equity)
            if value > 0
        ]
        estimated_start_balance = max(positive_candidates) if positive_candidates else 0.0

    loss_limit_usdt = estimated_start_balance * MAX_DAILY_LOSS_PCT if MAX_DAILY_LOSS_PCT > 0 else 0.0

    blocked_reason = None
    if loss_limit_usdt > 0 and net_closed_pnl <= -loss_limit_usdt:
        blocked_reason = (
            f"дневной лимит убытка достигнут: {net_closed_pnl:.2f} USDT <= -{loss_limit_usdt:.2f} USDT"
        )
    elif (
        not ignore_losses_stop
        and MAX_CONSECUTIVE_LOSSES > 0
        and consecutive_losses >= MAX_CONSECUTIVE_LOSSES
    ):
        blocked_reason = (
            f"достигнут лимит подряд убыточных закрытий: {consecutive_losses} >= {MAX_CONSECUTIVE_LOSSES}"
        )

    return DailyRiskState(
        day=day_str,
        gross_closed_pnl=gross_closed_pnl,
        net_closed_pnl=net_closed_pnl,
        total_fees=total_fees,
        trade_count=len(sorted_rows),
        consecutive_losses=consecutive_losses,
        estimated_start_balance=estimated_start_balance,
        current_wallet_balance=current_wallet_balance,
        current_available_balance=current_available_balance,
        current_equity=current_equity,
        loss_limit_usdt=loss_limit_usdt,
        blocked_reason=blocked_reason,
    )


# -------------------- Утилиты для импульса --------------------


def candle_range(kline: List[Any]) -> float:
    return abs(float(kline[2]) - float(kline[3]))


def strong_close_in_direction(kline: List[Any], side: str) -> bool:
    open_price = float(kline[1])
    high = float(kline[2])
    low = float(kline[3])
    close = float(kline[4])

    candle_size = high - low
    if candle_size <= 0:
        return False

    if side == "Buy":
        return close > open_price and ((high - close) / candle_size) <= 0.25
    if side == "Sell":
        return close < open_price and ((close - low) / candle_size) <= 0.25
    return False


def impulse_filter_ok(klines: List[List[Any]], entry_idx: int, side: str) -> bool:
    if len(klines) < 12 or entry_idx < 10:
        return True

    signal_bar = klines[entry_idx]
    signal_range = candle_range(signal_bar)
    previous_ranges = [candle_range(k) for k in klines[entry_idx - 10 : entry_idx]]
    avg_range = sum(previous_ranges) / len(previous_ranges) if previous_ranges else 0.0

    if avg_range <= 0:
        return True

    return signal_range >= avg_range * 0.9 or strong_close_in_direction(signal_bar, side)


# -------------------- Логика уровней и тренда --------------------


def detect_trend_4h(klines_4h: List[List[Any]]) -> str:
    if len(klines_4h) < 30:
        return "FLAT"

    closes = [float(k[4]) for k in klines_4h[-30:]]
    avg_close = sum(closes) / len(closes)
    last_close = closes[-1]
    threshold = (max(closes) - min(closes)) * 0.1

    if last_close > avg_close + threshold:
        return "UP"
    if last_close < avg_close - threshold:
        return "DOWN"
    return "FLAT"


def find_level_break_trend(klines_4h: List[List[Any]]) -> Optional[Tuple[float, float]]:
    if len(klines_4h) < 5:
        return None

    last_five = klines_4h[-5:]
    lows = [float(k[3]) for k in last_five]
    highs = [float(k[2]) for k in last_five]
    return min(lows), max(highs)


def is_range_dirty_around_level(klines_4h: List[List[Any]], level: float, lookback: int = 20) -> bool:
    if len(klines_4h) < lookback:
        return False

    recent = klines_4h[-lookback:]
    crosses = 0
    prev_pos: Optional[int] = None

    for candle in recent:
        high = float(candle[2])
        low = float(candle[3])

        if high < level:
            pos = -1
        elif low > level:
            pos = 1
        else:
            pos = 0

        if prev_pos is not None and pos != prev_pos:
            crosses += 1
        prev_pos = pos

    return crosses >= 6


# -------------------- Вспомогательные утилиты торговли --------------------


def build_order_link_id(symbol: str, side: str, suffix: str) -> str:
    base = f"{symbol.lower()}-{side.lower()}-{suffix}-{int(time.time() * 1000)}"
    return base[:36]


def wait_for_position(client: BybitClient, symbol: str, side: str) -> Optional[Dict[str, Any]]:
    deadline = time.time() + POSITION_POLL_TIMEOUT
    while time.time() < deadline:
        pos = client.get_position(symbol, side=side)
        if pos is not None:
            return pos
        time.sleep(POSITION_POLL_INTERVAL)
    return None


def force_close_position(
    client: BybitClient,
    symbol: str,
    side: str,
    fallback_qty: str,
    fallback_position_idx: int,
) -> Dict[str, Any]:
    pos = client.get_position(symbol, side=side)
    if pos:
        qty = str(pos.get("size") or fallback_qty)
        position_idx = int(pos.get("positionIdx", fallback_position_idx))
    else:
        qty = fallback_qty
        position_idx = fallback_position_idx
    return client.close_position_market(symbol, side, qty, position_idx)


def level_touched(candle: List[Any], level: float, luft: float) -> bool:
    high = float(candle[2])
    low = float(candle[3])
    close = float(candle[4])
    return (low <= level <= high) or (abs(close - level) <= luft)


def calc_signal_score(
    trend_score: float,
    level_score: float,
    distance_score: float,
    impulse_score: float,
    structure_score: float,
) -> float:
    return (
        trend_score * EDGE_WEIGHTS["trend_alignment"]
        + level_score * EDGE_WEIGHTS["level_quality"]
        + distance_score * EDGE_WEIGHTS["distance_to_level"]
        + impulse_score * EDGE_WEIGHTS["impulse"]
        + structure_score * EDGE_WEIGHTS["market_structure"]
    )


def normalize_v2(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val + 1e-9)))


def sigmoid_v2(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def compute_edge_score_v2(
    trend_score: float,
    level_score: float,
    distance_score: float,
    impulse_score: float,
    structure_score: float,
    atr: float,
    stop_distance: float,
    spread: float = 0.0,
) -> Optional[float]:
    if atr <= 0 or stop_distance <= 0:
        return None

    trend = normalize_v2(trend_score)
    level = normalize_v2(level_score)
    dist = normalize_v2(distance_score)
    impulse = normalize_v2(impulse_score)
    structure = normalize_v2(structure_score)

    base_score = (
        trend * 0.25
        + level * 0.25
        + dist * 0.20
        + impulse * 0.20
        + structure * 0.10
    )

    penalty = 0.0
    if level < 0.5:
        penalty += 0.15
    if dist < 0.2:
        penalty += 0.10
    if structure < 0.2:
        penalty += 0.05
    if stop_distance < atr * 0.3:
        penalty += 0.10
    if stop_distance > atr * 2.0:
        penalty += 0.10
    if spread > 0:
        penalty += min(0.05, spread)

    quality_boost = sigmoid_v2((trend + level + impulse) * 2 - 2.5) * 0.15
    edge_value = base_score + quality_boost - penalty
    return max(0.0, min(1.0, edge_value))


def calc_adaptive_edge_score(
    trend_score: float,
    level_score: float,
    distance_score: float,
    impulse_score: float,
    structure_score: float,
) -> float:
    adaptive_weights = get_adaptive_weights()
    if not adaptive_weights:
        return calc_signal_score(
            trend_score,
            level_score,
            distance_score,
            impulse_score,
            structure_score,
        )

    trend_weight = max(adaptive_weights.get("trend_score", 0.0), 0.10)
    level_weight = adaptive_weights.get("level_score", 0.0)
    distance_weight = max(adaptive_weights.get("distance_score", 0.0), -0.20)
    impulse_weight = adaptive_weights.get("impulse_score", 0.0)
    structure_weight = max(adaptive_weights.get("structure_score", 0.0), 0.10)

    return (
        trend_score * trend_weight
        + level_score * level_weight
        + distance_score * distance_weight
        + impulse_score * impulse_weight
        + structure_score * structure_weight
    )


# -------------------- Логика поиска и открытия сделок --------------------


def process_symbol(client: BybitClient, symbol: str, today_trades: int) -> int:
    constraints = get_symbol_constraints(client, symbol)
    if not constraints:
        logger.info("%s: не удалось получить параметры инструмента, NO TRADE", symbol)
        return today_trades

    existing_pos = client.get_position(symbol)
    if existing_pos:
        size = float(existing_pos.get("size", 0) or 0)
        side = existing_pos.get("side")
        logger.info("%s: уже есть открытая позиция side=%s size=%s, новые входы запрещены", symbol, side, size)
        return today_trades

    klines_4h = client.get_kline(symbol, TIMEFRAME_TREND, limit=250, closed_only=True)
    if len(klines_4h) < 30:
        logger.info("%s: недостаточно закрытых 4H свечей", symbol)
        return today_trades

    trend = detect_trend_4h(klines_4h)
    logger.info("%s: тренд %s = %s", symbol, TIMEFRAME_TREND, trend)

    if trend == "FLAT":
        logger.info("%s: FLAT по тренду, пропускаем", symbol)
        return today_trades

    level_pair = find_level_break_trend(klines_4h)
    if level_pair is None:
        logger.info("%s: нет уровня излома тренда", symbol)
        return today_trades

    level_low, level_high = level_pair
    level = level_low if trend == "UP" else level_high

    atr_4h = calc_atr_from_klines(klines_4h, period=14)
    if atr_4h <= 0:
        logger.info("%s: ATR 4H некорректен", symbol)
        return today_trades

    live_price = client.get_last_price(symbol)
    if live_price <= 0:
        logger.info("%s: не удалось получить актуальную цену", symbol)
        return today_trades

    max_level_distance = atr_4h * MAX_LEVEL_DISTANCE_ATR
    klines_1h_closed = client.get_kline(symbol, TIMEFRAME_SIGNAL, limit=60, closed_only=True)
    if len(klines_1h_closed) < 12:
        logger.info("%s: недостаточно закрытых %s свечей", symbol, TIMEFRAME_SIGNAL)
        return today_trades

    side = "Buy" if trend == "UP" else "Sell"

    trend_score = 1.0 if (
        (trend == "UP" and live_price >= level)
        or (trend == "DOWN" and live_price <= level)
    ) else 0.3

    if max_level_distance > 0:
        distance_score = max(0.1, 1 - abs(live_price - level) / max_level_distance)
    else:
        distance_score = 0.1

    impulse_ok_value = impulse_filter_ok(klines_1h_closed, len(klines_1h_closed) - 1, side)
    impulse_score = 1.0 if impulse_ok_value else 0.3

    clean_level_ok = not is_range_dirty_around_level(klines_4h, level)
    level_score = 1.0 if clean_level_ok else 0.2

    structure_touch = level_touched(klines_1h_closed[-1], level, atr_4h * 0.2)
    structure_score = 0.3 if structure_touch else 0.0

    edge_score = calc_adaptive_edge_score(
        trend_score,
        level_score,
        distance_score,
        impulse_score,
        structure_score,
    )
    score_threshold = 0.55

    logger.info(
        "%s EDGE_SCORE=%.3f (trend=%.2f level=%.2f dist=%.2f impulse=%.2f struct=%.2f) threshold=%.3f",
        symbol,
        edge_score,
        trend_score,
        level_score,
        distance_score,
        impulse_score,
        structure_score,
        score_threshold,
    )

    if edge_score < score_threshold:
        logger.info(
            "%s: EDGE_SCORE=%.3f < %.3f, NO TRADE",
            symbol,
            edge_score,
            score_threshold,
        )
        return today_trades

    entry = live_price
    if side == "Buy":
        raw_stop = level - atr_4h * SL_ATR_BUFFER
        raw_tp = entry + (entry - raw_stop) * MIN_RR
    else:
        raw_stop = level + atr_4h * SL_ATR_BUFFER
        raw_tp = entry - (raw_stop - entry) * MIN_RR

    if side == "Buy":
        stop_dec, stop_str = normalize_price(raw_stop, constraints, ROUND_DOWN)
        tp_dec, tp_str = normalize_price(raw_tp, constraints, ROUND_UP)
    else:
        stop_dec, stop_str = normalize_price(raw_stop, constraints, ROUND_UP)
        tp_dec, tp_str = normalize_price(raw_tp, constraints, ROUND_DOWN)

    stop = float(stop_dec)
    tp = float(tp_dec)

    if stop <= 0 or tp <= 0:
        logger.info("%s: после нормализации цены stop/tp некорректны, NO TRADE", symbol)
        return today_trades

    if side == "Buy":
        risk = entry - stop
        reward = tp - entry
    else:
        risk = stop - entry
        reward = entry - tp

    if risk <= 0 or reward <= 0:
        logger.info("%s: некорректный risk/reward после нормализации, NO TRADE", symbol)
        return today_trades

    try:
        edge_v2 = compute_edge_score_v2(
            trend_score,
            level_score,
            distance_score,
            impulse_score,
            structure_score,
            atr_4h,
            risk,
        )
    except Exception as exc:
        logger.warning("%s: ошибка расчёта edge_v2, fallback без блокировки: %s", symbol, exc)
        edge_v2 = None

    rr = reward / risk
    if rr < 2.8:
        logger.info("%s: RR=%.2f < %.2f, NO TRADE", symbol, rr, 2.8)
        return today_trades

    max_stop_distance = atr_4h * MAX_STOP_ATR
    if risk > max_stop_distance * 1.3:
        logger.info("%s: расстояние до стопа %.6f больше лимита %.6f, NO TRADE", symbol, risk, max_stop_distance * 1.3)
        return today_trades

    min_live_buffer = atr_4h * LIVE_SL_BUFFER
    if risk < min_live_buffer:
        logger.info("%s: до стопа осталось слишком мало %.6f < %.6f, NO TRADE", symbol, risk, min_live_buffer)
        return today_trades

    balance_info = client.get_usdt_balance_info("USDT")
    wallet_balance = float(balance_info.get("walletBalance", Decimal("0")))
    available_balance = float(balance_info.get("availableBalance", Decimal("0")))
    equity_balance = float(balance_info.get("equity", Decimal("0")))

    margin_balance = available_balance if available_balance > 0 else wallet_balance
    if margin_balance <= 0:
        margin_balance = equity_balance

    positive_risk_candidates = [value for value in (wallet_balance, margin_balance) if value > 0]
    risk_balance = min(positive_risk_candidates) if positive_risk_candidates else 0.0

    if risk_balance <= 0:
        logger.info("%s: не удалось получить usable balance, NO TRADE", symbol)
        return today_trades

    qty_by_risk = calc_qty_by_risk(risk_balance, RISK_PER_TRADE, entry, stop)
    qty_by_margin = calc_qty_by_margin(margin_balance, LEVERAGE, entry)
    qty_raw = min(qty_by_risk, qty_by_margin)

    if qty_raw <= 0:
        logger.info("%s: qty_raw <= 0, NO TRADE", symbol)
        return today_trades

    if qty_raw < qty_by_risk:
        logger.info(
            "%s: размер позиции ограничен маржой: risk_qty=%.6f -> capped_qty=%.6f",
            symbol,
            qty_by_risk,
            qty_raw,
        )

    qty_dec, qty_str = normalize_qty(qty_raw, constraints, order_type="Market")
    if qty_dec <= 0:
        logger.info("%s: размер позиции меньше minOrderQty или не проходит по шагу, NO TRADE", symbol)
        return today_trades

    reference_balance_candidates = [value for value in (wallet_balance, equity_balance, risk_balance) if value > 0]
    portfolio_reference_balance = max(reference_balance_candidates) if reference_balance_candidates else 0.0
    if portfolio_reference_balance <= 0:
        logger.info("%s: не удалось определить portfolio reference balance, NO TRADE", symbol)
        return today_trades

    planned_risk_usdt = float(qty_dec) * risk
    portfolio_state = get_open_portfolio_risk_state()
    same_side_count = int(portfolio_state["directional_count"].get(side, 0))
    if same_side_count >= MAX_POSITIONS_PER_DIRECTION:
        logger.info(
            "%s: уже открыто %s позиций в направлении %s (лимит %s), NO TRADE",
            symbol,
            same_side_count,
            side,
            MAX_POSITIONS_PER_DIRECTION,
        )
        return today_trades

    directional_risk_limit = portfolio_reference_balance * MAX_DIRECTIONAL_RISK_PCT
    total_open_risk_limit = portfolio_reference_balance * MAX_TOTAL_OPEN_RISK_PCT
    projected_directional_risk = portfolio_state["directional_risk"].get(side, 0.0) + planned_risk_usdt
    projected_total_open_risk = portfolio_state["total_risk_usdt"] + planned_risk_usdt

    if directional_risk_limit > 0 and projected_directional_risk > directional_risk_limit:
        logger.info(
            "%s: риск по направлению %s %.4f > %.4f, NO TRADE",
            symbol,
            side,
            projected_directional_risk,
            directional_risk_limit,
        )
        return today_trades

    if total_open_risk_limit > 0 and projected_total_open_risk > total_open_risk_limit:
        logger.info(
            "%s: общий открытый риск %.4f > %.4f, NO TRADE",
            symbol,
            projected_total_open_risk,
            total_open_risk_limit,
        )
        return today_trades

    notional = qty_dec * Decimal(str(entry))
    if notional < constraints["minNotionalValue"]:
        logger.info(
            "%s: notional %s < minNotionalValue %s, NO TRADE",
            symbol,
            notional,
            constraints["minNotionalValue"],
        )
        return today_trades

    desired_api_sl = entry - atr_4h * API_SL_ATR_BUFFER if side == "Buy" else entry + atr_4h * API_SL_ATR_BUFFER
    if side == "Buy":
        sl_for_api_dec, sl_for_api_str = normalize_price(
            min(stop, desired_api_sl),
            constraints,
            ROUND_DOWN,
        )
    else:
        sl_for_api_dec, sl_for_api_str = normalize_price(
            max(stop, desired_api_sl),
            constraints,
            ROUND_UP,
        )

    sl_for_api = float(sl_for_api_dec)
    if side == "Buy" and sl_for_api >= entry:
        logger.info("%s: sl_for_api >= entry, NO TRADE", symbol)
        return today_trades
    if side == "Sell" and sl_for_api <= entry:
        logger.info("%s: sl_for_api <= entry, NO TRADE", symbol)
        return today_trades

    try:
        client.set_leverage(symbol, LEVERAGE)
    except Exception as exc:
        logger.warning("%s: ошибка установки плеча: %s", symbol, exc)

    logger.info(
        "%s: сигнал %s edge_old=%.3f edge_v2=%s entry=%.6f stop=%s tp=%s sl_for_api=%s qty=%s wallet=%.2f available=%.2f equity=%.2f rr=%.2f",
        symbol,
        side,
        edge_score,
        "n/a" if edge_v2 is None else f"{edge_v2:.3f}",
        entry,
        stop_str,
        tp_str,
        sl_for_api_str,
        qty_str,
        wallet_balance,
        available_balance,
        equity_balance,
        rr,
    )

    order_resp = client.place_order(
        symbol=symbol,
        side=side,
        qty=qty_str,
        order_type="Market",
        position_idx=POSITION_IDX,
        order_link_id=build_order_link_id(symbol, side, "open"),
    )
    logger.info("%s: ответ на open order: %s", symbol, order_resp)

    if order_resp.get("retCode") != 0:
        return today_trades

    order_result = order_resp.get("result", {}) or {}
    order_id = str(order_result.get("orderId", "") or "")
    order_link_id = str(order_result.get("orderLinkId", "") or "")
    risk_usdt_value = round(risk_balance * RISK_PER_TRADE, 4)
    planned_rr_value = round(rr, 4)

    journal_base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side,
        "entry": round(entry, 8),
        "stop": round(stop, 8),
        "tp": round(tp, 8),
        "risk_usdt": risk_usdt_value,
        "planned_rr": planned_rr_value,
        "atr_4h": round(atr_4h, 8),
        "trend": trend,
        "level": round(level, 8),
        "impulse_ok": impulse_ok_value,
    }

    journal_meta = {
        "order_id": order_id,
        "order_link_id": order_link_id,
        "wallet_balance": round(wallet_balance, 8),
        "available_balance": round(available_balance, 8),
        "equity_balance": round(equity_balance, 8),
        "risk_balance": round(risk_balance, 8),
        "margin_balance": round(margin_balance, 8),
        "qty": qty_str,
        "sl_for_api": round(sl_for_api, 8),
        "edge_score": round(edge_score, 6),
        "edge_old": round(edge_score, 6),
        "edge_v2": None if edge_v2 is None else round(edge_v2, 6),
        "trend_score": round(trend_score, 6),
        "level_score": round(level_score, 6),
        "distance_score": round(distance_score, 6),
        "impulse_score": round(impulse_score, 6),
        "structure_score": round(structure_score, 6),
    }

    def journal(status: str, note: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"message": note, **journal_meta}
        if extra:
            payload.update(extra)
        append_journal_row(
            {
                **journal_base,
                "status": status,
                "note": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
        )

    position = wait_for_position(client, symbol, side)
    if not position:
        logger.info("%s: позиция не появилась после ACK ордера", symbol)
        journal("entry_ack_no_position", "order accepted but position not found in polling window")
        return today_trades

    actual_position_idx = int(position.get("positionIdx", POSITION_IDX))
    current_price = client.get_last_price(symbol)
    if current_price <= 0:
        logger.info("%s: нет актуальной цены перед установкой SL/TP, закрываем позицию", symbol)
        close_resp = force_close_position(client, symbol, side, qty_str, actual_position_idx)
        logger.info("%s: аварийное закрытие позиции: %s", symbol, close_resp)
        journal("force_closed_no_price", "no market price before trading-stop", {"close_response": close_resp})
        return today_trades

    if side == "Buy" and sl_for_api >= current_price:
        logger.info("%s: SL выше/равен рынку, закрываем позицию", symbol)
        close_resp = force_close_position(client, symbol, side, qty_str, actual_position_idx)
        logger.info("%s: аварийное закрытие позиции: %s", symbol, close_resp)
        journal(
            "force_closed_invalid_sl_buy",
            f"sl_for_api={sl_for_api}, current_price={current_price}",
            {"close_response": close_resp},
        )
        return today_trades

    if side == "Sell" and sl_for_api <= current_price:
        logger.info("%s: SL ниже/равен рынку, закрываем позицию", symbol)
        close_resp = force_close_position(client, symbol, side, qty_str, actual_position_idx)
        logger.info("%s: аварийное закрытие позиции: %s", symbol, close_resp)
        journal(
            "force_closed_invalid_sl_sell",
            f"sl_for_api={sl_for_api}, current_price={current_price}",
            {"close_response": close_resp},
        )
        return today_trades

    ts_resp = client.set_trading_stop(
        symbol=symbol,
        position_idx=actual_position_idx,
        stop_loss=sl_for_api_str,
        take_profit=tp_str,
        trigger_by=TRIGGER_BY,
    )
    logger.info("%s: установка SL/TP: %s", symbol, ts_resp)

    if ts_resp.get("retCode") != 0:
        logger.info("%s: trading-stop не установлен, закрываем позицию", symbol)
        close_resp = force_close_position(client, symbol, side, qty_str, actual_position_idx)
        logger.info("%s: аварийное закрытие позиции: %s", symbol, close_resp)
        journal(
            "force_closed_trading_stop_error",
            "trading-stop returned error",
            {"trading_stop_response": ts_resp, "close_response": close_resp},
        )
        return today_trades

    journal("opened_with_sl_tp", "ok", {"position_idx": actual_position_idx, "trading_stop_response": ts_resp})
    return today_trades + 1


# -------------------- Основной цикл --------------------


def current_day_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def validate_startup() -> None:
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise ValueError("Не заданы BYBIT_API_KEY / BYBIT_API_SECRET в .env")
    if not SYMBOLS:
        raise ValueError("В конфиге пустой список symbols")
    if not (0 < RISK_PER_TRADE < 1):
        raise ValueError("risk_per_trade должен быть между 0 и 1")
    if LEVERAGE <= 0:
        raise ValueError("leverage должен быть > 0")
    if MAX_DAILY_LOSS_PCT < 0:
        raise ValueError("max_daily_loss_pct должен быть >= 0")
    if MAX_CONSECUTIVE_LOSSES < 0:
        raise ValueError("max_consecutive_losses должен быть >= 0")
    if SLEEP_SECONDS <= 0:
        raise ValueError("sleep_seconds должен быть > 0")
    if SIGNAL_SCORE_THRESHOLD < 1 or SIGNAL_SCORE_THRESHOLD > 3:
        raise ValueError("signal_score_threshold должен быть между 1 и 3")
    if ADAPTIVE_HALF_LIFE_DAYS <= 0:
        raise ValueError("adaptive_half_life_days должен быть > 0")
    if MAX_DIRECTIONAL_RISK_PCT < 0:
        raise ValueError("max_directional_risk_pct должен быть >= 0")
    if MAX_TOTAL_OPEN_RISK_PCT < 0:
        raise ValueError("max_total_open_risk_pct должен быть >= 0")
    if MAX_POSITIONS_PER_DIRECTION <= 0:
        raise ValueError("max_positions_per_direction должен быть > 0")
    if MIN_LIVE_TRADES_FOR_ADAPTIVE_WEIGHTS < 0:
        raise ValueError("min_live_trades_for_adaptive_weights должен быть >= 0")
    if REQUEST_RECV_WINDOW_MS <= 0:
        raise ValueError("recv_window_ms должен быть > 0")


def main() -> None:
    validate_startup()

    client = BybitClient(BYBIT_API_KEY, BYBIT_API_SECRET, BASE_URL)
    synced_rows = sync_closed_trades_to_journal(client)
    if synced_rows > 0:
        logger.info("Журнал синхронизирован: закрытых сделок обновлено %s", synced_rows)
    adaptive_state = update_adaptive_model()
    if adaptive_state:
        logger.info(
            "Adaptive edge обновлён: learned_threshold=%.3f runtime_threshold=0.550 trades=%s live=%s backtest=%s weights=%s",
            adaptive_state["threshold"],
            adaptive_state["trade_count"],
            adaptive_state["live_trade_count"],
            adaptive_state["backtest_trade_count"],
            adaptive_state["weights"],
        )
    configured_symbols = normalize_symbols(SYMBOLS)
    active_symbols = (
        validate_symbols(client, configured_symbols)
        if SYMBOL_VALIDATION_ON_STARTUP
        else configured_symbols
    )
    if not active_symbols:
        raise ValueError("После валидации не осталось ни одного доступного symbol")

    today = current_day_str()
    today_trades = count_opened_trades_for_day(today)
    risk_state = sync_daily_risk_state(client, today)

    logger.info(
        "Старт бота MODE=%s base_url=%s risk=%.2f%% leverage=%s config=%s opened_today=%s active_symbols=%s/%s",
        MODE,
        BASE_URL,
        RISK_PER_TRADE * 100,
        LEVERAGE,
        os.path.basename(CONFIG_SOURCE),
        today_trades,
        len(active_symbols),
        len(configured_symbols),
    )
    logger.info(
        "Риск-снимок %s: wallet=%.2f available=%.2f equity=%.2f day_net=%.2f fees=%.2f trades=%s losses_in_row=%s daily_stop=%.2f",
        risk_state.day,
        risk_state.current_wallet_balance,
        risk_state.current_available_balance,
        risk_state.current_equity,
        risk_state.net_closed_pnl,
        risk_state.total_fees,
        risk_state.trade_count,
        risk_state.consecutive_losses,
        risk_state.loss_limit_usdt,
    )

    while True:
        synced_rows = sync_closed_trades_to_journal(client)
        if synced_rows > 0:
            logger.info("Журнал синхронизирован: закрытых сделок обновлено %s", synced_rows)
        adaptive_state = update_adaptive_model()
        if adaptive_state:
            logger.info(
                "Adaptive edge обновлён: learned_threshold=%.3f runtime_threshold=0.550 trades=%s live=%s backtest=%s weights=%s",
                adaptive_state["threshold"],
                adaptive_state["trade_count"],
                adaptive_state["live_trade_count"],
                adaptive_state["backtest_trade_count"],
                adaptive_state["weights"],
            )

        now_day = current_day_str()
        if now_day != today:
            today = now_day
            today_trades = count_opened_trades_for_day(today)
            risk_state = sync_daily_risk_state(client, today)
            logger.info("Новый день, счётчик сделок синхронизирован: %s", today_trades)
            logger.info(
                "Риск-снимок %s: wallet=%.2f available=%.2f equity=%.2f day_net=%.2f fees=%.2f trades=%s losses_in_row=%s daily_stop=%.2f",
                risk_state.day,
                risk_state.current_wallet_balance,
                risk_state.current_available_balance,
                risk_state.current_equity,
                risk_state.net_closed_pnl,
                risk_state.total_fees,
                risk_state.trade_count,
                risk_state.consecutive_losses,
                risk_state.loss_limit_usdt,
            )
        else:
            risk_state = sync_daily_risk_state(client, today)

        if risk_state.blocked_reason:
            logger.warning(
                "Новые входы остановлены до следующего дня: %s | day_net=%.2f fees=%.2f losses_in_row=%s",
                risk_state.blocked_reason,
                risk_state.net_closed_pnl,
                risk_state.total_fees,
                risk_state.consecutive_losses,
            )
            logger.info("Цикл по монетам завершён, спим %s секунд", SLEEP_SECONDS)
            time.sleep(SLEEP_SECONDS)
            continue

        for symbol in active_symbols:
            try:
                today_trades = process_symbol(client, symbol, today_trades)
            except Exception as exc:
                logger.exception("Ошибка при обработке %s: %s", symbol, exc)

        logger.info("Цикл по монетам завершён, спим %s секунд", SLEEP_SECONDS)
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
