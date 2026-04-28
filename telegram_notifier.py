import json
import logging
import os
import sys
import threading
import time
from typing import Dict
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

logger = logging.getLogger("meta_portfolio_forward.telegram")

TELEGRAM_BOT_TOKEN = str(os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
TELEGRAM_CHAT_ID = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()

TELEGRAM_TIMEOUT_SEC = 5
MESSAGE_DEDUP_WINDOW_SEC = 60.0
ERROR_LOG_DEDUP_WINDOW_SEC = 300.0

LEVEL_PREFIX = {
    "info": "ℹ️",
    "warning": "⚠️",
    "error": "🚨",
}

_lock = threading.Lock()
_last_sent_at: Dict[str, float] = {}
_last_error_logged_at: Dict[str, float] = {}
_missing_config_warned = False


def _configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _warn_missing_config_once() -> None:
    global _missing_config_warned
    with _lock:
        if _missing_config_warned:
            return
        _missing_config_warned = True
    logger.warning(
        "Telegram notifier disabled: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not fully configured; running in no-op mode"
    )


def _prune_old_entries(mapping: Dict[str, float], ttl_sec: float, now: float) -> None:
    expired = [key for key, ts in mapping.items() if now - ts >= ttl_sec]
    for key in expired:
        mapping.pop(key, None)


def _should_skip_duplicate_message(formatted_message: str) -> bool:
    now = time.monotonic()
    with _lock:
        _prune_old_entries(_last_sent_at, MESSAGE_DEDUP_WINDOW_SEC, now)
        last_sent_at = _last_sent_at.get(formatted_message)
        if last_sent_at is not None and now - last_sent_at < MESSAGE_DEDUP_WINDOW_SEC:
            return True
        _last_sent_at[formatted_message] = now
        return False


def _log_send_error_once(exc: Exception) -> None:
    error_key = f"{type(exc).__name__}:{exc}"
    now = time.monotonic()
    with _lock:
        _prune_old_entries(_last_error_logged_at, ERROR_LOG_DEDUP_WINDOW_SEC, now)
        last_logged_at = _last_error_logged_at.get(error_key)
        if last_logged_at is not None and now - last_logged_at < ERROR_LOG_DEDUP_WINDOW_SEC:
            return
        _last_error_logged_at[error_key] = now
    logger.warning("Telegram notify failed: %s: %s", type(exc).__name__, exc, exc_info=True)


def _send_formatted_message_sync(formatted_message: str) -> None:
    payload = urlencode(
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": formatted_message,
        }
    ).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=TELEGRAM_TIMEOUT_SEC) as response:
        response_body = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(response_body)
        if not parsed.get("ok", False):
            raise RuntimeError(f"Telegram API error: {response_body[:300]}")


def _send_formatted_message_async(formatted_message: str) -> None:
    try:
        _send_formatted_message_sync(formatted_message)
    except Exception as exc:
        _log_send_error_once(exc)


def notify(message: str, level: str = "info") -> None:
    text = str(message or "").strip()
    if not text:
        return

    if not _configured():
        _warn_missing_config_once()
        return

    prefix = LEVEL_PREFIX.get(level, LEVEL_PREFIX["info"])
    formatted_message = f"{prefix} {text}"

    if _should_skip_duplicate_message(formatted_message):
        return

    try:
        worker = threading.Thread(
            target=_send_formatted_message_async,
            args=(formatted_message,),
            daemon=True,
            name="telegram-notify",
        )
        # Fire-and-forget so Telegram I/O never blocks the trading cycle.
        worker.start()
    except Exception as exc:
        _log_send_error_once(exc)


def _cli() -> int:
    # CLI intentionally bypasses dedup/rate-limit state so manual testing is deterministic.
    text = " ".join(sys.argv[1:]).strip() or "telegram_notifier test"
    if not _configured():
        _warn_missing_config_once()
        return 1
    try:
        _send_formatted_message_sync(f"{LEVEL_PREFIX['info']} {text}")
        return 0
    except Exception as exc:
        _log_send_error_once(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
