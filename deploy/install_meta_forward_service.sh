#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_TEMPLATE="$PROJECT_DIR/deploy/bybit-meta-forward.service.template"
SERVICE_NAME="${SERVICE_NAME:-bybit-meta-forward.service}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
SERVICE_USER="${SERVICE_USER:-$USER}"
SERVICE_GROUP="${SERVICE_GROUP:-$(id -gn "$SERVICE_USER")}"
BTC_MODE="${BTC_MODE:-allow_flat}"
MIN_HOT_VOL_RATIO="${MIN_HOT_VOL_RATIO:-0.75}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

if [[ ! -f "$PROJECT_DIR/meta_portfolio_forward.py" ]]; then
  echo "meta_portfolio_forward.py не найден в $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
  echo ".env не найден в $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/config.json" ]]; then
  echo "config.json не найден в $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -f "$SERVICE_TEMPLATE" ]]; then
  echo "Шаблон сервиса не найден: $SERVICE_TEMPLATE" >&2
  exit 1
fi

command -v python3 >/dev/null 2>&1 || {
  echo "python3 не установлен" >&2
  exit 1
}

command -v sudo >/dev/null 2>&1 || {
  echo "sudo не установлен" >&2
  exit 1
}

if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
  python3 -m venv "$PROJECT_DIR/.venv"
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install -r "$PROJECT_DIR/requirements.txt" requests python-dotenv

TMP_SERVICE="$(mktemp)"
trap 'rm -f "$TMP_SERVICE"' EXIT

sed \
  -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
  -e "s|__USER__|$SERVICE_USER|g" \
  -e "s|__GROUP__|$SERVICE_GROUP|g" \
  -e "s|__PYTHON_BIN__|$PYTHON_BIN|g" \
  -e "s|__BTC_MODE__|$BTC_MODE|g" \
  -e "s|__MIN_HOT_VOL_RATIO__|$MIN_HOT_VOL_RATIO|g" \
  -e "s|__MAX_CONCURRENT__|$MAX_CONCURRENT|g" \
  -e "s|__SLEEP_SECONDS__|$SLEEP_SECONDS|g" \
  "$SERVICE_TEMPLATE" > "$TMP_SERVICE"

sudo install -m 644 "$TMP_SERVICE" "$SERVICE_PATH"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo
echo "Сервис установлен и запущен: $SERVICE_NAME"
echo "Проверить статус:"
echo "  sudo systemctl status $SERVICE_NAME"
echo "Смотреть логи:"
echo "  journalctl -u $SERVICE_NAME -f"
