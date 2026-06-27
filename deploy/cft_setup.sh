#!/usr/bin/env bash
# CFT prop bot — деплой на НОВЫЙ VPS (другой IP, НЕ HyroTrader/демо 136.244.94.113).
# Запуск как root:  bash <(curl -s ...)  или после git clone:  bash deploy/cft_setup.sh
set -euo pipefail
REPO="https://github.com/WISOWS/bybit-bot-repository.git"
DIR=/root/bybit-bot-repository

echo "== 1/5 системные пакеты =="
apt-get update -y
apt-get install -y python3-venv python3-pip git

echo "== 2/5 репозиторий =="
if [ -d "$DIR/.git" ]; then (cd "$DIR" && git pull --ff-only || true); else git clone "$REPO" "$DIR"; fi
cd "$DIR"

echo "== 3/5 venv + зависимости =="
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# live-боту нужны requests + dotenv; pandas тянется транзитивно через backtest-импорты
.venv/bin/pip install requests python-dotenv pandas

echo "== 4/5 .env_cft =="
if [ ! -f .env_cft ]; then
  cp .env_cft.example .env_cft
  echo ">>> ВПИШИ ключи CFT-Bybit аккаунта:  nano .env_cft   (BYBIT_API_KEY/SECRET + MODE)"
  echo ">>> потом:  .venv/bin/python cft_preflight.py   &&   systemctl start bybit-cft-forward"
fi

echo "== 5/5 systemd сервис =="
cp deploy/bybit-cft-forward.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable bybit-cft-forward

echo
echo "ГОТОВО. Осталось:"
echo "  1) nano .env_cft        # вписать ключи привязанного к CFT Bybit-аккаунта + MODE (DEMO/REAL)"
echo "  2) .venv/bin/python cft_preflight.py   # проверка: ключ рабочий, пары торгуемы"
echo "  3) systemctl start bybit-cft-forward   # старт (config_cft.json: риск 1% = спринт)"
echo "  4) journalctl -u bybit-cft-forward -n 40 --no-pager   # проверить старт"
echo "  ПОСЛЕ прохождения Stage2: в config_cft.json risk_per_trade -> 0.003, systemctl restart bybit-cft-forward"
