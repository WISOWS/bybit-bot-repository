"""CFT launcher (prop 2-Step, DOT/AVAX/RUNE/ETC/XLM) — тот же движок
meta_portfolio_forward.py, читает config_cft.json + .env_cft.

Отдельный CFT-Bybit аккаунт, отдельный VPS/IP (НЕ тот, где HyroTrader/демо).
Профили риска в config_cft.json: 0.01 спринт → 0.003 funded после прохождения.
См. docs/CFT_DEPLOY.md. env-before-import как в meta_portfolio_forward_bot6.py.

    python3 meta_portfolio_forward_cft.py \
        --symbols DOTUSDT,AVAXUSDT,RUNEUSDT,ETCUSDT,XLMUSDT --max-concurrent 3
"""

import os

# Жёстко прибиваем env/config ДО импорта движка (не setdefault).
os.environ["BYBIT_ENV_FILE"] = ".env_cft"
os.environ["BYBIT_CONFIG_FILE"] = "config_cft.json"

from meta_portfolio_forward import main  # noqa: E402  (must import after env setup)

if __name__ == "__main__":
    main()
