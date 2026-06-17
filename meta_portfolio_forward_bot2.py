"""Bot #2 launcher — same engine as meta_portfolio_forward.py, but a separate
account/config: reads config_bot2.json and .env_bot2 instead of config.json/.env.

main.py loads the config + .env at IMPORT time from the paths named by the
BYBIT_CONFIG_FILE / BYBIT_ENV_FILE env-vars, so we set them BEFORE importing the
forward runner, then delegate to its main().

Symbols and concurrency limit are passed on the command line (the forward runner
reads --symbols / --max-concurrent), e.g.:
    python3 meta_portfolio_forward_bot2.py --symbols ONDOUSDT,ZECUSDT,SUIUSDT --max-concurrent 3
"""

import os

# РАДИКАЛЬНО: жёстко прибиваем env-файл ДО любого импорта движка. Не setdefault —
# иначе чужой BYBIT_ENV_FILE/токен, утёкший в окружение процесса, мог бы выжить.
# Этот лаунчер = бот #2 и точка: всегда .env_bot2 / config_bot2.json.
os.environ["BYBIT_ENV_FILE"] = ".env_bot2"
os.environ["BYBIT_CONFIG_FILE"] = "config_bot2.json"

# только потом импорты движка (main.py читает эти переменные на import-time)
from meta_portfolio_forward import main  # noqa: E402  (must import after env setup)

if __name__ == "__main__":
    main()
