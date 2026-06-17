"""Bot #5 launcher (OP/ARB/TON) — same engine as meta_portfolio_forward.py,
reads config_bot5.json + .env_bot5. See meta_portfolio_forward_bot2.py for details.
"""

import os

# РАДИКАЛЬНО: жёстко прибиваем env-файл ДО импорта движка (не setdefault), чтобы
# чужой токен из окружения процесса не мог выжить. Бот #5 = всегда .env_bot5.
os.environ["BYBIT_ENV_FILE"] = ".env_bot5"
os.environ["BYBIT_CONFIG_FILE"] = "config_bot5.json"

# только потом импорты движка
from meta_portfolio_forward import main  # noqa: E402  (must import after env setup)

if __name__ == "__main__":
    main()
