"""Bot #6 launcher (SNX/BNB/BCH/ETH/DOGE/ADA) — same engine as
meta_portfolio_forward.py, reads config_bot6.json + .env_bot6.

config_bot6.json carries regime_params_override={"trend_rr_target": 1.8}, which
main.py applies to the live TP / RR-floor logic (optimized hybrid). See
meta_portfolio_forward_bot2.py for the env-before-import rationale.

    python3 meta_portfolio_forward_bot6.py \
        --symbols SNXUSDT,BNBUSDT,BCHUSDT,ETHUSDT,DOGEUSDT,ADAUSDT --max-concurrent 4
"""

import os

# РАДИКАЛЬНО: жёстко прибиваем env-файл ДО импорта движка (не setdefault), чтобы
# чужой токен из окружения процесса не мог выжить. Бот #6 = всегда .env_bot6.
os.environ["BYBIT_ENV_FILE"] = ".env_bot6"
os.environ["BYBIT_CONFIG_FILE"] = "config_bot6.json"

# только потом импорты движка
from meta_portfolio_forward import main  # noqa: E402  (must import after env setup)

if __name__ == "__main__":
    main()
