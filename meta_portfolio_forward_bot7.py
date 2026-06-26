"""Bot #7 launcher (prop-aware champion) — same engine as
meta_portfolio_forward.py, reads config_bot7.json + .env_bot7.

Пары = только положительные live-пары из анализа 26.06.2026
(TIA/SUI/DOT/ATOM/ETH/ETC/BNB). Дефолтный regime-движок (без overfit-гибрида).
Риск ужат под проп-челлендж (см. config_bot7.json: max_daily_loss_pct=0.04,
max_consecutive_losses=2). Бота #1 (живой проп) НЕ задевает — отдельный
аккаунт/конфиг/env. См. meta_portfolio_forward_bot6.py для логики env-before-import.

    python3 meta_portfolio_forward_bot7.py \
        --symbols TIAUSDT,SUIUSDT,DOTUSDT,ATOMUSDT,ETHUSDT,ETCUSDT,BNBUSDT --max-concurrent 4
"""

import os

# Жёстко прибиваем env-файл ДО импорта движка (не setdefault), чтобы чужой токен
# из окружения процесса не мог выжить. Бот #7 = всегда .env_bot7.
os.environ["BYBIT_ENV_FILE"] = ".env_bot7"
os.environ["BYBIT_CONFIG_FILE"] = "config_bot7.json"

# только потом импорты движка
from meta_portfolio_forward import main  # noqa: E402  (must import after env setup)

if __name__ == "__main__":
    main()
