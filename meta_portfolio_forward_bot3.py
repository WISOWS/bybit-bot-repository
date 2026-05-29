"""Bot #3 launcher (ATOM/INJ/TIA) — same engine as meta_portfolio_forward.py,
reads config_bot3.json + .env_bot3. See meta_portfolio_forward_bot2.py for details.
"""

import os

os.environ.setdefault("BYBIT_CONFIG_FILE", "config_bot3.json")
os.environ.setdefault("BYBIT_ENV_FILE", ".env_bot3")

from meta_portfolio_forward import main  # noqa: E402  (must import after env setup)

if __name__ == "__main__":
    main()
