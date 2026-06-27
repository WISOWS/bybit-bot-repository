# CFT prop bot — деплой (порт DOT-set под отдельный CFT-аккаунт)

Бот = тот же `meta_portfolio_forward.py` движок, пары DOT/AVAX/RUNE/ETC/XLM,
на ОТДЕЛЬНОМ CFT-Bybit аккаунте и ОТДЕЛЬНОМ VPS/IP (не там, где HyroTrader/демо).

Бэктест-обоснование (26.06.2026): PF 2.03, 608 сделок, OOS-WR 50%, P_pass CFT 2-Step ~92%,
ожид. funded-доход ~$2-4k/мес на $100k при 0.3% риске (haircut/raw). См. scratchpad бэктесты.

## 0. Почему отдельный VPS/IP (НЕ пропускать)
CFT (и любой проп) банит за «связанные аккаунты»: совпадение IP с другим юзером
и одинаковые сделки в пределах ~10мс по разным фирмам = флаг копи-трейда → отказ в выплате.
- Отдельный VPS с другим IP, чем `136.244.94.113` (HyroTrader/демо-флот).
- Пары DOT-set НЕ пересекаются с ботом #1 HyroTrader (NEAR/SOL/LINK/ENA) — это ОК, оставляем так.
- Отдельный платёж за челлендж (не с того же, что HyroTrader, если возможно).

## 1. Купить челлендж
- CFT 2-Step → $100,000 → платформа **BYBIT** → промокод **MATCH** (−10% → $475.20).
- НЕ брать Break/Instant/1-Step. Addon'ы не обязательны.
- Регистрация = «Войти через Google» в app.cryptofundtrader.com ИЛИ автоматически при покупке (доступы на email).

## 2. Привязать Bybit API
- В дашборде CFT: создать привязанный Bybit-аккаунт, сгенерить API key (с правами на торговлю), подключить к CFT.
- Уточнить у CFT: это Bybit DEMO или REAL аккаунт → выставить `MODE` в `.env_cft` (DEMO→api-demo, REAL→api.bybit.com).

## 3. Поднять VPS и задеплоить
```bash
# на новом VPS (Ubuntu, другой IP)
apt update && apt install -y python3.12-venv python3-pip git
git clone https://github.com/WISOWS/bybit-bot-repository.git
cd bybit-bot-repository
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # или нужные пакеты
cp .env_cft.example .env_cft && nano .env_cft        # вписать CFT-Bybit ключ + MODE
# config_cft.json уже в репо (risk 0.01 = спринт)
cp deploy/bybit-cft-forward.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now bybit-cft-forward
journalctl -u bybit-cft-forward -n 40 --no-pager      # проверить, что пары прошли валидацию
```

## 4. Спринт → Funded (КРИТИЧНО)
- Челлендж идёт на **risk 0.01** (спринт к +8% Stage1, затем +5% Stage2). ~10-14 дней, P_pass ~90%.
- Защита: `max_daily_loss_pct=0.04` тормозит день при −4% (под лимит CFT 5%).
- **СРАЗУ после прохождения** обоих фаз: в `config_cft.json` поставить `risk_per_trade: 0.003`,
  `systemctl restart bybit-cft-forward`. Иначе funded-счёт (лимит 10%) сольёт историческую DD ~−20% при 1%.
  На 0.3% maxDD ~−6% — с запасом.

## 5. Выплаты
- KYC в дашборде ДО первой выплаты (после funded). Выплаты крипта (USDT/BTC), 8-48ч, мин $100.
- Взнос $475 возвращается с первой выплатой.

## Файлы порта (в этом репо)
- `config_cft.json` — конфиг (риск-профили, DOT-set, daily 4%)
- `meta_portfolio_forward_cft.py` — обёртка (env_cft + config_cft)
- `.env_cft.example` — шаблон ключей (реальный `.env_cft` в .gitignore)
- `deploy/bybit-cft-forward.service` — systemd
- ModelSpec для DOT/AVAX/RUNE/ETC/XLM уже в `research_search_meta_portfolio.py`

## Чего НЕ хватает (блокеры на Егоре)
1. Отдельный VPS с другим IP.
2. Купленный CFT 2-Step + привязанный Bybit API-ключ (→ .env_cft).
3. Подтвердить у CFT: DEMO или REAL Bybit-аккаунт (→ MODE).
