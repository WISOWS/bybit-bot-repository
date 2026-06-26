#!/usr/bin/env python3
"""Часовая сводка по всем 6 Bybit-ботам -> Telegram.

Читает ключи из per-bot .env файлов, опрашивает Bybit (wallet-balance + позиции),
считает PnL от стартовых $100k, шлёт форматированное сообщение в Telegram.

Telegram-конфиг (по приоритету):
  1) .env_summary  -> SUMMARY_BOT_TOKEN / SUMMARY_CHAT_ID  (выделенный бот-сводчик)
  2) fallback: .env -> TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  (бот #1)

Запуск из cron раз в час. Никаких секретов в коде — всё из .env (в .gitignore).
"""
import os, time, hmac, hashlib, json, html
import urllib.parse, urllib.request
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
RECV = "20000"
START = 100000.0

# (env-файл, номер, эмодзи, пары)
BOTS = [
    (".env",      "1", "\U0001F7E2", "HyroTrader live: NEAR/SOL/LINK/ENA"),
    (".env_bot2", "2", "\U0001F535", "Fresh-A: XRP/HYPE/AAVE/TAO/APT"),
    (".env_bot3", "3", "\U0001F7E0", "Improved A/B: DOT/AVAX/RUNE/ETC/XLM"),
    (".env_bot4", "4", "\U0001F7E3", "Bot A: DOT/AVAX/RUNE/ETC/XLM"),
    (".env_bot5", "5", "\U0001F7E1", "Fresh-B: LTC/WLD/JTO/MNT/SEI"),
    (".env_bot6", "6", "⚪", "Bot B: BNB/ETH/ATOM/TIA/SUI"),
]


def load_env(fn):
    path = os.path.join(REPO, fn)
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def server_offset(base):
    """Смещение локальных часов VPS относительно сервера Bybit (мс)."""
    try:
        with urllib.request.urlopen(f"{base}/v5/market/time", timeout=20) as r:
            d = json.loads(r.read().decode())
        srv = int(d["result"]["timeNano"]) // 1_000_000
        return srv - int(time.time() * 1000)
    except Exception:
        return 0


def signed_get(base, path, params, key, secret, offset):
    ts = str(int(time.time() * 1000) + offset)
    qs = urllib.parse.urlencode(params)
    raw = f"{ts}{key}{RECV}{qs}"
    sig = hmac.new(secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        f"{base}{path}?{qs}",
        headers={
            "X-BAPI-API-KEY": key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": RECV,
            "X-BAPI-SIGN": sig,
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def query_bot(env):
    key = env.get("BYBIT_API_KEY")
    sec = env.get("BYBIT_API_SECRET")
    if not key or not sec:
        return {"error": "нет ключей в env"}
    mode = env.get("MODE", "LIVE").upper()
    base = "https://api-demo.bybit.com" if mode == "DEMO" else "https://api.bybit.com"
    off = server_offset(base)
    wb = signed_get(base, "/v5/account/wallet-balance",
                    {"accountType": "UNIFIED"}, key, sec, off)
    if wb.get("retCode") != 0:
        return {"error": f"WB {wb.get('retCode')}: {wb.get('retMsg')}", "mode": mode}
    lst = wb["result"]["list"]
    equity = float(lst[0].get("totalEquity") or 0) if lst else 0.0
    pos = signed_get(base, "/v5/position/list",
                     {"category": "linear", "settleCoin": "USDT"}, key, sec, off)
    opens = []
    if pos.get("retCode") == 0:
        for p in pos["result"]["list"]:
            if float(p.get("size") or 0) > 0:
                opens.append({
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "size": p["size"],
                    "upnl": float(p.get("unrealisedPnl") or 0),
                })
    return {"mode": mode, "equity": equity, "pnl": equity - START, "opens": opens}


def money(x):
    return f"${x:,.2f}".replace(",", " ")


def signed(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}".replace(",", " ")


def resolve_telegram():
    s = load_env(".env_summary")
    if s.get("SUMMARY_BOT_TOKEN") and s.get("SUMMARY_CHAT_ID"):
        return s["SUMMARY_BOT_TOKEN"], s["SUMMARY_CHAT_ID"], "summary"
    b1 = load_env(".env")
    return b1.get("TELEGRAM_BOT_TOKEN"), b1.get("TELEGRAM_CHAT_ID"), "bot1-fallback"


def send_telegram(token, chat, text):
    data = urllib.parse.urlencode({
        "chat_id": chat,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def build_message():
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    lines = [f"\U0001F4CA <b>СВОДКА ПО БОТАМ</b>", html.escape(now), ""]
    tot_eq = 0.0
    tot_base = 0.0
    ok = 0
    for fn, num, emo, pairs in BOTS:
        env = load_env(fn)
        try:
            r = query_bot(env)
        except Exception as ex:
            r = {"error": str(ex)}
        head = f"{emo} <b>#{num}</b> <i>{html.escape(pairs)}</i>"
        if r.get("error"):
            lines.append(head)
            lines.append(f"   ⚠️ {html.escape(str(r['error']))}")
            lines.append("")
            continue
        ok += 1
        tot_eq += r["equity"]
        tot_base += START
        sign_emo = "\U0001F7E2" if r["pnl"] >= 0 else "\U0001F534"
        lines.append(head)
        lines.append(f"   Equity: <code>{money(r['equity'])}</code>")
        lines.append(f"   PnL:    <code>{signed(r['pnl'])}</code> {sign_emo}")
        if r["opens"]:
            for o in r["opens"]:
                lines.append(
                    f"   • {html.escape(o['symbol'])} {o['side']} "
                    f"{o['size']}  uPnL <code>{signed(o['upnl'])}</code>"
                )
        else:
            lines.append("   — позиций нет")
        lines.append("")
    lines.append("━" * 18)
    tot_pnl = tot_eq - tot_base
    pct = (tot_pnl / tot_base * 100) if tot_base else 0.0
    tot_emo = "\U0001F4C8" if tot_pnl >= 0 else "\U0001F4C9"
    lines.append(f"\U0001F4B0 <b>ИТОГО equity:</b> <code>{money(tot_eq)}</code>")
    lines.append(f"{tot_emo} <b>PnL:</b> <code>{signed(tot_pnl)}</code> ({pct:+.2f}%)")
    lines.append(f"<i>ботов опрошено: {ok}/{len(BOTS)}</i>")
    return "\n".join(lines)


def main():
    msg = build_message()
    token, chat, src = resolve_telegram()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if not token or not chat:
        print(f"[{stamp}] НЕТ telegram-конфига (ни .env_summary, ни .env). Сводка:\n{msg}")
        return 1
    try:
        resp = send_telegram(token, chat, msg)
        print(f"[{stamp}] sent via {src}: ok={resp.get('ok')} "
              f"msg_id={resp.get('result', {}).get('message_id')}")
        return 0
    except Exception as ex:
        print(f"[{stamp}] ОШИБКА отправки ({src}): {ex}\n{msg}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
