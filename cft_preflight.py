#!/usr/bin/env python3
"""CFT pre-flight: проверка ПОСЛЕ привязки Bybit-ключа, ДО запуска бота.
Читает .env_cft, проверяет что ключ рабочий, баланс ~$100k, пары торгуемы.
Запуск:  .venv/bin/python cft_preflight.py
"""
import os, time, hmac, hashlib, json, urllib.parse, urllib.request
RECV="20000"
def load_env(fn=".env_cft"):
    d={}
    if not os.path.exists(fn):
        print(f"!! нет {fn} — скопируй .env_cft.example и впиши ключи"); raise SystemExit(1)
    for ln in open(fn,encoding="utf-8"):
        ln=ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k,v=ln.split("=",1); d[k.strip()]=v.strip()
    return d
e=load_env()
MODE=e.get("MODE","DEMO"); KEY=e.get("BYBIT_API_KEY",""); SEC=e.get("BYBIT_API_SECRET","")
BASE="https://api-demo.bybit.com" if MODE=="DEMO" else "https://api.bybit.com"
SYMS=["DOTUSDT","AVAXUSDT","RUNEUSDT","ETCUSDT","XLMUSDT"]
if not KEY or KEY.startswith("your_"):
    print("!! BYBIT_API_KEY не заполнен в .env_cft"); raise SystemExit(1)
def sign(ts,qs): return hmac.new(SEC.encode(),(ts+KEY+RECV+qs).encode(),hashlib.sha256).hexdigest()
def get(path,p,signed=True):
    qs=urllib.parse.urlencode(p); req=urllib.request.Request(BASE+path+"?"+qs)
    if signed:
        ts=str(int(time.time()*1000))
        for k,v in [("X-BAPI-API-KEY",KEY),("X-BAPI-TIMESTAMP",ts),("X-BAPI-RECV-WINDOW",RECV),("X-BAPI-SIGN",sign(ts,qs))]: req.add_header(k,v)
    with urllib.request.urlopen(req,timeout=30) as r: return json.loads(r.read())
print(f"=== CFT pre-flight (MODE={MODE}, base={BASE}) ===")
ok=True
# 1) key works + balance
try:
    b=get("/v5/account/wallet-balance",{"accountType":"UNIFIED"})
    if b.get("retCode")!=0: print(f"!! ключ/доступ: retCode={b.get('retCode')} {b.get('retMsg')}"); ok=False
    else:
        a=b["result"]["list"][0]; eq=float(a.get("totalEquity",0) or 0)
        print(f"[OK] ключ рабочий. Equity={eq:,.2f}")
        if eq<1000: print("   ?? баланс маленький — это точно CFT-evaluation аккаунт?")
except Exception as ex:
    print(f"!! запрос баланса упал: {ex}  (проверь MODE: DEMO vs REAL у CFT)"); ok=False
# 2) symbols tradable
try:
    syms={}; cur=""
    while True:
        p={"category":"linear","limit":1000}
        if cur: p["cursor"]=cur
        d=get("/v5/market/instruments-info",p,signed=False)
        for x in d["result"]["list"]: syms[x["symbol"]]=x.get("status")
        cur=d["result"].get("nextPageCursor","")
        if not cur: break
    for s in SYMS:
        st=syms.get(s,"NOT FOUND")
        print(f"   {s:<9} {st}" + (" [OK]" if st=="Trading" else " !! не торгуется"))
        if st!="Trading": ok=False
except Exception as ex:
    print(f"!! проверка символов упала: {ex}")
print("\n"+("ВСЁ ОК → systemctl start bybit-cft-forward" if ok else "ЕСТЬ ПРОБЛЕМЫ — почини перед запуском"))
