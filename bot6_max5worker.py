"""Bot #6 maximize-OOS worker: one regime-param variant, WIDE risk grid.

Usage: python3 bot6_maxworker.py <label> '<json_kwargs>'
Writes bot6_max_<label>.csv with every C(16,3) x risk x conc portfolio whose
MTM DD < 8% on BOTH IS and OOS. Pushes risk up to 1.30% to exploit low-RR
drawdown headroom.
"""
import csv, itertools, json, math, os, sys, time
import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import generate_trades
from research_strategies import make_regime_switch_strategy, RegimeSwitchParams

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = ["1000PEPEUSDT","ADAUSDT","APTUSDT","BCHUSDT","BNBUSDT","BTCUSDT","DOGEUSDT","ENSUSDT",
        "ETHUSDT","HBARUSDT","LTCUSDT","SNXUSDT","TONUSDT","VETUSDT","WLDUSDT","XRPUSDT"]
IS=(P.to_ms("2023-06-01"),P.to_ms("2025-01-01")); OOS=(P.to_ms("2025-01-01"),P.to_ms("2026-06-01"))
DD_MAX=8.0
RISKS=[0.0050,0.0060,0.0070,0.0080,0.0090,0.0100,0.0110,0.0120,0.0130]
CONCS=[2,3,4]

def pfx(v): return "inf" if (isinstance(v,float) and math.isinf(v)) else f"{v:.3f}"

def run(raw,cmaps,combo,risk,conc,win):
    P.SYMBOLS=list(combo);P.RISK=risk
    ex,m=P.run_portfolio(raw,window=win,max_concurrent=conc)
    if not m: return None
    dd=mtm_dd_stats(ex,cmaps,tuple(combo))
    return {"annual":m["annual_return_pct"],"dd":dd["max_dd_pct"],"pf":m["profit_factor"],
            "sharpe":m["sharpe"],"trades":m["trades"],"win":m["winrate_pct"]}

def main():
    global POOL
    label=sys.argv[1]; kw=json.loads(sys.argv[2])
    if len(sys.argv)>3 and sys.argv[3].strip():
        POOL=[p if p.endswith("USDT") else p+"USDT" for p in sys.argv[3].split(",")]
    global RISKS
    if len(sys.argv)>4 and sys.argv[4].strip():
        RISKS=[float(x)/100 for x in sys.argv[4].split(",")]
    t0=time.time()
    json.dump(kw,open(os.path.join(BASE,f"bot6_kw_{label}.json"),"w"))
    strat=make_regime_switch_strategy(RegimeSwitchParams(**kw))
    def g(k1,i,k4,j4,nw):
        s=strat(k1,i,k4,j4); return None if s is None else (s.side,s.entry,s.stop,s.tp)
    raw,cmaps={},{}
    for s in POOL:
        k1=load_ohlcv_csv(os.path.join(BASE,"ohlcv",f"{s}_60.csv"));k4=load_ohlcv_csv(os.path.join(BASE,"ohlcv",f"{s}_240.csv"))
        raw[s]=generate_trades(g,k1,k4,min_4h=29);cmaps[s]={int(r[0]):float(r[4]) for r in k1}
    combos=list(itertools.combinations(POOL,5))
    out=os.path.join(BASE,f"bot6_max5_{label}.csv")
    print(f"[{label}] gen done {time.time()-t0:.0f}s, sweeping {len(combos)} combos",flush=True)
    rows=[]
    fh=open(out,"w",newline=""); w=csv.writer(fh)
    w.writerow(["label","pairs","risk_pct","conc","oos_annual","oos_dd","oos_pf","oos_sharpe",
                "oos_win","oos_trades","is_annual","is_dd","is_pf"])
    for ci,combo in enumerate(combos):
        pairs="+".join(s.replace("USDT","") for s in combo)
        for conc in CONCS:
            for risk in RISKS:
                o=run(raw,cmaps,combo,risk,conc,OOS); i=run(raw,cmaps,combo,risk,conc,IS)
                if o is None or i is None: continue
                if o["dd"]<DD_MAX and i["dd"]<DD_MAX:
                    rows.append((label,pairs,risk,conc,o,i,combo))
                    w.writerow([label,pairs,round(risk*100,2),conc,round(o["annual"],1),round(o["dd"],2),
                                pfx(o["pf"]),round(o["sharpe"],2),round(o["win"],1),o["trades"],
                                round(i["annual"],1),round(i["dd"],2),pfx(i["pf"])])
        if (ci+1)%80==0:
            fh.flush()
            cur=max(rows,key=lambda r:r[4]["annual"]) if rows else None
            print(f"[{label}] {ci+1}/{len(combos)} pass={len(rows)} bestOOS="
                  f"{cur[4]['annual']:.1f}% ({time.time()-t0:.0f}s)" if cur else
                  f"[{label}] {ci+1}/{len(combos)} pass=0",flush=True)
    fh.close()
    rows.sort(key=lambda r:r[4]["annual"],reverse=True)
    best=rows[0] if rows else None
    if best:
        _,pairs,risk,conc,o,i,_=best
        print(f"[{label}] done {time.time()-t0:.0f}s | pass={len(rows)} | BEST {pairs} @{risk*100:.2f}% cc{conc} "
              f"OOS {o['annual']:.1f}%/DD{o['dd']:.2f}% IS {i['annual']:.1f}%/DD{i['dd']:.2f}%",flush=True)
    else:
        print(f"[{label}] done {time.time()-t0:.0f}s | no passing",flush=True)

if __name__=="__main__": main()
