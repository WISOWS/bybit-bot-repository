"""Quick single-pair RR curve: how avg OOS / DD vary with trend_rr_target."""
import math, os, time
import portfolio_mtm as P
from backtest import load_ohlcv_csv
from bot3_portfolio import mtm_dd_stats
from multi_strategy_backtest import generate_trades
from research_strategies import make_regime_switch_strategy, RegimeSwitchParams

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = ["1000PEPEUSDT","ADAUSDT","APTUSDT","BCHUSDT","BNBUSDT","BTCUSDT","DOGEUSDT","ENSUSDT",
        "ETHUSDT","HBARUSDT","LTCUSDT","SNXUSDT","TONUSDT","VETUSDT","WLDUSDT","XRPUSDT"]
IS=(P.to_ms("2023-06-01"),P.to_ms("2025-01-01")); OOS=(P.to_ms("2025-01-01"),P.to_ms("2026-06-01"))

def single(raw,cmaps,s,win):
    P.SYMBOLS=[s];P.RISK=0.0050
    ex,m=P.run_portfolio(raw,window=win,max_concurrent=1)
    if not m: return None
    dd=mtm_dd_stats(ex,cmaps,(s,))
    return m["annual_return_pct"],dd["max_dd_pct"],m["profit_factor"],m["winrate_pct"],m["trades"]

def main():
    K,cmaps={},{}
    for s in POOL:
        k1=load_ohlcv_csv(os.path.join(BASE,"ohlcv",f"{s}_60.csv"));k4=load_ohlcv_csv(os.path.join(BASE,"ohlcv",f"{s}_240.csv"))
        K[s]=(k1,k4);cmaps[s]={int(r[0]):float(r[4]) for r in k1}
    print("rr   avgOOS avgIS medOOSdd avgWin avgPF pass tr",flush=True)
    for rr in [1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.2]:
        strat=make_regime_switch_strategy(RegimeSwitchParams(trend_rr_target=rr))
        def g(k1,i,k4,j4,nw,_st=strat):
            s=_st(k1,i,k4,j4); return None if s is None else (s.side,s.entry,s.stop,s.tp)
        per=[];ntr=0;wins=[];pfs=[]
        for s in POOL:
            raw={s:generate_trades(g,*K[s],min_4h=29)};ntr+=len(raw[s])
            o=single(raw,cmaps,s,OOS);i=single(raw,cmaps,s,IS)
            if o and i: per.append((o,i));wins.append(o[3]);pfs.append(o[2] if math.isfinite(o[2]) else 5)
        ao=sum(o[0] for o,_ in per)/len(per);ai=sum(i[0] for _,i in per)/len(per)
        dds=sorted(o[1] for o,_ in per);md=dds[len(dds)//2]
        npass=sum(1 for o,i in per if o[1]<8 and i[1]<8 and o[0]>0)
        print(f"{rr:<4} {ao:6.1f} {ai:6.1f} {md:8.2f} {sum(wins)/len(wins):6.1f} {sum(pfs)/len(pfs):5.2f} {npass:>2}/16 {ntr//16}",flush=True)

if __name__=="__main__": main()
