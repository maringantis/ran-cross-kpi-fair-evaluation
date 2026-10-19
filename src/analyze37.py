import json, numpy as np, pandas as pd, os
from scipy import stats
H=os.path.dirname(os.path.abspath(__file__)); O=H+"/../results/"
R=json.load(open(O+"per_seed.json")); N=pd.read_csv(O+"node_level.csv")
rng=np.random.default_rng(5)
def seedvals(block,det,key,stress=None):
    out=[]
    for r in R:
        b=r["main"] if stress is None else r["stress"][stress]
        out.append(b[det].get(key,np.nan))
    return np.array(out,float)
def boot_ci(x,B=5000):
    x=x[~np.isnan(x)]; m=[rng.choice(x,len(x)).mean() for _ in range(B)]; return np.percentile(m,[2.5,97.5])
dets=list(R[0]["main"].keys())
rows=[]
for d in dets:
    lead=seedvals(0,d,"lead_med"); lo,hi=boot_ci(lead)
    rows.append(dict(method=d,lead=np.nanmean(lead),lead_lo=lo,lead_hi=hi,miss=100*np.mean(seedvals(0,d,"miss")),
        fa=100*np.mean(seedvals(0,d,"fa")),attr=(100*np.nanmean(seedvals(0,d,"attr")) if d in("Mean Cross W12","Uni-Traj W5") or "Cross-Traj" in d else np.nan)))
T=pd.DataFrame(rows); T.to_csv(O+"table_main.csv",index=False); print(T.round(2).to_string(index=False))
# per mode for Cross-Traj W5
nl=N[N.det=="Cross-Traj W5"].groupby(["seed","mode"]).agg(lead=("lead","median"),miss=("detected",lambda s:1-s.mean()),attr=("correct",lambda s:s.dropna().astype(float).mean())).reset_index()
pm=nl.groupby("mode").agg(lead=("lead","mean"),miss=("miss","mean"),attr=("attr","mean")); print(pm.round(3)); pm.to_csv(O+"table_mode.csv")
# paired tests
def paired(a,b,name,margin=None):
    d=a-b; w=stats.wilcoxon(a,b) if np.any(d!=0) else None
    t=stats.ttest_rel(a,b); ci=stats.t.interval(0.95,len(d)-1,loc=d.mean(),scale=stats.sem(d))
    s=f"{name}: mean diff {d.mean():.4f}, 95% CI [{ci[0]:.4f},{ci[1]:.4f}], t-test p={t.pvalue:.3g}, wilcoxon p={(w.pvalue if w else float('nan')):.3g}"
    if margin:
        se=stats.sem(d); p1=stats.t.sf((d.mean()+margin)/se,len(d)-1); p2=stats.t.cdf((d.mean()-margin)/se,len(d)-1)
        s+=f", TOST(margin {margin}) p={max(p1,p2):.3g}"
    print(s); return s
P=[]
P.append(paired(seedvals(0,"Cross-Traj W5","attr"),seedvals(0,"Uni-Traj W5","attr"),"attr Cross-Uni",0.01))
P.append(paired(seedvals(0,"Threshold+Var","lead_med"),seedvals(0,"Cross-Traj W5","lead_med"),"lead Thr+Var - Cross"))
P.append(paired(seedvals(0,"Threshold+Var","lead_med"),seedvals(0,"Uni-Traj W5","lead_med"),"lead Thr+Var - Uni"))
P.append(paired(seedvals(0,"Uni-Traj W5","lead_med"),seedvals(0,"Cross-Traj W5","lead_med"),"lead Uni - Cross"))
# pooled node-level attribution disagreement counts
a=N[N.det=="Cross-Traj W5"].reset_index(drop=True); b=N[N.det=="Uni-Traj W5"].reset_index(drop=True)
cu=((a.correct==True)&(b.correct==False)).sum(); uc=((a.correct==False)&(b.correct==True)).sum()
print("node-level attribution: cross right/uni wrong",cu,"uni right/cross wrong",uc,"n",len(a)); P.append(f"discordant {cu} {uc} n={len(a)}")
from scipy.stats import binomtest
if cu+uc>0: print("McNemar exact p",binomtest(cu,cu+uc,0.5).pvalue)
# common subset lead: Threshold-Mean vs Mean Cross
x=N[N.det=="Threshold-Mean"].set_index(["seed","node"]); y=N[N.det=="Mean Cross W12"].set_index(["seed","node"])
both=x.detected&y.detected
print("common detected nodes",both.sum(),"of",len(x))
cs=pd.DataFrame({"seed":x.index.get_level_values(0),"tm":x.lead.values,"mc":y.lead.values,"both":both.values})
cs=cs[cs.both].groupby("seed").agg(tm=("tm","median"),mc=("mc","median"))
print("common-subset seed-median lead: Threshold-Mean %.2f, Mean Cross %.2f"%(cs.tm.mean(),cs.mc.mean()))
P.append("common subset %d of %d: TM %.2f MC %.2f"%(both.sum(),len(x),cs.tm.mean(),cs.mc.mean()))
# miss-as-zero (restricted mean) per method
for d in ["Threshold-Mean","Mean Cross W12","Threshold+Var","Uni-Traj W5","Cross-Traj W5"]:
    z=N[N.det==d]; v=z.lead.fillna(0).groupby(z.seed).mean().mean(); print("miss=0 mean lead",d,round(v,2))
# stress
srows=[]
for lab in R[0]["stress"].keys():
    for d in ["Threshold+Var","CUSUM+Var","EWMA+Var","Uni-Traj W5","Cross-Traj W5"]:
        srows.append(dict(cond=lab,method=d,lead=np.nanmean(seedvals(0,d,"lead_med",lab)),miss=100*np.mean(seedvals(0,d,"miss",lab)),
                          fa=100*np.mean(seedvals(0,d,"fa",lab)),attr=100*np.nanmean(seedvals(0,d,"attr",lab)) if "Traj" in d else np.nan))
S=pd.DataFrame(srows); S.to_csv(O+"table_stress.csv",index=False); print(S.round(2).to_string(index=False))
# open set
for d in ["Threshold+Var","Uni-Traj W5","Cross-Traj W5"]:
    det=np.mean([r["open_set"][d]["detected"] for r in R]); print("open-set detected",d,round(det,3))
    if d!="Threshold+Var":
        fl=np.nanmean([r["open_set"][d]["forced_label"] for r in R],axis=0); print("  forced labels fan/power/thermal",fl.round(3))
open(O+"paired.txt","w").write("\n".join(P))
