import json, numpy as np, pandas as pd, os
import bench37 as b
b.MODES = ["ovA", "ovB", "ovC"]
b.SPEC_CROSS = {"ovA": ["temperature", "cpu_util"], "ovB": ["temperature", "fan_speed"], "ovC": ["cpu_util", "fan_speed"]}
b.SPEC_MEAN = b.SPEC_CROSS
b.SPEC_UNI = {"ovA": ["temperature"], "ovB": ["fan_speed"], "ovC": ["cpu_util"]}
res = []
for s in range(30):
    r, nd = b.run_seed(s, extras=False); res.append(r["main"])
rng = np.random.default_rng(3)
rows = []
for d in ["Threshold+Var", "Uni-Traj W5", "Cross-Traj W5"]:
    f = lambda k: np.array([r[d].get(k, np.nan) for r in res], float)
    lead = f("lead_med"); m = [rng.choice(lead[~np.isnan(lead)], 30).mean() for _ in range(3000)]
    rows.append(dict(method=d, lead=np.nanmean(lead), lo=np.percentile(m, 2.5), hi=np.percentile(m, 97.5), miss=100*np.nanmean(f("miss")),
                     fa=100*np.nanmean(f("fa")), attr=100*np.nanmean(f("attr")) if not np.all(np.isnan(f("attr"))) else np.nan))
T = pd.DataFrame(rows); T.to_csv(b.OUT + "/table_overlap.csv", index=False); print(T.round(2).to_string(index=False))
from scipy import stats
au = np.array([r["Uni-Traj W5"]["attr"] for r in res]); ac = np.array([r["Cross-Traj W5"]["attr"] for r in res])
d = ac - au; print("attr Cross-Uni diff %.4f CI" % d.mean(), stats.t.interval(0.95, 29, loc=d.mean(), scale=stats.sem(d)), "wilcoxon p", stats.wilcoxon(ac, au).pvalue)
