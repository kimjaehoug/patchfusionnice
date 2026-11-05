"""Label-free test-selection slope divergence (as selslope_eval.py) for the four Hallym hospitals, placed against the
eICU-CRD low/middle-testing distribution. No outcome labels of any target are used. Writes results/selslope_hallym_W.json."""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import train as T
import obsprofile as PR

rng = np.random.RandomState(0)
d = T.load("mimic_dyn"); i = rng.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i], d["yhosp"][i]); dv = PR.estimate(d["x"][i], d["m"][i], acu)
h = T.load("hallym_dyn"); site = h["site"]
out = {}
for s in np.unique(site):
    k = site == s
    tp = PR.estimate(h["x"][k], h["m"][k], acu, beta=dv.beta, n=int(k.sum()))
    gaps, used = [], []
    for c in PR.LAB:
        y = tp._tested[c]
        if y.sum() >= 20 and (~y).sum() >= 20:
            sl = LogisticRegression(max_iter=1000).fit(tp._aw[:, None], y).coef_[0, 0]
            gaps.append(abs(sl - dv.beta_w[c])); used.append([c, float(sl), float(dv.beta_w[c])])
    out[str(s)] = {"divergence": float(np.mean(gaps)) if gaps else None, "n_windows": int(k.sum()), "labs": used}
    print(s, out[str(s)]["divergence"], len(used), "labs", flush=True)
ev = json.load(open(f"{T.ROOT}/results/selslope_W.json"))
ed = np.array([v for v in ev["divergence"].values() if v == v])
out["eicu_low_mid"] = {"n": int(len(ed)), "median": float(np.median(ed)), "p90": float(np.percentile(ed, 90)), "max": float(ed.max())}
for s in np.unique(site):
    v = out[str(s)]["divergence"]
    if v is not None:
        out[str(s)]["eicu_percentile"] = float((ed < v).mean() * 100)
print(out["eicu_low_mid"], {s: out[str(s)].get("eicu_percentile") for s in np.unique(site)})
json.dump(out, open(f"{T.ROOT}/results/selslope_hallym_W.json", "w"), indent=1)
