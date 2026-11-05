"""Exploratory (pre-specified median split): does the proposed profile beat rate-matched thinning at hospitals whose
label-free test-selection slope on acuity departs most from the development hospital? eICU low and middle testing
tertiles; per-hospital divergence = mean over laboratory variables of |slope_site - slope_dev| (variables with >=20
tested and >=20 untested windows). Five-seed ensembles; hospital cluster bootstrap. Writes results/selslope_W.json."""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
import train as T
import obsprofile as PR
from eval_hallym import boot_idx
from bootstrap_par import boot_metric

S, EPS = (2, 3, 4, 5, 6), ["y24", "y48", "y168", "yhosp"]
rng = np.random.RandomState(0)
d = T.load("mimic_dyn"); i = rng.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i], d["yhosp"][i]); dv = PR.estimate(d["x"][i], d["m"][i], acu)
e = T.load("eicu_dyn"); site = e["site"]
crit = [7, 13, 14, 16, 20, 21]
hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
tert = np.full(len(hr), None, object); tert[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values

div = {}
for s in np.unique(site[np.isin(tert, ["low", "mid"])]):
    k = site == s
    tp = PR.estimate(e["x"][k], e["m"][k], acu, beta=dv.beta, n=int(k.sum()))
    gaps = []
    for c in PR.LAB:
        y = tp._tested[c]
        if y.sum() >= 20 and (~y).sum() >= 20:
            gaps.append(abs(LogisticRegression(max_iter=1000).fit(tp._aw[:, None], y).coef_[0, 0] - dv.beta_w[c]))
    div[s] = float(np.mean(gaps)) if gaps else np.nan
print(f"divergence estimated for {len(div)} hospitals; {sum(np.isnan(v) for v in div.values())} without usable labs")

out = {"divergence": {str(k): v for k, v in div.items()}}
c = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]{'*' if v[1] > 0 or v[2] < 0 else ' '}"
for tn, prop, thin in (("low", "W_propE", "W_thinE"), ("mid", "W_propM", "W_thinM")):
    M = tert == tn
    P = {g: np.mean([np.load(f"{T.ROOT}/results/preds/{tg}_s{s}.npz")["ext"] for s in S], 0)[M] for g, tg in (("prop", prop), ("thin", thin))}
    sm = site[M]; dvv = np.array([div.get(s, np.nan) for s in sm])
    hs = np.unique(sm); hv = np.array([div.get(s, np.nan) for s in hs]); med = np.nanmedian(hv)
    grp = {"high_div": dvv > med, "low_div": dvv <= med}
    print(f"\n=== {tn} tertile: {len(hs)} hospitals, median divergence {med:.2f} (range {np.nanmin(hv):.2f}-{np.nanmax(hv):.2f})")
    R = {"median": float(med)}
    BS = {}
    for gname, g in grp.items():
        cl = sm[g]; idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
        R[gname] = {"hospitals": int(len(np.unique(cl))), "n": int(g.sum())}
        for j, ep in enumerate(EPS):
            y = e[ep][M][g]; ii = [x for x in idx if 0 < y[x].sum() < len(x)]
            for mn, f in (("auroc", A), ("auprc", AP)):
                bs = boot_metric(f, y, P["prop"][g, j], ii) - boot_metric(f, y, P["thin"][g, j], ii)
                BS[(gname, ep, mn)] = bs
                R[gname][f"{ep}_{mn}"] = [f(y, P["prop"][g, j]) - f(y, P["thin"][g, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
            R[gname][f"{ep}_values"] = {k: [A(y, P[k][g, j]), AP(y, P[k][g, j])] for k in P}
    for j, ep in enumerate(EPS):
        for mn in ("auroc", "auprc"):
            a, b = R["high_div"][f"{ep}_{mn}"][0], R["low_div"][f"{ep}_{mn}"][0]
            dd = BS[("high_div", ep, mn)][:min(len(BS[("high_div", ep, mn)]), len(BS[("low_div", ep, mn)]))] - BS[("low_div", ep, mn)][:min(len(BS[("high_div", ep, mn)]), len(BS[("low_div", ep, mn)]))]
            R[f"contrast_{ep}_{mn}"] = [a - b, *np.percentile(dd, [2.5, 97.5]).tolist()]
        print(f"[{ep}] prop-thin AUROC  high-div({R['high_div']['hospitals']}h) {c(R['high_div'][f'{ep}_auroc'])}  low-div({R['low_div']['hospitals']}h) {c(R['low_div'][f'{ep}_auroc'])}  contrast {c(R[f'contrast_{ep}_auroc'])}")
        print(f"      prop-thin AUPRC  high-div {c(R['high_div'][f'{ep}_auprc'])}  low-div {c(R['low_div'][f'{ep}_auprc'])}  contrast {c(R[f'contrast_{ep}_auprc'])}")
    out[tn] = R
json.dump(out, open(f"{T.ROOT}/results/selslope_W.json", "w"), indent=1, default=float)
print("\nwritten results/selslope_W.json")
