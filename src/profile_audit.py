"""Label-free audit: do differences between a hospital's observation profile and the development profile predict the
frozen backbone's discrimination at that hospital? eICU hospitals with >= 20 deaths within 24 h. Predictors: (i) the
three-scale profile difference (site: absent variables; stay: selection intercept and slope, stream presence;
patch: laboratory, documentation and vital-sign cadence), (ii) rate-only difference (log hourly-rate ratio per
variable, the information used by rate-matched thinning), (iii) both. Ridge regression (inner CV for the penalty),
repeated 10-fold CV; out-of-fold R2 and Spearman; hospital bootstrap of the R2 difference. Writes results/audit_W.json."""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import train as T
import obsprofile as PR
from regime import VARS, VIT

rng = np.random.RandomState(0)
d = T.load("mimic_dyn"); i = rng.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i], d["yhosp"][i]); dv = PR.estimate(d["x"][i], d["m"][i], acu)
r_dev = PR.obs_rate(d["m"][i])
e = T.load("eicu_dyn"); site = e["site"]
P = np.mean([np.load(f"{T.ROOT}/results/preds/W_base_s{s}.npz")["ext"] for s in (2, 3, 4, 5, 6)], 0)
rows = []
for s in np.unique(site):
    k = site == s
    if k.sum() < 200 or e["y24"][k].sum() < 20:
        continue
    tp = PR.fit_window(PR.estimate(e["x"][k], e["m"][k], acu, beta=dv.beta, n=int(k.sum())), dv)
    slope = np.zeros(len(PR.LAB))
    for j, c in enumerate(PR.LAB):
        y = tp._tested[c]
        slope[j] = (LogisticRegression(max_iter=1000).fit(tp._aw[:, None], y).coef_[0, 0] - dv.beta_w[c]) if y.sum() >= 20 and (~y).sum() >= 20 else 0.0
    cr = lambda a, b: np.log((a + 1e-3) / (b + 1e-3))
    absent = np.zeros(len(VARS)); absent[tp.absent] = 1
    three = np.concatenate([absent, tp.alpha_w[PR.LAB] - dv.alpha_w[PR.LAB], slope, cr(tp.rho[PR.DOC], dv.rho[PR.DOC]),
                            cr(tp.cad[PR.LAB], dv.cad[PR.LAB]), cr(tp.dcad[PR.DOC], dv.dcad[PR.DOC]), cr(tp.vcad[VIT], dv.vcad[VIT])])
    rate = cr(PR.obs_rate(e["m"][k]), r_dev)
    rows.append({"site": str(s), "n": int(k.sum()), "deaths24": int(e["y24"][k].sum()),
                 "auroc24": float(roc_auc_score(e["y24"][k], P[k, 0])), "aurochosp": float(roc_auc_score(e["yhosp"][k], P[k, 3])),
                 "three": three.tolist(), "rate": rate.tolist()})
print(f"{len(rows)} hospitals; backbone 24h AUROC range {min(r['auroc24'] for r in rows):.3f}-{max(r['auroc24'] for r in rows):.3f}")
X3, XR = np.array([r["three"] for r in rows]), np.array([r["rate"] for r in rows])
SETS = {"three_scale": X3, "rate_only": XR, "both": np.c_[X3, XR]}
out = {"hospitals": len(rows), "rows": rows}
for tgt in ("auroc24", "aurochosp"):
    y = np.array([r[tgt] for r in rows]); w = np.array([r["n"] for r in rows], float)
    oof = {g: np.zeros((10, len(y))) for g in SETS}
    for rep in range(10):
        for tr, te in KFold(10, shuffle=True, random_state=rep).split(y):
            for g, X in SETS.items():
                mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(X[tr], y[tr])
                oof[g][rep, te] = mdl.predict(X[te])
    res = {}
    sst = ((y - y.mean()) ** 2).sum()
    for g in SETS:
        r2 = [1 - ((y - oof[g][rep]) ** 2).sum() / sst for rep in range(10)]
        res[g] = {"cv_r2": float(np.mean(r2)), "cv_r2_sd": float(np.std(r2)), "spearman": float(spearmanr(y, oof[g].mean(0))[0])}
    e3, er = (y - oof["three_scale"].mean(0)) ** 2, (y - oof["rate_only"].mean(0)) ** 2
    bs = []
    for b in range(2000):
        j = rng.randint(0, len(y), len(y)); ss = ((y[j] - y[j].mean()) ** 2).sum()
        bs.append((er[j].sum() - e3[j].sum()) / ss)
    res["r2_three_minus_rate"] = [float((er.sum() - e3.sum()) / sst), *np.percentile(bs, [2.5, 97.5]).tolist()]
    out[tgt] = res
    print(f"[{tgt}] " + "  ".join(f"{g}: CV R2 {res[g]['cv_r2']:+.3f} (sd {res[g]['cv_r2_sd']:.3f}) rho {res[g]['spearman']:+.2f}" for g in SETS)
          + f" | R2 three-rate {res['r2_three_minus_rate'][0]:+.3f} [{res['r2_three_minus_rate'][1]:+.3f},{res['r2_three_minus_rate'][2]:+.3f}]", flush=True)
json.dump(out, open(f"{T.ROOT}/results/audit_W.json", "w"), indent=1)
print("written results/audit_W.json")
