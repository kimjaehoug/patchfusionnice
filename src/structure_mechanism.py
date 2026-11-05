"""Why structure: calibration and discrimination of the three-scale profile (target only) vs rate-level imagination by
laboratory testing status and window acuity, at low-testing eICU-CRD hospitals and (contrast) the Hallym hospitals.
Five-seed ensembles; existing predictions. Writes results/structure_mechanism_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
import obsprofile as PR
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from sklearn.metrics import roc_auc_score as A
S = (2, 3, 4, 5, 6)
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn")
rs = np.random.RandomState(0); i0 = rs.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i0], d["yhosp"][i0])
def calib_int(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)); a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp))); a += (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-9)
    return a
out = {}
low = low_tertile(e)
SRC = {"eICU-low": (e, low, "W_propTE", "W_thinE", "ext"), "Hallym": (h, np.ones(len(h["y24"]), bool), "W_propTH", "W_thinH", None)}
for name, (ds, m, t3, tr, key) in SRC.items():
    if key:
        P3 = np.mean([np.load(f"{T.ROOT}/results/preds/{t3}_s{s}.npz")[key] for s in S], 0)[m][:, 0]
        PR_ = np.mean([np.load(f"{T.ROOT}/results/preds/{tr}_s{s}.npz")[key] for s in S], 0)[m][:, 0]
    else:
        P3 = np.mean([frozen_preds(f"{t3}_s{s}", ds, st, "cuda") for s in S], 0)[:, 0]
        PR_ = np.mean([frozen_preds(f"{tr}_s{s}", ds, st, "cuda") for s in S], 0)[:, 0]
    x, mm, y = ds["x"][m], ds["m"][m], ds["y24"][m].astype(float)
    tested = (mm[:, :, PR.LAB] > 0).any((1, 2))
    with np.errstate(all="ignore"):
        aw = np.nan_to_num(np.nanmean(acu(x), 1), nan=np.nan)
    at = pd.qcut(pd.Series(aw).rank(method="first"), 3, labels=["low acuity", "mid acuity", "high acuity"]).values
    R = {}
    strata = {"all": np.ones(len(y), bool), "labs tested": tested, "labs not tested": ~tested}
    for lab in ("low acuity", "mid acuity", "high acuity"):
        strata[lab] = at == lab
    print(f"\n[{name}] share tested by acuity third:", {lab: round(float(tested[at == lab].mean()), 3) for lab in ("low acuity", "mid acuity", "high acuity")}, flush=True)
    for sn, g in strata.items():
        if g.sum() < 100 or y[g].sum() < 5:
            continue
        r = {"n": int(g.sum()), "deaths": int(y[g].sum())}
        for mn, p in (("three", P3), ("rate", PR_)):
            r[mn] = {"oe": float(y[g].sum() / p[g].sum()), "intercept": float(calib_int(y[g], p[g])), "auroc": float(A(y[g], p[g])) if 0 < y[g].sum() < g.sum() else None}
        R[sn] = r
        print(f"  {sn:16s} n {r['n']:6d} deaths {r['deaths']:4d} | O/E three {r['three']['oe']:.2f} rate {r['rate']['oe']:.2f} | intercept three {r['three']['intercept']:+.2f} rate {r['rate']['intercept']:+.2f} | AUROC three {r['three']['auroc']:.3f} rate {r['rate']['auroc']:.3f}", flush=True)
    out[name] = R
json.dump(out, open(f"{T.ROOT}/results/structure_mechanism_W.json", "w"), indent=1, default=float)
print("written results/structure_mechanism_W.json")
