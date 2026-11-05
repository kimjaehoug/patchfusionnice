"""Pre-specified comparison (paper/notes/lgbm_reobs_prereg.md): re-observation applied to LightGBM vs the PatchFusion
encoder. Three-seed ensembles (seeds 2-4); low-testing eICU-CRD (hospital clusters) and Hallym (patient clusters),
1,000 bootstrap resamples; 24-h / 48-h AUROC and AUPRC, 24-h O/E. Writes results/lgbm_reobs_W.json."""
import json
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

S = (2, 3, 4)
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
LG = {"lgbm_src": "F_lgbm", "lgbm_rate": "L_rate{t}", "lgbm_three": "L_three{t}"}
PF = {"pf_src": "W_base", "pf_rate": "W_thin{t}", "pf_three": "W_propT{t}"}
PAIRS = [("lgbm_rate", "lgbm_src"), ("lgbm_three", "lgbm_src"), ("pf_rate", "pf_src"), ("pf_three", "pf_src"),
         ("pf_rate", "lgbm_rate"), ("pf_three", "lgbm_three"), ("pf_three", "lgbm_rate"), ("pf_rate", "lgbm_three"),
         ("pf_src", "lgbm_src")]
out = {}
for t in ("E", "H"):
    key = "ext" if t == "E" else "hal"
    P = {g: np.mean([np.load(f"{T.ROOT}/results/preds/{tag.format(t=t)}_s{s}.npz")[key] for s in S], 0) for g, tag in LG.items()}
    if t == "E":
        P = {g: p[low] for g, p in P.items()}
        P.update({g: np.mean([np.load(f"{T.ROOT}/results/preds/{tag.format(t=t)}_s{s}.npz")["ext"] for s in S], 0)[low] for g, tag in PF.items()})
        ds, m, cl = e, low, e["site"][low]
    else:
        P.update({g: np.mean([frozen_preds(f"{tag.format(t=t)}_s{s}", h, st, "cuda") for s in S], 0) for g, tag in PF.items()})
        ds, m, cl = h, np.ones(len(h["y24"]), bool), h["subject_id"]
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    R = {}
    for j, ep in enumerate(["y24", "y48"]):
        y = ds[ep][m]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        r = {g: [A(y, p[:, j]), AP(y, p[:, j]), float(y.sum() / p[:, j].sum())] for g, p in P.items()}
        for a_, b_ in PAIRS:
            for mn, f in (("auroc", A), ("auprc", AP)):
                bs = boot_metric(f, y, P[a_][:, j], ii) - boot_metric(f, y, P[b_][:, j], ii)
                r[f"{a_}-{b_}_{mn}"] = [f(y, P[a_][:, j]) - f(y, P[b_][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
        R[ep] = r
        c = lambda k: f"{r[k][0]:+.3f} [{r[k][1]:+.3f},{r[k][2]:+.3f}]"
        print(f"\n[{t} {ep}] " + "  ".join(f"{g} {v[0]:.3f}/{v[1]:.3f}" + (f" (O/E {v[2]:.2f})" if ep == "y24" else "") for g, v in r.items() if isinstance(v, list) and len(v) == 3))
        for a_, b_ in PAIRS:
            print(f"   {a_:>10} - {b_:<10} AUROC {c(f'{a_}-{b_}_auroc')}  AUPRC {c(f'{a_}-{b_}_auprc')}")
    out[t] = R
json.dump(out, open(f"{T.ROOT}/results/lgbm_reobs_W.json", "w"), indent=1, default=float)
print("written results/lgbm_reobs_W.json")
