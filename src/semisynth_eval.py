"""Evaluation for paper/notes/structure_semisynth_prereg.md. Writes results/semisynth_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
import obsprofile as PR
from eval_hallym import frozen_preds, boot_idx
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); S = (2, 3, 4)
groups = d["subject_id"]; us = np.unique(groups)
tr = np.where(np.isin(groups, np.random.RandomState(7).choice(us, int(0.7 * len(us)), replace=False)))[0]
acu = PR.BlockAcuity().fit(d["x"][tr], d["yhosp"][tr])
out = {}
for t in ("A", "B"):
    ds = T.load(f"syn{t}_dyn")
    P = {g: np.mean([frozen_preds(f"{tag}_s{s}", ds, st, "cuda") for s in S], 0) for g, tag in
         (("backbone", "SS_base"), ("rate", f"SS_rate{t}"), ("three", f"SS_three{t}"))}
    idx = list(boot_idx(ds["subject_id"], 1000, np.random.RandomState(0))); R = {}
    for j, ep in ((0, "y24"), (1, "y48")):
        y = ds[ep]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        r = {g: [A(y, p[:, j]), AP(y, p[:, j]), float(y.sum() / p[:, j].sum())] for g, p in P.items()}
        for a, b in (("three", "rate"), ("rate", "backbone"), ("three", "backbone")):
            for mn, f in (("auroc", A), ("auprc", AP)):
                bs = boot_metric(f, y, P[a][:, j], ii) - boot_metric(f, y, P[b][:, j], ii)
                r[f"{a}-{b}_{mn}"] = [f(y, P[a][:, j]) - f(y, P[b][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
        R[ep] = r
        c = lambda k: f"{r[k][0]:+.4f} [{r[k][1]:+.4f},{r[k][2]:+.4f}]"
        print(f"[target {t} {ep}] " + "  ".join(f"{g} {v[0]:.4f}/{v[1]:.4f} O/E {v[2]:.2f}" for g, v in r.items() if isinstance(v, list) and len(v) == 3))
        print(f"    three-rate AUROC {c('three-rate_auroc')} AUPRC {c('three-rate_auprc')} | rate-backbone AUROC {c('rate-backbone_auroc')} | three-backbone AUROC {c('three-backbone_auroc')}", flush=True)
    with np.errstate(all="ignore"):
        aw = np.nanmean(acu(ds["x"]), 1)
    at = pd.qcut(pd.Series(aw).rank(method="first"), 3, labels=["low", "mid", "high"]).values
    y = ds["y24"].astype(float); R["by_acuity"] = {}
    for lab in ("low", "mid", "high"):
        g = at == lab
        R["by_acuity"][lab] = {k: {"oe": float(y[g].sum() / P[k][g, 0].sum()), "auroc": float(A(y[g], P[k][g, 0]))} for k in P}
        print(f"    acuity {lab:4s} deaths {int(y[g].sum()):4d} | O/E three {R['by_acuity'][lab]['three']['oe']:.2f} rate {R['by_acuity'][lab]['rate']['oe']:.2f} backbone {R['by_acuity'][lab]['backbone']['oe']:.2f} "
              f"| AUROC three {R['by_acuity'][lab]['three']['auroc']:.3f} rate {R['by_acuity'][lab]['rate']['auroc']:.3f}", flush=True)
    out[t] = R
json.dump(out, open(f"{T.ROOT}/results/semisynth_W.json", "w"), indent=1, default=float)
print("written results/semisynth_W.json")
