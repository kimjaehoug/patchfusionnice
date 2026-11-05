"""Interim: new design (W_p2*) vs main, thinning, backbone on completed seeds only (same seeds for all)."""
import os, sys
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
EPS = ["y24", "y48", "y168", "yhosp"]
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"])
c = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]{'*' if v[1] > 0 or v[2] < 0 else ' '}"
for tgt in sys.argv[1:]:
    S = [s for s in (2, 3, 4, 5, 6) if os.path.exists(f"{T.ROOT}/results/preds/W_p2{tgt}_s{s}.npz")]
    G = {"backbone": "W_base", "thinning": f"W_thin{tgt}", "main": f"W_prop{tgt}", "new": f"W_p2{tgt}"}
    if tgt == "H":
        ds = T.load("hallym_dyn"); M = np.ones(len(ds["y24"]), bool); cl = ds["subject_id"]
        P = {g: np.mean([frozen_preds(f"{t}_s{s}", ds, st, "cuda") for s in S], 0) for g, t in G.items()}
    else:
        ds = T.load("eicu_dyn"); M = low_tertile(ds); cl = ds["site"][M]
        P = {g: np.mean([np.load(f"{T.ROOT}/results/preds/{t}_s{s}.npz")["ext"] for s in S], 0)[M] for g, t in G.items()}
    idx = list(boot_idx(cl, 500, np.random.RandomState(0)))
    print(f"\n===== {tgt} seeds {S}")
    for j, ep in enumerate(EPS):
        y = ds[ep][M]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        print(f"[{ep}] " + "  ".join(f"{g} {A(y, P[g][:, j]):.3f}/{AP(y, P[g][:, j]):.3f} O/E {y.sum() / P[g][:, j].sum():.2f}" for g in G))
        for b in ("thinning", "main", "backbone"):
            r = {}
            for mn, f in (("AUROC", A), ("AUPRC", AP)):
                bs = boot_metric(f, y, P["new"][:, j], ii) - boot_metric(f, y, P[b][:, j], ii)
                r[mn] = [f(y, P["new"][:, j]) - f(y, P[b][:, j]), *np.percentile(bs, [2.5, 97.5])]
            print(f"     new-{b:9s} AUROC {c(r['AUROC'])}  AUPRC {c(r['AUPRC'])}")
