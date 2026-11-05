"""Paired patient-cluster bootstrap of formal (full-data) models at the Hallym hospitals, all available seeds."""
import os, numpy as np, train as T
from eval_hallym import frozen_preds, boot_idx
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn")
S = [s for s in (2, 3, 4, 5, 6)]
G = {}
for g in ["base", "propH", "propE", "selH"]:
    t = [f"F_{g}_s{s}" for s in S if os.path.exists(f"{T.ROOT}/results/preds/F_{g}_s{s}.npz")]
    G[g] = np.stack([frozen_preds(x, h, st, "cuda") for x in t])
np.savez(f"{T.ROOT}/results/hallym_formal_preds5.npz", **G)
idx = list(boot_idx(h["subject_id"], 1000, np.random.RandomState(0)))
for j, ep in enumerate(["y24", "y48", "y168", "yhosp"]):
    y = h[ep]; ok = [i for i in idx if 0 < y[i].sum() < len(i)]; E = {g: v.mean(0)[:, j] for g, v in G.items()}
    print(f"== {ep} n_seeds={ {g: len(v) for g, v in G.items()} }  per-seed propH {[round(A(y, p[:, j]), 3) for p in G['propH']]} propE {[round(A(y, p[:, j]), 3) for p in G['propE']]}")
    for a, b in [("propH", "propE"), ("propH", "selH"), ("propH", "base"), ("propE", "base")]:
        for nm, f in (("AUROC", A), ("AUPRC", AP)):
            bs = np.array([f(y[i], E[a][i]) - f(y[i], E[b][i]) for i in ok])
            print(f"  {a}-{b} {nm} {f(y, E[a]) - f(y, E[b]):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]", end="")
        print()
