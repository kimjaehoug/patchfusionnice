"""Hallym: W + 4 attention layers + Hallym imagination vs W (2 layers) + imagination vs 2.53M + imagination (3 seeds)."""
import numpy as np, train as T
from eval_hallym import frozen_preds, boot_idx
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
S = (2, 3, 4); d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn")
G = {"WL4_propH": "W+4L +imag", "W_propH": "W +imag", "F_propH": "2.53M +imag", "W_noresetPropH": "no-reset W +imag"}
P = {g: np.mean([frozen_preds(f"{g}_s{s}", h, st, "cuda") for s in S], 0) for g in G}
idx = list(boot_idx(h["subject_id"], 500, np.random.RandomState(0)))
for j, ep in enumerate(["y24", "y48", "y168", "yhosp"]):
    y = h[ep]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
    print(f"[Hallym {ep}] " + "  ".join(f"{G[g]} {A(y, P[g][:, j]):.3f}/{AP(y, P[g][:, j]):.3f}" for g in G))
    for a, b in (("WL4_propH", "F_propH"), ("WL4_propH", "W_propH"), ("W_noresetPropH", "WL4_propH")):
        out = []
        for mn, f in (("AUROC", A), ("AUPRC", AP)):
            bs = np.array([f(y[i], P[a][i, j]) - f(y[i], P[b][i, j]) for i in ii])
            out.append(f"{mn} {f(y, P[a][:, j]) - f(y, P[b][:, j]):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
        print(f"     {G[a]} - {G[b]}: " + "  ".join(out), flush=True)
