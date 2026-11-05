"""Encoder smoke: source-only variants vs the reset (XRbase) and no-reset (XNbase) backbones, same protocol (30%, 10 epochs).
eICU-CRD all hospitals (hospital-cluster bootstrap of the difference to XRbase) and Hallym; seed ensembles."""
import sys, os, numpy as np, train as T
from eval_hallym import frozen_preds, boot_idx
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
S = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2, 3]
G = ["XRbase", "XNbase", "ES", "EH", "ESH", "EL4", "EQ", "EW"] + sys.argv[2:]
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn"); e = T.load("eicu_dyn")
P, Q = {}, {}
for g in G:
    t = [f"{g}_s{s}" for s in S if os.path.exists(f"{T.ROOT}/results/preds/{g}_s{s}.npz")]
    if len(t) < len(S): print(g, "incomplete", t); continue
    P[g] = np.mean([np.load(f"{T.ROOT}/results/preds/{x}.npz")["ext"] for x in t], 0)
    Q[g] = np.mean([frozen_preds(x, h, st, "cuda") for x in t], 0)
idx = list(boot_idx(e["site"], 300, np.random.RandomState(0)))
for j, ep in [(0, "y24"), (2, "y168")]:
    y, yh = e[ep], h[ep]; ok = [i for i in idx if 0 < y[i].sum() < len(i)]
    print(f"== {ep}")
    for g in P:
        bs = np.array([AP(y[i], P[g][i, j]) - AP(y[i], P["XRbase"][i, j]) for i in ok]) if g != "XRbase" else np.zeros(1)
        print(f"  {g:7s} eICU {A(y, P[g][:, j]):.3f}/{AP(y, P[g][:, j]):.3f}  dAUPRC vs reset {AP(y, P[g][:, j]) - AP(y, P['XRbase'][:, j]):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]  | Hallym {A(yh, Q[g][:, j]):.3f}/{AP(yh, Q[g][:, j]):.3f}", flush=True)
