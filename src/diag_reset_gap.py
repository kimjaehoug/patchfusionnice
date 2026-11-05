"""Why does the no-reset encoder transport better than PatchFusion (source-only, eICU-CRD)? Split the AUPRC/AUROC gap by
testing tertile, landmark, observation density and pooled vs within-hospital ranking (hospital-cluster bootstrap)."""
import numpy as np, pandas as pd, train as T
from eval_hallym import boot_idx
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
from link_eval import low_tertile
e = T.load("eicu_dyn"); S = (2, 3, 4)
P = {g: np.mean([np.load(f"{T.ROOT}/results/preds/{p}_s{s}.npz")["ext"] for s in S], 0) for g, p in [("reset", "F_base"), ("noreset", "E_noreset"), ("lstm", "E_lstm")]}
site = e["site"]; j = 0; y = e["y24"]
crit = [7, 13, 14, 16, 20, 21]
hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
tert = np.full(len(hr), "small", object); tert[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values.astype(str)
dens = e["m"][:, :, :22].mean((1, 2)); dq = pd.qcut(dens, 3, labels=["sparse", "medium", "dense"]).astype(str)
idx = list(boot_idx(site, 300, np.random.RandomState(0)))
def gap(mask, f=AP, g="noreset"):
    ii = np.where(mask)[0]; yy = y[ii]
    pt = f(yy, P[g][ii, j]) - f(yy, P["reset"][ii, j])
    bs = []
    for b in idx:
        b = b[mask[b]]
        if 0 < y[b].sum() < len(b): bs.append(f(y[b], P[g][b, j]) - f(y[b], P["reset"][b, j]))
    return f"{pt:+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]  (n={len(ii)}, deaths={int(yy.sum())}, reset AUPRC {f(yy, P['reset'][ii, j]):.3f})"
print("ALL  AUPRC", gap(np.ones(len(y), bool)), "\n     AUROC", gap(np.ones(len(y), bool), A))
for t in ["low", "mid", "high", "small"]: print("tertile", t, gap(tert == t))
for L in sorted(set(e["landmark"])): print("landmark", L, gap(e["landmark"] == L))
for q in ["sparse", "medium", "dense"]: print("density", q, gap(dq == q))
# pooled vs within-hospital ranking (24-h AUROC averaged over hospitals with >=5 deaths, weighted by deaths)
def within(g):
    r = []
    for s_ in np.unique(site):
        i = site == s_
        if y[i].sum() >= 5 and (1 - y[i]).sum() >= 5: r.append((A(y[i], P[g][i, j]), y[i].sum()))
    r = np.array(r); return (r[:, 0] * r[:, 1]).sum() / r[:, 1].sum(), len(r)
for g in P: print("within-hospital AUROC", g, "%.4f (hospitals %d)" % within(g))
# lstm vs reset too
print("LSTM-reset ALL AUPRC", gap(np.ones(len(y), bool), AP, "lstm"))
