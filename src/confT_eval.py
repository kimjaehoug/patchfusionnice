"""Middle-testing tertile (not used in design): target-only (tfrac=1) vs main design vs thinning vs backbone/generic.
Five-seed ensembles, hospital cluster bootstrap. Writes results/confT_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
from eval_hallym import boot_idx
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

S, EPS = (2, 3, 4, 5, 6), ["y24", "y48", "y168", "yhosp"]
e = T.load("eicu_dyn")
site = e["site"]; crit = [7, 13, 14, 16, 20, 21]
hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
t = np.full(len(hr), None, object); t[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
M = t == "mid"
G = {"backbone": "W_base", "generic": "W_gen", "main": "W_propM", "target_only": "W_propTM", "thinning": "W_thinM"}
P = {g: np.mean([np.load(f"{T.ROOT}/results/preds/{tg}_s{s}.npz")["ext"] for s in S], 0)[M] for g, tg in G.items()}
cl = site[M]; idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
PAIRS = [("target_only", "thinning"), ("target_only", "main"), ("target_only", "backbone"), ("thinning", "backbone"), ("main", "thinning")]
out = {"n": int(M.sum()), "hospitals": int(len(np.unique(cl)))}
c = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]{'*' if v[1] > 0 or v[2] < 0 else ' '}"
print(f"middle tertile: n={out['n']} hospitals={out['hospitals']}")
for j, ep in enumerate(EPS):
    y = e[ep][M]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
    r = {g: [A(y, P[g][:, j]), AP(y, P[g][:, j]), float(y.sum() / P[g][:, j].sum())] for g in G}
    for a_, b_ in PAIRS:
        for mn, f in (("auroc", A), ("auprc", AP)):
            bs = boot_metric(f, y, P[a_][:, j], ii) - boot_metric(f, y, P[b_][:, j], ii)
            r[f"{a_}-{b_}_{mn}"] = [f(y, P[a_][:, j]) - f(y, P[b_][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
    out[ep] = r
    print(f"\n[{ep}] " + "  ".join(f"{g} {r[g][0]:.3f}/{r[g][1]:.3f} O/E {r[g][2]:.2f}" for g in G))
    for a_, b_ in PAIRS:
        print(f"   {a_}-{b_}: AUROC {c(r[f'{a_}-{b_}_auroc'])}  AUPRC {c(r[f'{a_}-{b_}_auprc'])}")
json.dump(out, open(f"{T.ROOT}/results/confT_W.json", "w"), indent=1, default=float)
