"""Where does the three-scale profile beat rate-level imagination at low-testing eICU-CRD hospitals: windows with or
without Glasgow Coma Scale / urine-output streams, and with or without laboratory tests (five-seed ensembles, target
profiles only; hospital-cluster bootstrap, 1,000). Writes results/stream_strata_W.json."""
import json
import numpy as np
import train as T
import obsprofile as PR
from eval_hallym import boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A
S = (2, 3, 4, 5, 6)
e = T.load("eicu_dyn"); low = low_tertile(e)
P3 = np.mean([np.load(f"{T.ROOT}/results/preds/W_propTE_s{s}.npz")["ext"] for s in S], 0)[low][:, 0]
PR_ = np.mean([np.load(f"{T.ROOT}/results/preds/W_thinE_s{s}.npz")["ext"] for s in S], 0)[low][:, 0]
m = e["m"][low]; y = e["y24"][low].astype(float); cl = e["site"][low]
gcs, uo = (m[:, :, PR.VARS.index("gcs")] > 0).any(1), (m[:, :, PR.VARS.index("uo")] > 0).any(1)
labs = (m[:, :, PR.LAB] > 0).any((1, 2))
idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
STR = {"all": np.ones(len(y), bool), "GCS and UO streams present": gcs & uo, "one stream absent": gcs ^ uo,
       "both streams absent": ~gcs & ~uo, "labs tested": labs, "labs not tested": ~labs}
out = {}
for name, g in STR.items():
    gi = np.where(g)[0]; yy = y[gi]
    if len(gi) < 200 or yy.sum() < 10:
        print(name, "too few"); continue
    pos = {k: v for k, v in zip(gi, range(len(gi)))}
    sub = [np.array([pos[i] for i in b if i in pos]) for b in idx]
    sub = [s for s in sub if len(s) and 0 < yy[s].sum() < len(s)]
    oe = lambda p, s: yy[s].sum() / p[s].sum()
    b_oe = np.array([oe(P3[gi], s) - oe(PR_[gi], s) for s in sub])
    b_au = boot_metric(A, yy, P3[gi], sub) - boot_metric(A, yy, PR_[gi], sub)
    r = {"n": int(len(gi)), "deaths": int(yy.sum()), "share": float(g.mean()),
         "oe_three": float(yy.sum() / P3[gi].sum()), "oe_rate": float(yy.sum() / PR_[gi].sum()),
         "oe_diff": [float(yy.sum() / P3[gi].sum() - yy.sum() / PR_[gi].sum()), *np.percentile(b_oe, [2.5, 97.5]).tolist()],
         "auroc_three": float(A(yy, P3[gi])), "auroc_rate": float(A(yy, PR_[gi])),
         "auroc_diff": [float(A(yy, P3[gi]) - A(yy, PR_[gi])), *np.percentile(b_au, [2.5, 97.5]).tolist()]}
    out[name] = r
    print(f"{name:28s} n {r['n']:6d} ({100 * r['share']:4.1f}%) deaths {r['deaths']:4d} | O/E three {r['oe_three']:.2f} rate {r['oe_rate']:.2f} "
          f"diff {r['oe_diff'][0]:+.3f} [{r['oe_diff'][1]:+.3f},{r['oe_diff'][2]:+.3f}] | AUROC three {r['auroc_three']:.3f} rate {r['auroc_rate']:.3f} "
          f"diff {r['auroc_diff'][0]:+.4f} [{r['auroc_diff'][1]:+.4f},{r['auroc_diff'][2]:+.4f}]", flush=True)
json.dump(out, open(f"{T.ROOT}/results/stream_strata_W.json", "w"), indent=1, default=float)
print("written results/stream_strata_W.json")
