"""Full-scale reset x adaptation interaction (hidden size 256, seeds 2-4 for every model):
gain_R = metric(reset + imagination) - metric(reset, source-only); gain_N likewise without reset;
interaction = gain_R - gain_N. Paired cluster bootstrap (hospitals in eICU-CRD low testing, patients in Hallym).
Writes results/full_interaction_W.json."""
import json
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

S = (2, 3, 4)
EPS = ["y24", "y48", "y168", "yhosp"]
MOD = {"E": {"Rfull": "W_propE", "Rbase": "W_base", "Nfull": "W_noresetPropE", "Nbase": "W_noreset"},
       "H": {"Rfull": "W_propH", "Rbase": "W_base", "Nfull": "W_noresetPropH", "Nbase": "W_noreset"}}
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
out = {}
for tgt, mods in MOD.items():
    if tgt == "E":
        ds, m, cl = e, low, e["site"][low]
        P = {k: np.mean([np.load(f"{T.ROOT}/results/preds/{t}_s{s}.npz")["ext"] for s in S], 0)[low] for k, t in mods.items()}
    else:
        ds, m, cl = h, np.ones(len(h["y24"]), bool), h["subject_id"]
        P = {k: np.mean([frozen_preds(f"{t}_s{s}", h, st, "cuda") for s in S], 0) for k, t in mods.items()}
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    for j, ep in enumerate(EPS):
        y = ds[ep][m]
        ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        for mn, f in (("auroc", A), ("auprc", AP)):
            pt = {k: f(y, P[k][:, j]) for k in mods}
            bs = {k: boot_metric(f, y, P[k][:, j], ii) for k in mods}
            gR, gN = bs["Rfull"] - bs["Rbase"], bs["Nfull"] - bs["Nbase"]
            q = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
            r = {"gain_reset": [pt["Rfull"] - pt["Rbase"], *q(gR)], "gain_noreset": [pt["Nfull"] - pt["Nbase"], *q(gN)],
                 "interaction": [(pt["Rfull"] - pt["Rbase"]) - (pt["Nfull"] - pt["Nbase"]), *q(gR - gN), float((gR - gN > 0).mean())],
                 "adapted_reset_minus_noreset": [pt["Rfull"] - pt["Nfull"], *q(bs["Rfull"] - bs["Nfull"])], "values": pt}
            out[f"{tgt}_{ep}_{mn}"] = r
            c = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]"
            print(f"{tgt} {ep:5s} {mn}: gain reset {c(r['gain_reset'])}  gain no-reset {c(r['gain_noreset'])}  interaction {c(r['interaction'])} P>0={r['interaction'][3]:.2f}", flush=True)
json.dump(out, open(f"{T.ROOT}/results/full_interaction_W.json", "w"), indent=1, default=float)
print("written results/full_interaction_W.json")
