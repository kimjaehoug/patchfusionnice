"""Exploratory: proposed model trained with target profiles only (tfrac = 1, W_propT*) vs the main design (tfrac = 0.5)
and rate-matched thinning, seeds 2-4 for all. Writes results/tfrac_W.json."""
import json, numpy as np, train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
S = (2, 3, 4)
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
out = {}
for tgt, mods in (("E", {"main": "W_propE", "target_only": "W_propTE", "thinning": "W_thinE"}), ("H", {"main": "W_propH", "target_only": "W_propTH", "thinning": "W_thinH"})):
    if tgt == "E":
        ds, m, cl = e, low, e["site"][low]
        P = {g: np.mean([np.load(f"{T.ROOT}/results/preds/{t}_s{s}.npz")["ext"] for s in S], 0)[low] for g, t in mods.items()}
    else:
        ds, m, cl = h, np.ones(len(h["y24"]), bool), h["subject_id"]
        P = {g: np.mean([frozen_preds(f"{t}_s{s}", h, st, "cuda") for s in S], 0) for g, t in mods.items()}
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    for j, ep in enumerate(["y24", "y48", "y168", "yhosp"]):
        y = ds[ep][m]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        r = {g: [A(y, P[g][:, j]), AP(y, P[g][:, j]), float(y.sum() / P[g][:, j].sum())] for g in mods}
        for a_, b_ in (("target_only", "main"), ("target_only", "thinning")):
            for mn, f in (("auroc", A), ("auprc", AP)):
                bs = boot_metric(f, y, P[a_][:, j], ii) - boot_metric(f, y, P[b_][:, j], ii)
                r[f"{a_}-{b_}_{mn}"] = [f(y, P[a_][:, j]) - f(y, P[b_][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
        out[f"{tgt}_{ep}"] = r
        print(f"[{tgt} {ep}] main {r['main'][0]:.3f}/{r['main'][1]:.3f} target-only {r['target_only'][0]:.3f}/{r['target_only'][1]:.3f} thinning {r['thinning'][0]:.3f}/{r['thinning'][1]:.3f} | "
              f"T-main AUROC {r['target_only-main_auroc'][0]:+.3f} AUPRC {r['target_only-main_auprc'][0]:+.3f} [{r['target_only-main_auprc'][1]:+.3f},{r['target_only-main_auprc'][2]:+.3f}] | "
              f"T-thin AUPRC {r['target_only-thinning_auprc'][0]:+.3f} [{r['target_only-thinning_auprc'][1]:+.3f},{r['target_only-thinning_auprc'][2]:+.3f}] | O/E main {r['main'][2]:.2f} T {r['target_only'][2]:.2f} thin {r['thinning'][2]:.2f}", flush=True)
json.dump(out, open(f"{T.ROOT}/results/tfrac_W.json", "w"), indent=1, default=float)
print("written results/tfrac_W.json")
