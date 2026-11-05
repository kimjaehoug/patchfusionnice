"""Equal-exposure comparison with five seeds: proposed model trained with target profiles only (tfrac = 1, W_propT*)
vs rate-level imagination (W_thin*) and the main design (W_propE/H), seeds 2-6 for all (tfrac_eval.py: seeds 2-4).
Discrimination at every horizon and 24-h calibration (O/E, intercept, slope, Brier) with cluster bootstrap CIs
(hospitals / patients, 1,000 resamples). Writes results/tfrac5_W.json."""
import json, numpy as np, train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
S = (2, 3, 4, 5, 6)
PAIRS = (("target_only", "thinning"), ("target_only", "main"), ("main", "thinning"))


def calib(y, p):                                       # as clinical_utility.calib
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    b = np.zeros(2); X = np.c_[np.ones_like(lp), lp]
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (y - mu); H = (X * (mu * (1 - mu))[:, None]).T @ X
        b = b + np.linalg.solve(H + 1e-9 * np.eye(2), g)
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp)))
        a += (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-9)
    return {"intercept": a, "slope": b[1], "brier": np.mean((p - y) ** 2), "oe": y.sum() / p.sum()}


d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
out = {"seeds": list(S)}
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
        for a_, b_ in PAIRS:
            for mn, f in (("auroc", A), ("auprc", AP)):
                bs = boot_metric(f, y, P[a_][:, j], ii) - boot_metric(f, y, P[b_][:, j], ii)
                r[f"{a_}-{b_}_{mn}"] = [f(y, P[a_][:, j]) - f(y, P[b_][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
        if ep == "y24":
            y = y.astype(float)
            r["calibration"] = {g: calib(y, P[g][:, j]) for g in mods}
            for a_, b_ in PAIRS:
                bs = np.array([[calib(y[i], P[a_][i, j])[k] - calib(y[i], P[b_][i, j])[k] for k in ("intercept", "slope", "brier", "oe")] for i in ii])
                for q, k in enumerate(("intercept", "slope", "brier", "oe")):
                    r[f"{a_}-{b_}_{k}"] = [r["calibration"][a_][k] - r["calibration"][b_][k], *np.percentile(bs[:, q], [2.5, 97.5]).tolist()]
        out[f"{tgt}_{ep}"] = r
        c = lambda k: f"{r[k][0]:+.3f} [{r[k][1]:+.3f},{r[k][2]:+.3f}]"
        print(f"[{tgt} {ep}] main {r['main'][0]:.3f}/{r['main'][1]:.3f} T {r['target_only'][0]:.3f}/{r['target_only'][1]:.3f} thin {r['thinning'][0]:.3f}/{r['thinning'][1]:.3f} | "
              f"T-thin AUROC {c('target_only-thinning_auroc')} AUPRC {c('target_only-thinning_auprc')} | T-main AUROC {c('target_only-main_auroc')} AUPRC {c('target_only-main_auprc')}", flush=True)
        if ep == "y24":
            print("   calib", {g: {k: round(float(v), 4) for k, v in cc.items()} for g, cc in r["calibration"].items()},
                  "| T-thin", {k: c(f"target_only-thinning_{k}") for k in ("intercept", "slope", "brier", "oe")}, flush=True)
json.dump(out, open(f"{T.ROOT}/results/tfrac5_W.json", "w"), indent=1, default=float)
print("written results/tfrac5_W.json")
