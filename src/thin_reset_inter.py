"""Does the reset encoder strengthen rate-only adaptation too? Interaction (adapted - backbone | reset) -
(adapted - backbone | no reset) for rate-matched thinning and for the proposed profile, seeds 2-4, eICU-CRD low testing
and Hallym; cluster bootstrap (hospitals / patients). Writes results/thin_reset_W.json."""
import json
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

S, EPS = (2, 3, 4), ["y24", "y48", "y168", "yhosp"]
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"])
q = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
c = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]{'*' if v[1] > 0 or v[2] < 0 else ' '}"
out = {}
for tgt in ("E", "H"):
    G = {"Rb": "W_base", "Nb": "W_noreset", "Rt": f"W_thin{tgt}", "Nt": f"W_noresetThin{tgt}", "Rp": f"W_prop{tgt}", "Np": f"W_noresetProp{tgt}"}
    if tgt == "H":
        ds = T.load("hallym_dyn"); M = np.ones(len(ds["y24"]), bool); cl = ds["subject_id"]
        P = {k: np.mean([frozen_preds(f"{t}_s{s}", ds, st, "cuda") for s in S], 0) for k, t in G.items()}
    else:
        ds = T.load("eicu_dyn"); M = low_tertile(ds); cl = ds["site"][M]
        P = {k: np.mean([np.load(f"{T.ROOT}/results/preds/{t}_s{s}.npz")["ext"] for s in S], 0)[M] for k, t in G.items()}
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    print(f"\n===== {tgt}")
    for j, ep in enumerate(EPS):
        y = ds[ep][M]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        print(f"[{ep}] " + "  ".join(f"{k} {A(y, P[k][:, j]):.3f}/{AP(y, P[k][:, j]):.3f}" for k in G))
        for mn, f in (("auroc", A), ("auprc", AP)):
            bs = {k: boot_metric(f, y, P[k][:, j], ii) for k in G}; pt = {k: f(y, P[k][:, j]) for k in G}
            r = {}
            for m_ in ("t", "p"):
                gR, gN = bs[f"R{m_}"] - bs["Rb"], bs[f"N{m_}"] - bs["Nb"]
                r[m_] = {"gain_reset": [pt[f"R{m_}"] - pt["Rb"], *q(gR)], "gain_noreset": [pt[f"N{m_}"] - pt["Nb"], *q(gN)],
                         "interaction": [(pt[f"R{m_}"] - pt["Rb"]) - (pt[f"N{m_}"] - pt["Nb"]), *q(gR - gN)]}
            out[f"{tgt}_{ep}_{mn}"] = {"thinning": r["t"], "proposed": r["p"]}
            print(f"   {mn}: thinning gain R {c(r['t']['gain_reset'])} N {c(r['t']['gain_noreset'])} inter {c(r['t']['interaction'])} | "
                  f"proposed gain R {c(r['p']['gain_reset'])} N {c(r['p']['gain_noreset'])} inter {c(r['p']['interaction'])}", flush=True)
json.dump(out, open(f"{T.ROOT}/results/thin_reset_W.json", "w"), indent=1)
print("written results/thin_reset_W.json")
