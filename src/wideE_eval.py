"""W + eICU-target imagination: reset vs no-reset (3 seeds). Primary: low-testing eICU-CRD hospitals (target);
also all eICU-CRD and MIMIC-IV held-out fold. Hospital-/patient-cluster bootstrap."""
import numpy as np, train as T
from eval_hallym import boot_idx
from link_eval import low_tertile
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
S = (2, 3, 4); d = T.load("mimic_dyn"); e = T.load("eicu_dyn"); low = low_tertile(e)
G = {"W_propE": "reset W +imagE", "W_noresetPropE": "no-reset W +imagE", "W_base": "reset W src", "W_noreset": "no-reset W src", "F_propE": "reset 2.53M +imagE"}
ld = lambda p, k: np.nanmean([np.load(f"{T.ROOT}/results/preds/{p}_s{s}.npz")[k] for s in S], 0)
PE = {g: ld(g, "ext") for g in G}; PO = {g: ld(g, "oof") for g in G}; ok = ~np.isnan(PO["W_propE"][:, 0])
src = {"eICU-low": (e, low, e["site"][low], PE), "eICU-all": (e, np.ones(len(low), bool), e["site"], PE), "MIMIC": (d, ok, d["subject_id"][ok], PO)}
for nm, (ds, m, cl, P) in src.items():
    idx = list(boot_idx(cl, 500, np.random.RandomState(0)))
    for j, ep in [(0, "y24"), (1, "y48"), (2, "y168"), (3, "yhosp")]:
        y = ds[ep][m]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]; pr = lambda g: P[g][m][:, j]
        print(f"[{nm} {ep}] " + "  ".join(f"{G[g]} {A(y, pr(g)):.3f}/{AP(y, pr(g)):.3f}" for g in G))
        for a, b in (("W_noresetPropE", "W_propE"), ("W_propE", "W_base"), ("W_noresetPropE", "W_noreset"), ("W_propE", "F_propE")):
            out = []
            for mn, f in (("AUROC", A), ("AUPRC", AP)):
                bs = np.array([f(y[i], pr(a)[i]) - f(y[i], pr(b)[i]) for i in ii])
                out.append(f"{mn} {f(y, pr(a)) - f(y, pr(b)):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
            print(f"     {G[a]} - {G[b]}: " + "  ".join(out), flush=True)
