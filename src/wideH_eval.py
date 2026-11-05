"""Formal 3-seed comparison: reset vs no-reset at 2.53M (F_base / E_noreset) and at the wider size (W_base / W_noreset).
24-h and 7-day AUROC/AUPRC in MIMIC-IV (held-out fold), eICU-CRD (hospital-cluster bootstrap) and Hallym (patient-cluster)."""
import numpy as np, train as T
from eval_hallym import frozen_preds, boot_idx
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
S = (2, 3, 4)
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn"); e = T.load("eicu_dyn")
G = {"F_propH": "reset 2.53M +imag", "W_propH": "reset wide +imag", "W_noresetPropH": "no reset wide +imag", "W_base": "reset wide src-only", "W_noreset": "no reset wide src-only"}
ld = lambda p, k: np.nanmean([np.load(f"{T.ROOT}/results/preds/{p}_s{s}.npz")[k] for s in S], 0)
PE = {g: ld(g, "ext") for g in G}; PO = {g: ld(g, "oof") for g in G}
PH = {g: np.mean([frozen_preds(f"{g}_s{s}", h, st, "cuda") for s in S], 0) for g in G}
ok = ~np.isnan(PO["W_propH"][:, 0])
src = {"MIMIC": (d, ok, d["subject_id"][ok], PO), "eICU": (e, np.ones(len(e["y24"]), bool), e["site"], PE), "Hallym": (h, np.ones(len(h["y24"]), bool), h["subject_id"], PH)}
for nm, (ds, m, cl, P) in src.items():
    idx = list(boot_idx(cl, 500, np.random.RandomState(0)))
    for j, ep in [(0, "y24"), (2, "y168")]:
        y = ds[ep][m]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        pr = lambda g: P[g][m][:, j] if nm == "MIMIC" else P[g][:, j]
        print(f"[{nm} {ep}] " + "  ".join(f"{G[g]} {A(y, pr(g)):.3f}/{AP(y, pr(g)):.3f}" for g in G))
        for a, b in (("W_noresetPropH", "W_propH"), ("W_propH", "W_base"), ("W_noresetPropH", "W_noreset"), ("W_propH", "F_propH")):
            for mn, f in (("AUROC", A), ("AUPRC", AP)):
                bs = np.array([f(y[i], pr(a)[i]) - f(y[i], pr(b)[i]) for i in ii])
                print(f"     {G[a]} - {G[b]} {mn} {f(y, pr(a)) - f(y, pr(b)):+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]", end="")
            print(flush=True)
