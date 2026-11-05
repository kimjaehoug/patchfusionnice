"""Reset vs no reset at full scale (three seeds each): source-only at hidden size 189 and 256, and target-label-free
adaptation at hidden size 256 (Hallym- and eICU-targeted imagination). Difference = reset minus no reset, paired
cluster bootstrap (patients in MIMIC-IV and Hallym, hospitals in eICU-CRD). Writes results/reset_full_W.json."""
from bootstrap_par import boot_diff
import json
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

S = (2, 3, 4)
EPS = ["y24", "y48", "y168", "yhosp"]
PAIRS = {"src_189": ("F_base", "E_noreset"), "src_256": ("W_base", "W_noreset"),
         "imagH_256": ("W_propH", "W_noresetPropH"), "imagE_256": ("W_propE", "W_noresetPropE")}
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn"); e = T.load("eicu_dyn"); low = low_tertile(e)
tags = sorted({t for p in PAIRS.values() for t in p})
ld = lambda p, k: np.nanmean([np.load(f"{T.ROOT}/results/preds/{p}_s{s}.npz")[k] for s in S], 0)
PO = {t: ld(t, "oof") for t in tags}
PE = {t: ld(t, "ext") for t in tags}
PH = {t: np.mean([frozen_preds(f"{t}_s{s}", h, st, "cuda") for s in S], 0) for t in tags}
ok = ~np.isnan(PO["W_base"][:, 0])
SRC = {"mimic": (d, ok, d["subject_id"][ok], PO, True), "eicu": (e, np.ones(len(low), bool), e["site"], PE, False),
       "eicu_low": (e, low, e["site"][low], PE, False), "hallym": (h, np.ones(len(h["y24"]), bool), h["subject_id"], PH, False)}
out = {}
for sn, (ds, m, cl, P, oof) in SRC.items():
    idx = list(boot_idx(cl, 500, np.random.RandomState(0)))
    for j, ep in enumerate(EPS):
        y = ds[ep][m]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        pr = lambda t: P[t][m][:, j] if oof else P[t][m][:, j]
        for pn, (r, n) in PAIRS.items():
            rec = {"reset": [A(y, pr(r)), AP(y, pr(r))], "noreset": [A(y, pr(n)), AP(y, pr(n))]}
            for mn, f in (("auroc", A), ("auprc", AP)):
                bs = boot_diff(f, y, pr(r), pr(n), ii)
                rec[f"d_{mn}"] = [f(y, pr(r)) - f(y, pr(n)), *np.percentile(bs, [2.5, 97.5]).tolist()]
            out[f"{pn}|{sn}|{ep}"] = rec
            print(f"{pn:10s} {sn:8s} {ep:5s} reset {rec['reset'][0]:.3f}/{rec['reset'][1]:.3f} no-reset {rec['noreset'][0]:.3f}/{rec['noreset'][1]:.3f} "
                  f"dAUROC {rec['d_auroc'][0]:+.3f} [{rec['d_auroc'][1]:+.3f},{rec['d_auroc'][2]:+.3f}] dAUPRC {rec['d_auprc'][0]:+.3f} [{rec['d_auprc'][1]:+.3f},{rec['d_auprc'][2]:+.3f}]", flush=True)
json.dump(out, open(f"{T.ROOT}/results/reset_full_W.json", "w"), indent=1, default=float)
print("written results/reset_full_W.json")
