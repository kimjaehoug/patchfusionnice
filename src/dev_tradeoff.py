"""Development (held-out MIMIC-IV fold) performance of adapted models versus the source-only backbone, five seeds,
patient-cluster bootstrap. Usage: python dev_tradeoff.py F dev_tradeoff.json | python dev_tradeoff.py W dev_tradeoff_W.json"""
from bootstrap_par import boot_diff
import sys
import json
import numpy as np
import train as T
from eval_hallym import boot_idx
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

PFX = sys.argv[1] if len(sys.argv) > 1 else "F"
OUT = sys.argv[2] if len(sys.argv) > 2 else "dev_tradeoff.json"
S = (2, 3, 4, 5, 6)
d = T.load("mimic_dyn")
P = {g: np.nanmean([np.load(f"{T.ROOT}/results/preds/{PFX}_{g}_s{s}.npz")["oof"] for s in S], 0)
     for g in ["base", "gen", "propE", "propH", "selE", "selH"]}
ok = ~np.isnan(P["base"][:, 0])
cl = d["subject_id"][ok]
idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
out = {}
for j, ep in enumerate(["y24", "y48", "y168", "yhosp"]):
    y = d[ep][ok]
    ii = [i for i in idx if 0 < y[i].sum() < len(i)]
    for g in ["gen", "propE", "propH"]:
        for nm, f in (("AUROC", A), ("AUPRC", AP)):
            a, b = P[g][ok][:, j], P["base"][ok][:, j]
            bs = boot_diff(f, y, a, b, ii)
            out[f"{g}_{ep}_{nm}"] = [f(y, a) - f(y, b), *np.percentile(bs, [2.5, 97.5]).tolist()]
        print(ep, g, "AUROC %+.3f [%+.3f,%+.3f]" % tuple(out[f"{g}_{ep}_AUROC"]),
              "AUPRC %+.3f [%+.3f,%+.3f]" % tuple(out[f"{g}_{ep}_AUPRC"]), flush=True)
json.dump(out, open(f"{T.ROOT}/results/{OUT}", "w"), default=float, indent=1)
print("written", OUT)
