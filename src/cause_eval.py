"""Root-cause smoke (20% data, 5 epochs, seeds 2-3): thinning vs proposed variants on Hallym (and eICU-low 24 h)."""
import os
import numpy as np
import train as T
from eval_hallym import frozen_preds
from link_eval import low_tertile
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

V = {"CS0": "thinning", "CS1": "proposed (50% mix)", "CS2": "proposed tfrac=1", "CS3": "tfrac=1, no sel", "CS4": "tfrac=1, no vitcad", "CS5": "thinning, vitals kept", "CS6": "tfrac=1 + target slope", "CS7": "tfrac=1, hourly cadence", "CS8": "tfrac=1, block, no keep-one"}
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn"); e = T.load("eicu_dyn"); low = low_tertile(e)
print(f"{'variant':24s} | Hallym 24h AUROC/AUPRC  48h AUROC/AUPRC  hosp AUROC/AUPRC  O/E24 | eICU-low 24h AUROC/AUPRC | dev OOF 24h")
for k, nm in V.items():
    S = [s for s in (2, 3) if os.path.exists(f"{T.ROOT}/results/preds/{k}_s{s}.npz")]
    if not S:
        continue
    P = np.mean([frozen_preds(f"{k}_s{s}", h, st, "cuda") for s in S], 0)
    Z = [np.load(f"{T.ROOT}/results/preds/{k}_s{s}.npz") for s in S]
    pe = np.mean([z["ext"] for z in Z], 0)[low]
    o = np.nanmean([z["oof"] for z in Z], 0) if "oof" in Z[0] else np.full((len(d["y24"]), 4), np.nan); ok = ~np.isnan(o[:, 0])
    r = [f"{A(h[ep], P[:, j]):.3f}/{AP(h[ep], P[:, j]):.3f}" for j, ep in ((0, "y24"), (1, "y48"), (3, "yhosp"))]
    print(f"{k} {nm:20s} n{len(S)} | {r[0]}  {r[1]}  {r[2]}  {h['y24'].sum() / P[:, 0].sum():.2f} | {A(e['y24'][low], pe[:, 0]):.3f}/{AP(e['y24'][low], pe[:, 0]):.3f} | " +
          (f"{A(d['y24'][ok], o[ok, 0]):.3f}/{AP(d['y24'][ok], o[ok, 0]):.3f}" if ok.sum() > 100 else "-"), flush=True)
