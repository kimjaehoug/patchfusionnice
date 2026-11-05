"""Sensitivity of the Hallym results to the definitions of ICU-level care and of in-hospital death time.
Variants (data/hallym_<v>_dyn_tensor.npz): K6 / K10 = ICU-level care from >= 6 / 10 of 12 monitored hours (default 8);
noon / eod = in-hospital death at noon / 23:00 of the death date (default: last vital-sign record on that date).
Frozen hidden-256 models (five seeds), no retraining. Writes results/time_sens_W.json."""
import json
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

S = (2, 3, 4, 5, 6)
EPS = ["y24", "y48", "y168", "yhosp"]
MODELS = {"backbone": "W_base", "generic": "W_gen", "selection_only": "W_selH", "proposed": "W_propH", "proposed_eicu": "W_propE"}
VARIANTS = {"default": "hallym_dyn", "K6": "hallym_K6_dyn", "K10": "hallym_K10_dyn", "noon": "hallym_noon_dyn", "eod": "hallym_eod_dyn"}
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"])
thr = {g: float(np.quantile((lambda o: o[~np.isnan(o)])(np.nanmean([np.load(f"{T.ROOT}/results/preds/{t}_s{s}.npz")["oof"] for s in S], 0)[:, 0]), 0.9))
       for g, t in MODELS.items()}
out = {}
for v, name in VARIANTS.items():
    h = T.load(name)
    P = {g: np.mean([frozen_preds(f"{t}_s{s}", h, st, "cuda") for s in S], 0) for g, t in MODELS.items()}
    idx = list(boot_idx(h["subject_id"], 1000, np.random.RandomState(0)))
    r = {"n": int(len(h["y24"])), "stays": int(len(np.unique(h["stay_id"]))), "events": {ep: int(h[ep].sum()) for ep in EPS}}
    for j, ep in enumerate(EPS):
        y = h[ep]
        ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        r[ep] = {g: [A(y, P[g][:, j]), AP(y, P[g][:, j]), float(y.sum() / P[g][:, j].sum())] for g in MODELS}
        for mn, f in (("auroc", A), ("auprc", AP)):
            bs = boot_metric(f, y, P["proposed"][:, j], ii) - boot_metric(f, y, P["backbone"][:, j], ii)
            r[ep][f"d_{mn}"] = [f(y, P["proposed"][:, j]) - f(y, P["backbone"][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
    y = h["y24"]
    r["deployable10"] = {g: {"alert_rate": float((P[g][:, 0] >= thr[g]).mean()),
                             "sensitivity": float(((P[g][:, 0] >= thr[g]) & (y == 1)).sum() / max(y.sum(), 1)),
                             "ppv": float(((P[g][:, 0] >= thr[g]) & (y == 1)).sum() / max((P[g][:, 0] >= thr[g]).sum(), 1))} for g in MODELS}
    out[v] = r
    c = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]"
    print(f"[{v}] n={r['n']} stays={r['stays']} deaths24={r['events']['y24']} hosp={r['events']['yhosp']} | 24h prop {r['y24']['proposed'][0]:.3f}/{r['y24']['proposed'][1]:.3f} base {r['y24']['backbone'][0]:.3f}/{r['y24']['backbone'][1]:.3f} dAUROC {c(r['y24']['d_auroc'])} dAUPRC {c(r['y24']['d_auprc'])} | "
          f"7d dAUROC {c(r['y168']['d_auroc'])} | hosp dAUROC {c(r['yhosp']['d_auroc'])} | O/E24 prop {r['y24']['proposed'][2]:.2f} base {r['y24']['backbone'][2]:.2f} | "
          f"alerts10 prop {100 * r['deployable10']['proposed']['alert_rate']:.1f}% sens {r['deployable10']['proposed']['sensitivity']:.3f} vs base {100 * r['deployable10']['backbone']['alert_rate']:.1f}% {r['deployable10']['backbone']['sensitivity']:.3f}", flush=True)
json.dump(out, open(f"{T.ROOT}/results/time_sens_W.json", "w"), indent=1, default=float)
print("written results/time_sens_W.json")
