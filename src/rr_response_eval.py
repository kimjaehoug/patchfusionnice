"""Respiratory-rate response of reset vs no-reset encoders (Supplementary S18). Every recorded respiratory-rate value
and its carried-forward copies are raised by 6 breaths/min (equivalent to shifting measured values and re-imputing);
standardised values and first differences are recomputed by the input function (a constant shift leaves first
differences, masks and availability unchanged). Response = mean change of the 24-h logit; reported as the ratio of
mean logit changes (reset / no reset), not as a ratio of risks. Per seed and for three-seed ensembles, 95% patient-
cluster bootstrap (1,000). Also the response of the adapted reset encoders (three-scale vs rate-level, seeds 2-6).
Writes results/rr_response_W.json."""
import json
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from sklearn.metrics import roc_auc_score as A

d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn")
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
x = h["x"].copy(); x[:, :, 0] = x[:, :, 0] + 6; h2 = {**h, "x": x}
dl = lambda tag: logit(frozen_preds(tag, h2, st, "cuda")[:, 0]) - logit(frozen_preds(tag, h, st, "cuda")[:, 0])
y = h["y24"].astype(int)
idx = list(boot_idx(h["subject_id"], 1000, np.random.RandomState(0)))
def ratio_ci(a, b):
    bs = np.array([a[i].mean() / b[i].mean() for i in idx])
    return [float(a.mean() / b.mean()), *np.percentile(bs, [2.5, 97.5]).tolist()]
out = {"per_seed": {}, "ensemble": {}}
D = {}
for s in (2, 3, 4):
    for k in ("W_base", "W_noreset", "W_propH", "W_noresetPropH"):
        D[(k, s)] = dl(f"{k}_s{s}")
    gap = A(y, frozen_preds(f"W_base_s{s}", h, st, "cuda")[:, 0]) - A(y, frozen_preds(f"W_noreset_s{s}", h, st, "cuda")[:, 0])
    r_src, r_ad = ratio_ci(D[("W_base", s)], D[("W_noreset", s)]), ratio_ci(D[("W_propH", s)], D[("W_noresetPropH", s)])
    out["per_seed"][s] = {"source_only_ratio": r_src, "adapted_ratio": r_ad, "source_only_auroc_gap": gap}
    print(f"seed {s}: source-only ratio {r_src[0]:.2f} [{r_src[1]:.2f},{r_src[2]:.2f}]  AUROC gap {gap:+.4f} | adapted ratio {r_ad[0]:.2f} [{r_ad[1]:.2f},{r_ad[2]:.2f}]", flush=True)
for lab, r, n in (("source_only", "W_base", "W_noreset"), ("adapted", "W_propH", "W_noresetPropH")):
    a = np.mean([D[(r, s)] for s in (2, 3, 4)], 0); b = np.mean([D[(n, s)] for s in (2, 3, 4)], 0)
    out["ensemble"][lab] = {"reset_response": float(a.mean()), "noreset_response": float(b.mean()), "ratio": ratio_ci(a, b)}
    print(f"ensemble {lab}: reset {a.mean():+.3f} noreset {b.mean():+.3f} ratio {ratio_ci(a, b)}", flush=True)
# adapted reset encoders: three-scale vs rate-level, seeds 2-6
out["adapted_reset"] = {}
for s in (2, 3, 4, 5, 6):
    a3, ar = dl(f"W_propH_s{s}"), dl(f"W_thinH_s{s}")
    au3 = A(y, frozen_preds(f"W_propH_s{s}", h, st, "cuda")[:, 0]); aur = A(y, frozen_preds(f"W_thinH_s{s}", h, st, "cuda")[:, 0])
    out["adapted_reset"][s] = {"three_scale_response": float(a3.mean()), "rate_level_response": float(ar.mean()), "auroc_three": au3, "auroc_rate": aur}
    print(f"adapted reset seed {s}: RR response three-scale {a3.mean():+.3f} rate-level {ar.mean():+.3f} | 24h AUROC three {au3:.3f} rate {aur:.3f}", flush=True)
json.dump(out, open(f"{T.ROOT}/results/rr_response_W.json", "w"), indent=1, default=float)
print("written results/rr_response_W.json")
