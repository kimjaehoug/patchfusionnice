"""Transplant test for the reset's Hallym loss. Clue (reset_signal_trace.py): at Hallym only, delta = logit(reset) -
logit(no reset) falls with respiratory and heart rate (Spearman -0.57 / -0.43; MIMIC-IV -0.04 / +0.06). Candidate
contexts that differ at Hallym: sparse SpO2 (hourly rate 0.095 vs 0.59) and the never-recorded variables. Low-testing
eICU-CRD samples (labels available; design data) receive one Hallym observation feature at a time; a cause must
reproduce both (1) a reset loss in 24-h AUROC and (2) the negative delta correlation with respiratory / heart rate.
Frozen W_base vs W_noreset, seeds 2-4. Writes results/reset_transplant_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
import obsprofile as PR
from eval_hallym import frozen_preds
from link_eval import low_tertile
from sklearn.metrics import roc_auc_score as A

d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn"); e = T.load("eicu_dyn"); S = (2, 3, 4)
low = np.where(low_tertile(e))[0]; idx = np.sort(np.random.RandomState(0).choice(low, 30000, replace=False))
E = {k: (v[idx] if hasattr(v, "__len__") and len(v) == len(e["stay_id"]) else v) for k, v in e.items()}
rate_h, rate_e = PR.obs_rate(h["m"]), PR.obs_rate(E["m"])
absent_h = [c for c in range(22) if rate_h[c] == 0 and rate_e[c] > 0]
SPO2, RR, HR = 3, 0, 2
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
ens = lambda tag, ds: np.mean([frozen_preds(f"{tag}_s{s}", ds, st, "cuda") for s in S], 0)[:, 0]


def variant(spo2=False, absent=False, seed=0):
    r_t = rate_e.copy()
    if spo2:
        r_t[SPO2] = min(rate_h[SPO2], rate_e[SPO2])
    if absent:
        r_t[absent_h] = 0.0
    xs, ms = PR.imagine_rate(E["x"], E["m"], rate_e, r_t, np.random.RandomState(seed))
    return {**E, "x": xs, "m": ms}


def mean48(ds, c):
    with np.errstate(all="ignore"):
        return np.nanmean(np.where(ds["m"][:, :, c] > 0, ds["x"][:, :, c], np.nan), 1)


out = {"absent_h": absent_h, "spo2_rate": [float(rate_e[SPO2]), float(rate_h[SPO2])]}
y = E["y24"].astype(int)
for name, kw in (("factual", None), ("spo2_sparse", dict(spo2=True)), ("absent_vars", dict(absent=True)), ("both", dict(spo2=True, absent=True))):
    ds = E if kw is None else variant(**kw)
    pr, pn = ens("W_base", ds), ens("W_noreset", ds)
    dl = logit(pr) - logit(pn)
    rr, hr = mean48(ds, RR), mean48(ds, HR)
    ok_r, ok_h = ~np.isnan(rr), ~np.isnan(hr)
    res = {"reset": A(y, pr), "noreset": A(y, pn), "gap": A(y, pr) - A(y, pn),
           "rho_delta_rr": float(pd.Series(rr[ok_r]).corr(pd.Series(dl[ok_r]), method="spearman")),
           "rho_delta_hr": float(pd.Series(hr[ok_h]).corr(pd.Series(dl[ok_h]), method="spearman")),
           "delta_auroc_death": float(A(y, dl))}
    out[name] = res
    print(f"{name:12s} reset {res['reset']:.4f} noreset {res['noreset']:.4f} gap {res['gap']:+.4f} | rho(delta, RR) {res['rho_delta_rr']:+.3f} "
          f"rho(delta, HR) {res['rho_delta_hr']:+.3f} | AUROC(delta) {res['delta_auroc_death']:.3f}", flush=True)
print("absent at Hallym:", absent_h, "| SpO2 hourly rate eICU-low / Hallym", out["spo2_rate"])
json.dump(out, open(f"{T.ROOT}/results/reset_transplant_W.json", "w"), indent=1, default=float)
print("written results/reset_transplant_W.json")
