"""Trace of the persistent part (b) of the reset's source-only loss at the Hallym hospitals (24-h AUROC), after input
channels were excluded (reset_gap_forensics.py). Frozen models: reset W_base vs no reset W_noreset (seeds 2-4).
1. Value statistics, Hallym vs MIMIC-IV (observed hours).
2. Value manipulations of the Hallym input (quantile mapping to MIMIC-IV, smoothing): does the gap close?
3. Representation shift: mean-pooled encoder output on MIMIC-IV, low-testing eICU-CRD and Hallym; linear separability
   from MIMIC-IV (5-fold logistic regression AUROC) and standardised mean shift, reset vs no reset.
Diagnostic use of Hallym outcomes only. Writes results/reset_b_W.json."""
import json, os
import numpy as np
import torch
import train as T
from eval_hallym import frozen_preds
from link_eval import low_tertile
from sklearn.metrics import roc_auc_score as A
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

VARS = ["rr", "temp", "hr", "spo2", "sbp", "dbp", "bun", "lactate", "wbc", "hb", "na", "k", "cl", "ph", "be", "ag",
        "ne", "vaso", "fio2", "vent", "uo", "gcs"]
VIT, LAB = list(range(6)), list(range(6, 16))
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn"); e = T.load("eicu_dyn")
y = h["y24"].astype(int); S = (2, 3, 4); dev = "cuda"
ens = lambda tag, ds: np.mean([frozen_preds(f"{tag}_s{s}", ds, st, dev) for s in S], 0)[:, 0]
out = {}

# ---- 1. value statistics on observed hours ----
def vstats(ds, c):
    m, x = ds["m"][:, :, c] > 0, ds["x"][:, :, c]
    v = x[m]
    xo = np.where(m, x, np.nan)
    dd = np.abs(np.diff(xo, axis=1)); same = (dd == 0)
    ok = ~np.isnan(dd)
    return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "abs_hourly_change_median": float(np.nanmedian(dd[ok])) if ok.any() else None,
            "share_identical_consecutive": float(same[ok].mean()) if ok.any() else None, "share_integer": float(np.mean(np.isclose(v, np.round(v))))}
out["values"] = {}
for c in VIT + LAB:
    if (h["m"][:, :, c] > 0).sum() < 100:
        continue
    out["values"][VARS[c]] = {"mimic": vstats(d, c), "hallym": vstats(h, c)}
    a, b = out["values"][VARS[c]]["mimic"], out["values"][VARS[c]]["hallym"]
    fm = lambda v: "-" if v is None else f"{v:.2f}"
    print(f"{VARS[c]:5s} mean {fm(a['mean'])}/{fm(b['mean'])}  sd {fm(a['sd'])}/{fm(b['sd'])}  |dx| {fm(a['abs_hourly_change_median'])}/{fm(b['abs_hourly_change_median'])}  "
          f"identical {fm(a['share_identical_consecutive'])}/{fm(b['share_identical_consecutive'])}  integer {fm(a['share_integer'])}/{fm(b['share_integer'])}  (MIMIC/Hallym)", flush=True)

# ---- 2. value manipulations ----
rs = np.random.RandomState(0)
ref = {c: np.sort(rs.choice(d["x"][:, :, c][d["m"][:, :, c] > 0], 200000)) for c in VIT + LAB if (d["m"][:, :, c] > 0).any()}
def qmap(ds, cols):
    x = ds["x"].copy()
    for c in cols:
        if c not in ref or (ds["m"][:, :, c] > 0).sum() < 100:
            continue
        own = np.sort(ds["x"][:, :, c][ds["m"][:, :, c] > 0])
        v = x[:, :, c]; ok = ~np.isnan(v)
        q = np.searchsorted(own, v[ok], side="right") / len(own)
        v[ok] = np.quantile(ref[c], np.clip(q, 0, 1)); x[:, :, c] = v
    return {**ds, "x": x}
def smooth(ds, cols, w=3):
    x = ds["x"].copy()
    for c in cols:
        v = x[:, :, c]; ok = ~np.isnan(v)
        num = np.zeros_like(v); den = np.zeros_like(v)
        for k in range(-(w // 2), w // 2 + 1):
            sv = np.roll(np.where(ok, v, 0), k, 1); sm = np.roll(ok, k, 1).astype(float)
            if k > 0: sv[:, :k] = 0; sm[:, :k] = 0
            if k < 0: sv[:, k:] = 0; sm[:, k:] = 0
            num += sv; den += sm
        x[:, :, c] = np.where(ok, num / np.maximum(den, 1), np.nan)
    return {**ds, "x": x}
VARIANTS = {"factual": h, "qmap_vitals": qmap(h, VIT), "qmap_labs": qmap(h, LAB), "qmap_all": qmap(h, VIT + LAB),
            "smooth_vitals": smooth(h, VIT)}
for c in VIT:
    VARIANTS[f"qmap_{VARS[c]}"] = qmap(h, [c])
out["manip"] = {}
for k, ds in VARIANTS.items():
    r_, n_ = ens("W_base", ds), ens("W_noreset", ds)
    out["manip"][k] = {"reset": A(y, r_), "noreset": A(y, n_), "gap": A(y, r_) - A(y, n_)}
    print(f"{k:14s} reset {A(y, r_):.4f} noreset {A(y, n_):.4f} gap {A(y, r_) - A(y, n_):+.4f}", flush=True)

# ---- 3. representation shift ----
def reps(tag, ds, n=None):
    r = {}
    for fn in ("runs.jsonl", "runs_quick.jsonl"):
        for l in open(f"{T.ROOT}/results/{fn}"):
            if l.strip():
                q = json.loads(l); r[q["tag"]] = q
    cfg = json.loads(json.dumps(r[tag]["cfg"])); cfg.setdefault("enc", {}).setdefault("T", 48)
    T.AVAIL = True; cfg["enc"]["in_dim"] = 88
    model = T.build(r[tag]["model"], cfg).to(dev); model.load_state_dict(torch.load(f"{T.ROOT}/results/preds/{tag}.pt", map_location=dev), strict=False); model.eval()
    idx = np.arange(len(ds["stay_id"])) if n is None else n
    X = T.tensors(ds, idx, st, torch.device("cpu"))[0]
    zs = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            zs.append(model.enc(X[i:i + 2048].to(dev)).mean(1).float().cpu().numpy())
    return np.concatenate(zs)
rsel = np.random.RandomState(1)
mi = np.sort(rsel.choice(len(d["stay_id"]), 6000, replace=False))
low = np.where(low_tertile(e))[0]; ei = np.sort(rsel.choice(low, 6000, replace=False))
hi = np.sort(rsel.choice(len(h["stay_id"]), min(6000, len(h["stay_id"])), replace=False))
def shift(zs, zt):
    lab = np.r_[np.zeros(len(zs)), np.ones(len(zt))]; Z = np.r_[zs, zt]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1), Z, lab, cv=5, method="predict_proba")[:, 1]
    ms = np.linalg.norm(zt.mean(0) - zs.mean(0)) / np.sqrt(zs.var(0).mean())
    return A(lab, p), float(ms)
out["repr"] = {}
for kind, tag in (("reset", "W_base"), ("noreset", "W_noreset")):
    for s in S:
        zm, ze, zh = reps(f"{tag}_s{s}", d, mi), reps(f"{tag}_s{s}", e, ei), reps(f"{tag}_s{s}", h, hi)
        (ae, me), (ah, mh) = shift(zm, ze), shift(zm, zh)
        out["repr"][f"{kind}_s{s}"] = {"eicu_probe_auroc": ae, "eicu_mean_shift": me, "hallym_probe_auroc": ah, "hallym_mean_shift": mh}
        print(f"[repr {kind} s{s}] MIMIC vs eICU-low probe AUROC {ae:.3f} shift {me:.3f} | MIMIC vs Hallym probe AUROC {ah:.3f} shift {mh:.3f}", flush=True)
json.dump(out, open(f"{T.ROOT}/results/reset_b_W.json", "w"), indent=1, default=float)
print("written results/reset_b_W.json")
