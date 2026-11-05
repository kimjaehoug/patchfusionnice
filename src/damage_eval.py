"""Damage decomposition (Table 4) and charting-cadence robustness (Section 6.7) for 30%-trained models.
Held-out development patients = patients outside the fixed 30% training subset (train.py: RandomState(7)). One profile
component at a time is set to the target value; the change of 24-h AUROC/AUPRC and in-hospital AUROC is reported per seed.
Usage: python damage_eval.py PFX OUT   e.g. python damage_eval.py WX damage_W.json   (models {PFX}Rbase, {PFX}gen, {PFX}RselE, {PFX}RfullE)"""
import sys
import copy
import json
import numpy as np
import train as T
import obsprofile as PR
from eval_hallym import frozen_preds
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

PFX = sys.argv[1] if len(sys.argv) > 1 else "WX"
OUT = sys.argv[2] if len(sys.argv) > 2 else "damage_W.json"
SEEDS = (2, 3, 4, 5)
d = T.load("mimic_dyn")
st = T.fit_stats(d["x"])
groups = d["subject_id"]
us = np.unique(groups)
keep = np.random.RandomState(7).choice(us, int(0.3 * len(us)), replace=False)
train_mask = np.isin(groups, keep)
tr = np.where(train_mask)[0]
ho = np.where(~train_mask)[0]
idx = np.sort(np.random.RandomState(0).choice(ho, 30000, replace=False))
acu = PR.BlockAcuity().fit(d["x"][tr], d["yhosp"][tr])
devp = PR.estimate(d["x"][tr], d["m"][tr], acu)
TG = {"H": [PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp) for sx, sm in PR.target_sites("hallym_dyn:sites")],
      "E": [PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp) for sx, sm in PR.target_sites("eicu_dyn:low:sites")]}
COMP = [("absent", "H", "Absent variables (site)"), ("streams", "E", "Charting streams removed per stay (stay)"),
        ("sel", "H", "Laboratory selection (stay)"), ("sel", "E", "Laboratory selection (stay)"),
        ("doccad", "E", "Charting cadence, GCS and urine output (patch)"), ("cad", "H", "Laboratory cadence (patch)"),
        ("cad", "E", "Laboratory cadence (patch)"), ("vitcad", "H", "Vital-sign cadence (patch)")]


def view(kind, tg):
    x, m = d["x"][idx], d["m"][idx]
    xs, ms = np.empty_like(x), np.empty_like(m)
    r = np.random.RandomState(1)
    site = r.randint(0, len(TG[tg]), len(idx))
    for k in range(len(TG[tg])):
        part = np.where(site == k)[0]
        if not len(part):
            continue
        p = copy.deepcopy(TG[tg][k])
        if kind != "absent":
            p.absent = []
        if kind != "streams":
            p.rho = devp.rho.copy()
        parts = {"absent": (), "streams": (), "sel": ("sel",), "doccad": ("doccad",), "cad": ("cad",), "vitcad": ("vitcad",)}[kind]
        xs[part], ms[part] = PR.imagine_two(x[part], m[part], acu, devp, p, parts=parts, rng=r)
    return xs, ms


base = {k: d[k][idx] for k in ("stay_id", "t_event", "event", "y24", "y48", "y168", "yhosp")}
DS = {"factual": {**base, "x": d["x"][idx], "m": d["m"][idx]}}
for kind, tg, _ in COMP:
    if f"{kind}_{tg}" not in DS:
        xs, ms = view(kind, tg)
        DS[f"{kind}_{tg}"] = {**base, "x": xs, "m": ms}
y24, yh = base["y24"], base["yhosp"]
met = lambda p: (A(y24, p[:, 0]), AP(y24, p[:, 0]), A(yh, p[:, 3]))
out = {"damage": {}, "robust": {}}
# damage decomposition: backbone, per seed
for kind, tg, lab in COMP:
    rows = []
    for s in SEEDS:
        t = f"{PFX}Rbase_s{s}"
        f0, f1 = met(frozen_preds(t, DS["factual"], st, "cuda")), met(frozen_preds(t, DS[f"{kind}_{tg}"], st, "cuda"))
        rows.append([b - a for a, b in zip(f0, f1)])
    rows = np.array(rows)
    out["damage"][f"{kind}_{tg}"] = {"label": lab, "target": tg, "per_seed": rows.tolist(), "mean": rows.mean(0).tolist(), "min": rows.min(0).tolist(), "max": rows.max(0).tolist()}
    print(f"{lab:48s} {tg} dAUROC24 {rows[:, 0].mean():+.3f} ({rows[:, 0].min():+.3f},{rows[:, 0].max():+.3f})  dAUPRC24 {rows[:, 1].mean():+.3f}  dAUROChosp {rows[:, 2].mean():+.3f}", flush=True)
# robustness to charting cadence (eICU-CRD low rate): seed ensembles
for g, lab in ((f"{PFX}Rbase", "Backbone"), (f"{PFX}gen", "Generic counterfactual training"),
               (f"{PFX}RselE", "Profile matching, stay scale only"), (f"{PFX}RfullE", "Profile matching with charting cadence")):
    import os
    if not all(os.path.exists(f"{T.ROOT}/results/preds/{g}_s{s}.pt") for s in SEEDS):
        print("skip (missing)", g); continue
    pf = np.mean([frozen_preds(f"{g}_s{s}", DS["factual"], st, "cuda") for s in SEEDS], 0)
    pt = np.mean([frozen_preds(f"{g}_s{s}", DS["doccad_E"], st, "cuda") for s in SEEDS], 0)
    f0, f1 = met(pf), met(pt)
    out["robust"][g] = {"label": lab, "factual": f0, "thinned": f1}
    print(f"{lab:40s} factual {f0[0]:.3f}/{f0[1]:.3f} thinned {f1[0]:.3f}/{f1[1]:.3f} d24 {f1[0] - f0[0]:+.3f} dhosp {f1[2] - f0[2]:+.3f}", flush=True)
json.dump(out, open(f"{T.ROOT}/results/{OUT}", "w"), indent=1, default=float)
print("written", OUT)
