"""Quick check of the absence-conditioned reset start (QA, dyn_h0="avail") against the reset (WXRbase) and no-reset
(WXNbase) source-only models, seeds 2-3, same held-out development patients and changes as reset_hyp_eval.py, plus factual
held-out and low-testing eICU-CRD performance. Hallym outcomes are not used. Writes results/qa_check_W.json.
Based on: Phase 0 of paper/notes/revision_cadence_prereg.md: does the reset protect against within-stay changes but not against
site-scale absence? Held-out development patients (outside the 30% training subset, as damage_eval.py) are re-observed with
one kind of change only: (a) the Hallym absent-variable set, (b) Hallym within-stay cadence (laboratory + vital signs),
(c) low-testing eICU-CRD within-stay cadence (laboratory + charting). Frozen 30%-trained models with reset (WXR*) and
without (WXN*), four-seed ensembles. Damage = metric(changed) - metric(factual); reset protection = damage(reset) -
damage(no reset) (> 0: the reset loses less), 95% patient bootstrap CI (1,000). Also the share of 6-h patches whose input
changes. Writes results/reset_hyp_W.json."""
import copy
import json
import numpy as np
import train as T
import obsprofile as PR
from eval_hallym import frozen_preds, boot_idx
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

SEEDS = (2, 3)
from link_eval import low_tertile
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); groups = d["subject_id"]
us = np.unique(groups)
train_mask = np.isin(groups, np.random.RandomState(7).choice(us, int(0.3 * len(us)), replace=False))
tr, ho = np.where(train_mask)[0], np.where(~train_mask)[0]
idx = np.sort(np.random.RandomState(0).choice(ho, 30000, replace=False))
acu = PR.BlockAcuity().fit(d["x"][tr], d["yhosp"][tr]); devp = PR.estimate(d["x"][tr], d["m"][tr], acu)
TG = {t: [PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp) for sx, sm in PR.target_sites(src)]
      for t, src in (("H", "hallym_dyn:sites"), ("E", "eicu_dyn:low:sites"))}
CHANGES = {"absent_H": ("H", "absent", ()), "cadence_H": ("H", "cad", ("cad", "vitcad")), "cadence_E": ("E", "cad", ("cad", "doccad"))}


def view(tg, kind, parts):
    x, m = d["x"][idx], d["m"][idx]
    xs, ms = np.empty_like(x), np.empty_like(m)
    r = np.random.RandomState(1); site = r.randint(0, len(TG[tg]), len(idx))
    for k in range(len(TG[tg])):
        part = np.where(site == k)[0]
        p = copy.deepcopy(TG[tg][k]); p.rho = devp.rho.copy()
        if kind != "absent":
            p.absent = []
        xs[part], ms[part] = PR.imagine_two(x[part], m[part], acu, devp, p, parts=parts, rng=r)
    return xs, ms


def patch_share(x0, m0, x1, m1):
    """share of (sample, 6-h patch) whose hourly input (values incl. carried values, masks, availability) changed,
    among samples with any change"""
    diff = (m0 != m1) | (np.isnan(x0) != np.isnan(x1)) | (np.nan_to_num(x0, nan=0) != np.nan_to_num(x1, nan=0))
    per = diff.any(2).reshape(len(x0), 8, 6).any(2)                # [N, 8 patches]
    anyc = per.any(1)
    return float(per[anyc].mean()), float(anyc.mean())


base = {k: d[k][idx] for k in ("stay_id", "t_event", "event", "y24", "y48", "y168", "yhosp")}
fact = {**base, "x": d["x"][idx], "m": d["m"][idx]}
ens = lambda tag, ds: np.mean([frozen_preds(f"{tag}_s{s}", ds, st, "cuda") for s in SEEDS], 0)
MOD = {"reset": "WXRbase", "reset_avail": "QA", "noreset": "WXNbase"}
F = {g: ens(t, fact) for g, t in MOD.items()}
y = base["y24"]
out = {"factual_heldout_y24": {g: [A(y, p[:, 0]), AP(y, p[:, 0])] for g, p in F.items()}}
print("factual held-out 24h", {g: [round(v, 4) for v in r] for g, r in out["factual_heldout_y24"].items()}, flush=True)
for name, (tg, kind, parts) in CHANGES.items():
    xs, ms = view(tg, kind, parts)
    ds = {**base, "x": xs, "m": ms}
    out[name] = {}
    for g, t in MOD.items():
        p1 = ens(t, ds)
        out[name][g] = {"y24_auroc": A(y, p1[:, 0]) - A(y, F[g][:, 0]), "y24_auprc": AP(y, p1[:, 0]) - AP(y, F[g][:, 0]),
                        "yhosp_auroc": A(base["yhosp"], p1[:, 3]) - A(base["yhosp"], F[g][:, 3])}
    print(name, {g: {k: round(v, 4) for k, v in r.items()} for g, r in out[name].items()}, flush=True)
e = T.load("eicu_dyn"); low = low_tertile(e)
for g, t in MOD.items():
    p = np.mean([np.load(f"{T.ROOT}/results/preds/{t}_s{s}.npz")["ext"] for s in SEEDS], 0)[low]
    out.setdefault("eicu_low_source_only", {})[g] = {ep: [A(e[ep][low], p[:, j]), AP(e[ep][low], p[:, j])] for j, ep in enumerate(["y24", "y48", "y168", "yhosp"])}
print("eICU-low", {g: {k: [round(x, 3) for x in v] for k, v in r.items()} for g, r in out["eicu_low_source_only"].items()}, flush=True)
json.dump(out, open(f"{T.ROOT}/results/qa_check_W.json", "w"), indent=1, default=float)
print("written results/qa_check_W.json")
