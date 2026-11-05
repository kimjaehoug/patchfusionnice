"""Phase 0 of paper/notes/revision_cadence_prereg.md: does the reset protect against within-stay changes but not against
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

SEEDS = (2, 3, 4, 5)
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
bidx = list(boot_idx(groups[idx], 1000, np.random.RandomState(0)))
out = {}
for name, (tg, kind, parts) in CHANGES.items():
    xs, ms = view(tg, kind, parts)
    share, touched = patch_share(fact["x"], fact["m"], xs, ms)
    ds = {**base, "x": xs, "m": ms}
    R = {"patch_share_changed": share, "samples_changed": touched}
    for regime, (rt, nt) in {"source_only": ("WXRbase", "WXNbase"), "adapted": (f"WXRfull{tg}", f"WXNfull{tg}")}.items():
        P = {"R0": ens(rt, fact), "R1": ens(rt, ds), "N0": ens(nt, fact), "N1": ens(nt, ds)}
        for j, ep in ((0, "y24"), (3, "yhosp")):
            y = base[ep]; ii = [i for i in bidx if 0 < y[i].sum() < len(i)]
            for mn, f in (("auroc", A), ("auprc", AP)):
                b = {k: boot_metric(f, y, p[:, j], ii) for k, p in P.items()}
                dr, dn = f(y, P["R1"][:, j]) - f(y, P["R0"][:, j]), f(y, P["N1"][:, j]) - f(y, P["N0"][:, j])
                prot = (b["R1"] - b["R0"]) - (b["N1"] - b["N0"])
                R[f"{regime}_{ep}_{mn}"] = {"damage_reset": dr, "damage_noreset": dn,
                                           "protection": [dr - dn, *np.percentile(prot, [2.5, 97.5]).tolist()]}
        r = R[f"{regime}_y24_auroc"]
        print(f"[{name:9s} {regime:11s}] patches changed {share:.2f} | 24h AUROC damage reset {r['damage_reset']:+.3f} no-reset {r['damage_noreset']:+.3f} "
              f"| protection {r['protection'][0]:+.4f} [{r['protection'][1]:+.4f},{r['protection'][2]:+.4f}]", flush=True)
    out[name] = R
json.dump(out, open(f"{T.ROOT}/results/reset_hyp_W.json", "w"), indent=1, default=float)
print("written results/reset_hyp_W.json")
