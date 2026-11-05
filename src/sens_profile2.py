"""Cause test for the Hallym long-horizon loss of larger models: AUROC drop (factual -> view) on 20k MIMIC-IV patients
under Hallym-style re-observation (absent-only / cadence-only / full), 24-h and 7-day risk, seed-averaged per group.
Usage: python sens_profile2.py name=tag1,tag2,tag3 ..."""
import sys, copy, numpy as np, train as T, obsprofile as PR
from eval_hallym import frozen_preds
from sklearn.metrics import roc_auc_score as A
groups = [g.split("=") for g in sys.argv[1:]]
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"])
rs = np.random.RandomState(0); idx = np.sort(rs.choice(len(d["x"]), 20000, replace=False))
acu = PR.BlockAcuity().fit(d["x"][idx], d["yhosp"][idx]); devp = PR.estimate(d["x"][idx], d["m"][idx], acu)
sites = [PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp) for sx, sm in PR.target_sites("hallym_dyn:sites")]
def view(kind):
    x, m = d["x"][idx], d["m"][idx]; xs, ms = np.empty_like(x), np.empty_like(m); r = np.random.RandomState(1)
    for k, part in enumerate(np.array_split(np.arange(len(idx)), len(sites))):
        p = copy.deepcopy(sites[k])
        if kind == "absent":  p.rho = devp.rho.copy(); parts = ()
        if kind == "cadence": p.rho = devp.rho.copy(); p.absent = []; parts = ("cad", "doccad", "vitcad")
        if kind == "full":    parts = ("sel", "cad", "vitcad", "doccad")
        xs[part], ms[part] = PR.imagine_two(x[part], m[part], acu, devp, p, parts=parts, rng=r)
    return xs, ms
base = {k: d[k][idx] for k in ("stay_id", "t_event", "event", "y24", "y48", "y168", "yhosp")}
DS = {"factual": {**base, "x": d["x"][idx], "m": d["m"][idx]}}
for k in ("absent", "cadence", "full"):
    xs, ms = view(k); DS[k] = {**base, "x": xs, "m": ms}
for name, tags in groups:
    R = {}
    for t in tags.split(","):
        P = {k: frozen_preds(t, ds, st, "cuda") for k, ds in DS.items()}
        for j, ep in ((0, "y24"), (2, "y168")):
            y = d[ep][idx]
            for k in DS: R.setdefault((ep, k), []).append(A(y, P[k][:, j]))
    for ep in ("y24", "y168"):
        f = np.mean(R[(ep, "factual")])
        print(f"{name:14s} {ep:5s} factual {f:.3f} | " + "  ".join(f"{k} {np.mean(R[(ep, k)]):.3f} (drop {f - np.mean(R[(ep, k)]):.3f} ±{np.std([a - b for a, b in zip(R[(ep, 'factual')], R[(ep, k)])]):.3f})" for k in ("absent", "cadence", "full")), flush=True)
