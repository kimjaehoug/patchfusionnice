"""Sensitivity of trained models to the Hallym observation process, on development (MIMIC-IV) patients.
Views: absent-only (Hallym's never-recorded variables removed), cadence-only (patch-scale thinning of tests, charts and
vital signs; nothing removed), full Hallym profile. Metrics on 24-h risk: AUROC, Spearman rank stability factual->view,
mean |delta logit|. Same patients and same imagined views for every model."""
import sys, copy, numpy as np, torch, train as T, obsprofile as PR
from eval_hallym import frozen_preds
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as A
tags = sys.argv[1].split(",")
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"])
rs = np.random.RandomState(0); idx = np.sort(rs.choice(len(d["x"]), 20000, replace=False))
acu = PR.BlockAcuity().fit(d["x"][idx], d["yhosp"][idx]); devp = PR.estimate(d["x"][idx], d["m"][idx], acu)
sites = [PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp) for sx, sm in PR.target_sites("hallym_dyn:sites")]
def view(kind):
    x, m = d["x"][idx], d["m"][idx]; xs, ms = np.empty_like(x), np.empty_like(m); r = np.random.RandomState(1)
    for k, part in enumerate(np.array_split(np.arange(len(idx)), len(sites))):
        p = copy.deepcopy(sites[k])
        if kind == "absent":   p.rho = devp.rho.copy(); parts = ()
        if kind == "cadence":  p.rho = devp.rho.copy(); p.absent = []; parts = ("cad", "doccad", "vitcad")
        if kind == "full":     parts = ("sel", "cad", "vitcad", "doccad")
        xs[part], ms[part] = PR.imagine_two(x[part], m[part], acu, devp, p, parts=parts, rng=r)
    return xs, ms
V = {k: view(k) for k in ("absent", "cadence", "full")}
y = d["y24"][idx]
# dataset dicts for frozen_preds
base = {k: d[k][idx] for k in ("stay_id", "t_event", "event", "y24", "y48", "y168", "yhosp")}
DS = {"factual": {**base, "x": d["x"][idx], "m": d["m"][idx]}}
for k, (xs, ms) in V.items(): DS[k] = {**base, "x": xs, "m": ms}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
print(f"{'model':16s} " + "  ".join(f"{k:>26s}" for k in ("absent", "cadence", "full")) + "   (AUROC view [factual] | Spearman | mean|dlogit|)")
for t in tags:
    P = {k: frozen_preds(t, ds, st, "cuda")[:, 0] for k, ds in DS.items()}
    a0 = A(y, P["factual"])
    print(f"{t:16s} AUROC factual {a0:.3f} | " + "  ".join(f"{k}: {A(y, P[k]):.3f} rho {spearmanr(P['factual'], P[k])[0]:.3f} dl {np.abs(lg(P[k]) - lg(P['factual'])).mean():.3f}" for k in V), flush=True)
