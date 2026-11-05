"""Reviewer round 3 analyses (hidden size 256).
(a) Confirmation on hospitals never used in design: eICU-CRD middle testing tertile; proposed model targeted at these
    hospitals (W_propM) vs backbone, generic training and the model targeted at low-testing hospitals (W_propE).
(b) Closest simple alternative: rate-matched thinning (W_thinE / W_thinH) vs proposed vs backbone.
(c) Full-scale patch-scale interaction: (proposed - selection-only | reset) - (same | no reset), seeds 2-4.
Cluster bootstrap (hospitals / patients), 1,000 resamples. Writes results/rev3_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

EPS = ["y24", "y48", "y168", "yhosp"]
S5, S3 = (2, 3, 4, 5, 6), (2, 3, 4)
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn")


def tertile(name):
    site = e["site"]; crit = [7, 13, 14, 16, 20, 21]
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    t = np.full(len(hr), None, object); t[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    return t == name


MASK = {"E": low_tertile(e), "M": tertile("mid")}
_hc = {}


def pred(tag, tgt, seeds):
    if tgt == "H":
        key = (tag, seeds)
        if key not in _hc:
            _hc[key] = np.mean([frozen_preds(f"{tag}_s{s}", h, st, "cuda") for s in seeds], 0)
        return _hc[key]
    return np.mean([np.load(f"{T.ROOT}/results/preds/{tag}_s{s}.npz")["ext"] for s in seeds], 0)[MASK[tgt]]


def setup(tgt):
    if tgt == "H":
        return h, np.ones(len(h["y24"]), bool), h["subject_id"]
    return e, MASK[tgt], e["site"][MASK[tgt]]


def compare(tgt, models, seeds, pairs):
    ds, m, cl = setup(tgt)
    P = {g: pred(t, tgt, seeds) for g, t in models.items()}
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    res = {"n": int(m.sum()), "hospitals_or_patients": int(len(np.unique(cl)))}
    for j, ep in enumerate(EPS):
        y = ds[ep][m]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        res[ep] = {g: [A(y, P[g][:, j]), AP(y, P[g][:, j]), float(y.sum() / P[g][:, j].sum())] for g in models}
        for a_, b_ in pairs:
            for mn, f in (("auroc", A), ("auprc", AP)):
                bs = boot_metric(f, y, P[a_][:, j], ii) - boot_metric(f, y, P[b_][:, j], ii)
                res[ep][f"{a_}-{b_}_{mn}"] = [f(y, P[a_][:, j]) - f(y, P[b_][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
    return res


out = {}
c = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]"
# (a) confirmation
out["confirm_mid"] = compare("M", {"backbone": "W_base", "generic": "W_gen", "proposed_low": "W_propE", "proposed_mid": "W_propM"}, S5,
                             [("proposed_mid", "backbone"), ("proposed_mid", "generic"), ("proposed_low", "backbone")])
for ep in EPS:
    r = out["confirm_mid"][ep]
    print(f"[confirm mid {ep}] " + "  ".join(f"{g} {v[0]:.3f}/{v[1]:.3f}" for g, v in r.items() if isinstance(v, list) and len(v) == 3)
          + f" | mid-base AUROC {c(r['proposed_mid-backbone_auroc'])} AUPRC {c(r['proposed_mid-backbone_auprc'])} | mid-generic AUROC {c(r['proposed_mid-generic_auroc'])}", flush=True)
# (b) rate-matched thinning
for tgt, prop, thin in (("E", "W_propE", "W_thinE"), ("H", "W_propH", "W_thinH")):
    out[f"thin_{tgt}"] = compare(tgt, {"backbone": "W_base", "thinning": thin, "proposed": prop}, S5, [("proposed", "thinning"), ("thinning", "backbone")])
    for ep in EPS:
        r = out[f"thin_{tgt}"][ep]
        print(f"[thin {tgt} {ep}] base {r['backbone'][0]:.3f}/{r['backbone'][1]:.3f} thin {r['thinning'][0]:.3f}/{r['thinning'][1]:.3f} prop {r['proposed'][0]:.3f}/{r['proposed'][1]:.3f} | "
              f"prop-thin AUROC {c(r['proposed-thinning_auroc'])} AUPRC {c(r['proposed-thinning_auprc'])} | thin-base AUROC {c(r['thinning-backbone_auroc'])}", flush=True)
# (c) full-scale patch-scale interaction
for tgt, pr, se, npr, nse in (("E", "W_propE", "W_selE", "W_noresetPropE", "W_noresetSelE"), ("H", "W_propH", "W_selH", "W_noresetPropH", "W_noresetSelH")):
    ds, m, cl = setup(tgt)
    P = {k: pred(t, tgt, S3) for k, t in (("Rf", pr), ("Rs", se), ("Nf", npr), ("Ns", nse))}
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    for j, ep in enumerate(EPS):
        y = ds[ep][m]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        for mn, f in (("auroc", A), ("auprc", AP)):
            bs = {k: boot_metric(f, y, P[k][:, j], ii) for k in P}
            pt = {k: f(y, P[k][:, j]) for k in P}
            gR, gN = bs["Rf"] - bs["Rs"], bs["Nf"] - bs["Ns"]
            q = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
            rr = {"patch_gain_reset": [pt["Rf"] - pt["Rs"], *q(gR)], "patch_gain_noreset": [pt["Nf"] - pt["Ns"], *q(gN)],
                  "interaction": [(pt["Rf"] - pt["Rs"]) - (pt["Nf"] - pt["Ns"]), *q(gR - gN)]}
            out[f"patch_inter_{tgt}_{ep}_{mn}"] = rr
            print(f"[patch-inter {tgt} {ep} {mn}] gain R {c(rr['patch_gain_reset'])} N {c(rr['patch_gain_noreset'])} interaction {c(rr['interaction'])}", flush=True)
json.dump(out, open(f"{T.ROOT}/results/rev3_W.json", "w"), indent=1, default=float)
print("written results/rev3_W.json")
