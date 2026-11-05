"""Profile-guided choice between the three-scale profile and rate-level imagination, frozen before application.
Rule (from the low-testing tertile only; threshold = its median label-free selection-slope divergence, 0.48, the split
of Supplementary S14): a hospital whose divergence is at most 0.48 gets the three-scale model, otherwise rate-level.
Applied unchanged, hospital by hospital, to the middle-testing eICU-CRD tertile and to the four Hallym hospitals, neither
of which contributed to the rule. Compared with always-three-scale and always-rate-level (five-seed ensembles; hospital /
patient cluster bootstrap, 1,000). Three-scale = target profiles only (equal exposure) and, separately, the main design.
Writes results/rule_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
from eval_hallym import frozen_preds, boot_idx
from bootstrap_par import boot_metric
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

THR = 0.48
S = (2, 3, 4, 5, 6); EPS = ["y24", "y48", "y168", "yhosp"]
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn")
div_e = {int(float(k)): v for k, v in json.load(open(f"{T.ROOT}/results/selslope_W.json"))["divergence"].items()}
div_h = {k: v["divergence"] for k, v in json.load(open(f"{T.ROOT}/results/selslope_hallym_W.json")).items() if isinstance(v, dict) and "divergence" in v}


def tertile(name):
    site = e["site"]; crit = [7, 13, 14, 16, 20, 21]
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    t = np.full(len(hr), None, object); t[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    return t == name


out = {"threshold": THR}
for tgt in ("M", "H"):
    if tgt == "M":
        m = tertile("mid"); ds, site, cl = e, e["site"][m], e["site"][m]
        ld = lambda tag: np.mean([np.load(f"{T.ROOT}/results/preds/{tag}_s{s}.npz")["ext"] for s in S], 0)[m]
        P = {"three_T": ld("W_propTM"), "three_main": ld("W_propM"), "rate": ld("W_thinM")}
        div = np.array([div_e.get(int(s), np.nan) for s in site])
    else:
        m = np.ones(len(h["y24"]), bool); ds, site, cl = h, h["site"].astype(str), h["subject_id"]
        P = {"three_T": np.mean([frozen_preds(f"W_propTH_s{s}", h, st, "cuda") for s in S], 0),
             "three_main": dict(np.load(f"{T.ROOT}/results/hallym_W_preds.npz"))["W_propH"],
             "rate": dict(np.load(f"{T.ROOT}/results/hallym_W_preds.npz"))["W_thinH"]}
        div = np.array([div_h.get(s, np.nan) for s in site])
    use3 = np.where(np.isnan(div), False, div <= THR)             # no usable laboratory slopes -> rate-level
    hs = pd.Series(use3, index=site).groupby(level=0).first()
    R = {"hospitals": int(len(hs)), "three_scale_hospitals": int(hs.sum()), "samples_three_scale": float(use3.mean())}
    for kind in ("three_T", "three_main"):
        P[f"guided_{kind}"] = np.where(use3[:, None], P[kind], P["rate"])
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    for j, ep in enumerate(EPS):
        y = ds[ep][m]; ii = [i for i in idx if 0 < y[i].sum() < len(i)]
        r = {g: [A(y, p[:, j]), AP(y, p[:, j])] for g, p in P.items()}
        for kind in ("three_T", "three_main"):
            for other in (kind, "rate"):
                for mn, f in (("auroc", A), ("auprc", AP)):
                    bs = boot_metric(f, y, P[f"guided_{kind}"][:, j], ii) - boot_metric(f, y, P[other][:, j], ii)
                    r[f"guided_{kind}-{other}_{mn}"] = [f(y, P[f"guided_{kind}"][:, j]) - f(y, P[other][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
        R[ep] = r
        c = lambda k: f"{r[k][0]:+.3f} [{r[k][1]:+.3f},{r[k][2]:+.3f}]"
        print(f"[{tgt} {ep}] three_T {r['three_T'][0]:.3f}/{r['three_T'][1]:.3f} rate {r['rate'][0]:.3f}/{r['rate'][1]:.3f} guided_T {r['guided_three_T'][0]:.3f}/{r['guided_three_T'][1]:.3f} | "
              f"gT-3T AUROC {c('guided_three_T-three_T_auroc')} AUPRC {c('guided_three_T-three_T_auprc')} | gT-rate AUROC {c('guided_three_T-rate_auroc')} AUPRC {c('guided_three_T-rate_auprc')}", flush=True)
    print(tgt, {k: v for k, v in R.items() if not k.startswith("y")}, flush=True)
    out[tgt] = R
json.dump(out, open(f"{T.ROOT}/results/rule_W.json", "w"), indent=1, default=float)
print("written results/rule_W.json")
