"""Which signal does the reset model read in addition to the no-reset model, and does it lose its meaning at Hallym?
Per sample, delta = logit(reset) - logit(no reset) of the 24-h risk (three-seed ensembles, W_base vs W_noreset).
Summary features of the last 6 h (one reset patch) and of the 48-h window are computed per vital sign. On MIMIC-IV
(development; 30,000 random samples) and Hallym: (i) Spearman correlation of each feature with delta (what the reset
adds), (ii) association of each feature with 24-h death (AUROC of the feature), compared between sites. A signal that
drives delta and is predictive in MIMIC-IV but not at Hallym is a candidate cause. Writes results/reset_signal_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
from eval_hallym import frozen_preds
from sklearn.metrics import roc_auc_score as A

VARS = ["rr", "temp", "hr", "spo2", "sbp", "dbp"]
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn"); S = (2, 3, 4)
rs = np.random.RandomState(0); mi = np.sort(rs.choice(len(d["stay_id"]), 30000, replace=False))
dm = {k: (v[mi] if hasattr(v, "__len__") and len(v) == len(d["stay_id"]) else v) for k, v in d.items()}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
ens = lambda tag, ds: np.mean([frozen_preds(f"{tag}_s{s}", ds, st, "cuda") for s in S], 0)[:, 0]


def feats(ds):
    x, m = ds["x"], ds["m"]
    F = {}
    for c, v in enumerate(VARS):
        xo = np.where(m[:, :, c] > 0, x[:, :, c], np.nan)
        for name, sl in (("last6", slice(42, 48)), ("all48", slice(0, 48))):
            w = xo[:, sl]
            with np.errstate(all="ignore"):
                F[f"{v}_{name}_mean"] = np.nanmean(w, 1); F[f"{v}_{name}_sd"] = np.nanstd(w, 1)
                F[f"{v}_{name}_min"] = np.nanmin(w, 1); F[f"{v}_{name}_max"] = np.nanmax(w, 1)
                F[f"{v}_{name}_absdiff"] = np.nanmean(np.abs(np.diff(w, axis=1)), 1)
                F[f"{v}_{name}_n"] = (~np.isnan(w)).sum(1).astype(float)
            t = np.arange(w.shape[1]); ok = ~np.isnan(w)
            num = np.nansum((t - t.mean()) * (w - np.nanmean(w, 1, keepdims=True)), 1); den = np.sum(((t - t.mean()) ** 2) * ok, 1)
            F[f"{v}_{name}_slope"] = np.where(ok.sum(1) >= 3, num / np.maximum(den, 1e-9), np.nan)
        lastv = pd.DataFrame(xo).ffill(axis=1).iloc[:, -1].values
        F[f"{v}_last_minus_mean48"] = lastv - F[f"{v}_all48_mean"]
    return pd.DataFrame(F)


out = {}
for site, ds in (("mimic", dm), ("hallym", h)):
    dl = logit(ens("W_base", ds)) - logit(ens("W_noreset", ds))
    F = feats(ds); y = ds["y24"].astype(int)
    rows = []
    for c in F.columns:
        f = F[c].values; ok = ~np.isnan(f)
        if ok.sum() < 200 or y[ok].sum() < 10 or np.nanstd(f) == 0:
            continue
        rows.append({"feature": c, "rho_delta": float(pd.Series(f[ok]).corr(pd.Series(dl[ok]), method="spearman")),
                     "auroc_death": float(A(y[ok], f[ok])), "coverage": float(ok.mean())})
    out[site] = {"delta_mean": float(dl.mean()), "delta_sd": float(dl.std()), "delta_auroc_death": float(A(y, dl)),
                 "features": rows}
    print(f"[{site}] delta logit mean {dl.mean():+.3f} sd {dl.std():.3f} | AUROC of delta for death {A(y, dl):.3f}", flush=True)
M = pd.DataFrame(out["mimic"]["features"]).set_index("feature"); H = pd.DataFrame(out["hallym"]["features"]).set_index("feature")
J = M.join(H, lsuffix="_mimic", rsuffix="_hallym", how="inner")
J["drives_delta"] = J[["rho_delta_mimic", "rho_delta_hallym"]].abs().mean(1)
J["death_assoc_change"] = (J["auroc_death_hallym"] - 0.5) - (J["auroc_death_mimic"] - 0.5)
J["sign_flip"] = np.sign(J["auroc_death_hallym"] - 0.5) != np.sign(J["auroc_death_mimic"] - 0.5)
top = J.sort_values("drives_delta", ascending=False).head(15)
print(top[["rho_delta_mimic", "rho_delta_hallym", "auroc_death_mimic", "auroc_death_hallym", "sign_flip"]].round(3).to_string(), flush=True)
out["joined_top"] = top.reset_index().to_dict(orient="records")
json.dump(out, open(f"{T.ROOT}/results/reset_signal_W.json", "w"), indent=1, default=float)
print("written results/reset_signal_W.json")
