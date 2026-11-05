"""Hallym death-time uncertainty, evaluated directly. The death DATE comes from the discharge record (death_date);
only the hour within that date is unknown. For every in-hospital death the death time is drawn uniformly from the
interval that the records allow: from max(00:00 of the death date, last vital-sign record) to 24:00 of the death date
(the patient was alive while monitored; landmarks precede the last record, so the sample set is unchanged).
Each of 2,000 patient-cluster bootstrap resamples uses its own draw, so the interval combines sampling and death-time
uncertainty. Frozen five-seed ensembles (backbone W_base, proposed W_propH, rate-level W_thinH); no retraining.
Writes results/hallym_time_mi_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
from eval_hallym import boot_idx
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

RAW = f"{T.ROOT}/data/hallym_raw"
h = T.load("hallym_dyn")
coh = pd.read_parquet(f"{T.ROOT}/data/hallym_cohort72.parquet").set_index("stay_id")
P = dict(np.load(f"{T.ROOT}/results/hallym_W_preds.npz"))
MODELS = {"backbone": "W_base", "proposed": "W_propH", "rate_level": "W_thinH", "generic": "W_gen"}

dd = {}
for s in coh["hospital"].unique():
    d = pd.read_csv(f"{RAW}/{s}/discharge.csv", low_memory=False, usecols=["hadm_id", "death_flag", "death_date"])
    d["stay_id"] = s + "_" + d["hadm_id"].astype(str)
    dd.update(dict(zip(d["stay_id"], pd.to_datetime(d["death_date"], errors="coerce"))))
coh["death_date"] = coh.index.map(dd)
dead = coh["t_death_h"].notna()
t_death = coh["t0"] + pd.to_timedelta(coh["t_death_h"], unit="h")
t_disch = coh["t0"] + pd.to_timedelta(coh["t_disch_h"], unit="h")
# last vital record: equals the default death time when it fell on the death date or the day before, else unknown
# (then the default is noon and monitoring ended earlier, so the lower bound is 00:00 of the death date)
from_last = dead & (t_death.dt.normalize() >= coh["death_date"] - pd.Timedelta(days=1)) & (t_death.dt.normalize() <= coh["death_date"]) \
            & ~((t_death - t_death.dt.normalize()) == pd.Timedelta(hours=12))
day0 = coh["death_date"]
lo = np.where(from_last, np.maximum(t_death, day0), day0)
lo = pd.to_datetime(pd.Series(lo, index=coh.index))
hi = day0 + pd.Timedelta(hours=24)
ok = dead & coh["death_date"].notna() & (hi > lo)
lo_h = ((lo - coh["t0"]).dt.total_seconds() / 3600)[ok]; hi_h = ((hi - coh["t0"]).dt.total_seconds() / 3600)[ok]
info = {"deaths": int(dead.sum()), "with_interval": int(ok.sum()), "from_last_vital": int((from_last & ok).sum()),
        "interval_hours_median": float((hi_h - lo_h).median()), "interval_hours_iqr": [float(x) for x in (hi_h - lo_h).quantile([.25, .75])]}

sid = h["stay_id"].astype(str); L = h["landmark"].astype(float)
# alive at every landmark by construction: the earliest possible death time is never before the last landmark of the stay
maxL = pd.Series(L).groupby(sid).max()
raised = lo_h < maxL.reindex(lo_h.index).fillna(-np.inf)
lo_h = np.maximum(lo_h, maxL.reindex(lo_h.index).fillna(-np.inf)); info["lower_bound_raised_to_last_landmark"] = int(raised.sum())
pos = pd.Series(np.arange(len(coh)), index=coh.index)
lo_s = lo_h.reindex(sid).values; hi_s = hi_h.reindex(sid).values; has = ~np.isnan(lo_s)
print(info, flush=True)
assert np.all(L[has] <= lo_s[has] + 1e-6), "a landmark lies after the earliest possible death time"
H = {"y24": 24, "y48": 48, "y168": 168}
dflt = {ep: h[ep].astype(int) for ep in H}


def labels(td):
    """td: death hour from t0 per sample (nan = alive); labels for the fixed sample set."""
    return {ep: np.where(np.isnan(td), 0, ((td > L) & (td <= L + w)).astype(int)) for ep, w in H.items()}


rs = np.random.RandomState(0)
u_of_stay = lambda: pd.Series(rs.rand(len(lo_h)), index=lo_h.index)
cl = h["subject_id"]; idx = list(boot_idx(cl, 2000, np.random.RandomState(1)))
out = {"info": info}
for j, ep in enumerate(H):
    vals, gains = {g: [] for g in MODELS}, {k: [] for k in ("proposed-backbone", "rate_level-backbone", "proposed-rate_level", "proposed-generic")}
    draws_pos = []
    for b, ii in enumerate(idx):
        u = u_of_stay()
        td = (lo_h + u * (hi_h - lo_h)).reindex(sid).values
        y = labels(td)[ep]
        draws_pos.append(int(y.sum()))
        yb = y[ii]
        if not 0 < yb.sum() < len(yb):
            continue
        r = {g: (A(yb, P[t][ii, j]), AP(yb, P[t][ii, j])) for g, t in MODELS.items()}
        for k in gains:
            a_, b_ = k.split("-"); gains[k].append([r[a_][0] - r[b_][0], r[a_][1] - r[b_][1]])
    out[ep] = {"events_default": int(dflt[ep].sum()), "events_imputed_median": float(np.median(draws_pos)),
               "events_imputed_range": [int(min(draws_pos)), int(max(draws_pos))]}
    for k, v in gains.items():
        v = np.array(v)
        out[ep][k] = {"auroc": [float(np.median(v[:, 0])), *np.percentile(v[:, 0], [2.5, 97.5]).tolist()],
                      "auprc": [float(np.median(v[:, 1])), *np.percentile(v[:, 1], [2.5, 97.5]).tolist()],
                      "share_auroc_gt0": float((v[:, 0] > 0).mean())}
    c = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]"
    print(f"[{ep}] events default {out[ep]['events_default']} imputed {out[ep]['events_imputed_median']:.0f} {out[ep]['events_imputed_range']} | "
          + " | ".join(f"{k} AUROC {c(out[ep][k]['auroc'])} AUPRC {c(out[ep][k]['auprc'])}" for k in gains), flush=True)
json.dump(out, open(f"{T.ROOT}/results/hallym_time_mi_W.json", "w"), indent=1)
print("written results/hallym_time_mi_W.json")
