"""Proposed model vs rate-matched thinning on alert efficiency, calibration and net benefit (24-h death, five-seed
ensembles), with the definitions of clinical_utility.py: deployable threshold = 10% alert rate on the held-out
development fold; target-tuned operating points as reference. Cluster bootstrap of proposed minus thinning.
Writes results/thin_utility_W.json."""
import os
import json
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric

S = (2, 3, 4, 5, 6)
PTS = np.round(np.arange(0.01, 0.2501, 0.005), 3)
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
cache = f"{T.ROOT}/results/hallym_W_preds.npz"
HP = dict(np.load(cache)) if os.path.exists(cache) else {}


def preds(tag, tgt):
    oof = np.nanmean([np.load(f"{T.ROOT}/results/preds/{tag}_s{s}.npz")["oof"] for s in S], 0)[:, 0]
    if tgt == "E":
        p = np.mean([np.load(f"{T.ROOT}/results/preds/{tag}_s{s}.npz")["ext"] for s in S], 0)[low, 0]
    else:
        if tag not in HP:
            HP[tag] = np.mean([frozen_preds(f"{tag}_s{s}", h, st, "cuda") for s in S], 0)
        p = HP[tag][:, 0]
    return oof[~np.isnan(oof)], p


def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    X = np.c_[np.ones_like(lp), lp]; b = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ b))); b = b + np.linalg.solve((X * (mu * (1 - mu))[:, None]).T @ X + 1e-9 * np.eye(2), X.T @ (y - mu))
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp))); a += (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-9)
    return {"intercept": a, "slope": b[1], "brier": float(np.mean((p - y) ** 2)), "oe": float(y.sum() / p.sum())}


def nb(y, p, pt):
    a = p >= pt
    return ((a & (y == 1)).sum() - (a & (y == 0)).sum() * pt / (1 - pt)) / len(y)


def stay_detect(ds, m, p, t):
    sid, L, y, a = ds["stay_id"][m], ds["landmark"][m], ds["y24"][m], p >= t
    pos = det = 0
    for s_ in np.unique(sid[y == 1]):
        k = np.where(sid == s_)[0]
        Lp = L[k][y[k] == 1].min()
        pos += 1; det += int((a[k] & (L[k] <= Lp)).any())
    return det, pos


out = {}
c = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]"
for tgt, models in (("E", {"thinning": "W_thinE", "proposed": "W_propE"}), ("H", {"thinning": "W_thinH", "proposed": "W_propH"})):
    ds, m, cl = (e, low, e["site"][low]) if tgt == "E" else (h, np.ones(len(h["y24"]), bool), h["subject_id"])
    y = ds["y24"][m].astype(int)
    R = {}
    for g, tag in models.items():
        po, p = preds(tag, tgt)
        t10 = float(np.quantile(po, 0.9))
        a = p >= t10
        det, pos = stay_detect(ds, m, p, t10)
        R[g] = {"p": p, "t10": t10, "calibration": calib(y, p),
                "deployable10": {"alert_rate": float(a.mean()), "sensitivity": float((a & (y == 1)).sum() / y.sum()), "ppv": float((a & (y == 1)).sum() / max(a.sum(), 1)), "stays_detected": det, "positive_stays": pos},
                "oracle10_sensitivity": float(((p >= np.quantile(p, 0.9)) & (y == 1)).sum() / y.sum()),
                "alerts_per100_sens80": float(100 * (p >= np.quantile(p[y == 1], 0.2)).mean()),
                "nb_curve": [float(nb(y, p, pt)) for pt in PTS]}
    idx = [i for i in boot_idx(cl, 1000, np.random.RandomState(0)) if 0 < y[i].sum() < len(i)]
    pp, pt_ = R["proposed"]["p"], R["thinning"]["p"]; tp, tt = R["proposed"]["t10"], R["thinning"]["t10"]
    F = {"deployable10_alert_rate": (lambda yy, s: (s >= tp).mean(), lambda yy, s: (s >= tt).mean()),
         "deployable10_sensitivity": (lambda yy, s: ((s >= tp) & (yy == 1)).sum() / max(yy.sum(), 1), lambda yy, s: ((s >= tt) & (yy == 1)).sum() / max(yy.sum(), 1)),
         "deployable10_ppv": (lambda yy, s: ((s >= tp) & (yy == 1)).sum() / max((s >= tp).sum(), 1), lambda yy, s: ((s >= tt) & (yy == 1)).sum() / max((s >= tt).sum(), 1)),
         "oracle10_sensitivity": (lambda yy, s: ((s >= np.quantile(s, 0.9)) & (yy == 1)).sum() / max(yy.sum(), 1),) * 2,
         "net_benefit_0.05": (lambda yy, s: nb(yy, s, 0.05),) * 2, "net_benefit_0.10": (lambda yy, s: nb(yy, s, 0.10),) * 2,
         "brier": (lambda yy, s: np.mean((s - yy) ** 2),) * 2,
         "calibration_slope": (lambda yy, s: calib(yy, s)["slope"],) * 2, "calibration_intercept": (lambda yy, s: calib(yy, s)["intercept"],) * 2}
    diffs = {}
    for k, (fp, ft) in F.items():
        bs = boot_metric(fp, y, pp, idx) - boot_metric(ft, y, pt_, idx)
        diffs[k] = [float(fp(y, pp) - ft(y, pt_)), *np.percentile(bs, [2.5, 97.5]).tolist()]
    nbp, nbt = np.array(R["proposed"]["nb_curve"]), np.array(R["thinning"]["nb_curve"])
    diffs["nb_thresholds_proposed_higher"] = int((nbp > nbt).sum()); diffs["nb_thresholds_total"] = len(PTS)
    for g in R:
        R[g].pop("p")
    out[tgt] = {"models": R, "diff_proposed_minus_thinning": diffs}
    print(f"===== {tgt}")
    for g in R:
        r = R[g]; dp = r["deployable10"]; cb = r["calibration"]
        print(f"  {g:9s} deployable10: alerts {100 * dp['alert_rate']:.1f}% sens {dp['sensitivity']:.3f} PPV {dp['ppv']:.3f} stays {dp['stays_detected']}/{dp['positive_stays']} | oracle10 sens {r['oracle10_sensitivity']:.3f} | alerts/100 @80% {r['alerts_per100_sens80']:.1f} | "
              f"cal int {cb['intercept']:+.2f} slope {cb['slope']:.2f} Brier {cb['brier']:.4f} O/E {cb['oe']:.2f}")
    for k, v in diffs.items():
        print(f"  proposed - thinning {k}: {c(v) if isinstance(v, list) else v}")
np.savez(cache, **HP)
json.dump(out, open(f"{T.ROOT}/results/thin_utility_W.json", "w"), indent=1, default=float)
print("written results/thin_utility_W.json")
