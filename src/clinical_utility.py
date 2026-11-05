"""Clinical utility of 24-h mortality alerts (hidden-256 models, five-seed ensembles) at low-testing eICU-CRD hospitals and
the Hallym hospitals.

Thresholds
  deployable : set on the held-out development fold only (alert rate q in MIMIC-IV), applied unchanged to the target;
  oracle     : set with the target's labels (alert rate q, or sensitivity s, in the target) - reference only.
Units
  landmark   : every 12-h reassessment is one alert opportunity (workload);
  stay       : a stay is positive if death follows any of its landmarks within 24 h; it is detected if an alert was raised
               at that landmark or an earlier one; lead time = death time minus the first alert.
Also calibration (intercept, slope, Brier, O/E) and net benefit. Cluster bootstrap (hospitals in eICU-CRD, patients in
Hallym) of proposed minus backbone. Writes results/utility_W.json."""
import os
import json
import numpy as np
import train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from bootstrap_par import boot_metric

S = (2, 3, 4, 5, 6)
J = 0                                              # 24-h death
QS, SENS, PTS = (0.05, 0.10, 0.20), (0.5, 0.7, 0.8), (0.02, 0.05, 0.10, 0.15, 0.20)
GROUPS = {"E": {"backbone": "W_base", "generic": "W_gen", "selection_only": "W_selE", "thinning": "W_thinE", "proposed": "W_propE"},
          "H": {"backbone": "W_base", "generic": "W_gen", "selection_only": "W_selH", "thinning": "W_thinH", "proposed": "W_propH"}}
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
cache = f"{T.ROOT}/results/hallym_W_preds.npz"
HP = dict(np.load(cache)) if os.path.exists(cache) else {}


def preds(tag, tgt):
    oof = np.nanmean([np.load(f"{T.ROOT}/results/preds/{tag}_s{s}.npz")["oof"] for s in S], 0)[:, J]
    if tgt == "E":
        ext = np.mean([np.load(f"{T.ROOT}/results/preds/{tag}_s{s}.npz")["ext"] for s in S], 0)[low, J]
    else:
        if tag not in HP:
            HP[tag] = np.mean([frozen_preds(f"{tag}_s{s}", h, st, "cuda") for s in S], 0)
        ext = HP[tag][:, J]
    return oof[~np.isnan(oof)], ext


def at(y, p, t):
    a = p >= t
    tp = (a & (y == 1)).sum()
    return {"alert_rate": a.mean(), "sensitivity": tp / max(y.sum(), 1), "ppv": tp / max(a.sum(), 1), "alerts_per_100": 100 * a.mean()}


def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    b = np.zeros(2)                                    # slope: logit(y) ~ b0 + b1 * lp (Newton)
    X = np.c_[np.ones_like(lp), lp]
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (y - mu); H = (X * (mu * (1 - mu))[:, None]).T @ X
        b = b + np.linalg.solve(H + 1e-9 * np.eye(2), g)
    a = 0.0                                            # calibration-in-the-large: intercept with offset lp
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp)))
        a += (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-9)
    return {"intercept": a, "slope": b[1], "brier": np.mean((p - y) ** 2), "oe": y.sum() / p.sum()}


def nb(y, p, pt):
    a = p >= pt
    n = len(y)
    return ((a & (y == 1)).sum() - (a & (y == 0)).sum() * pt / (1 - pt)) / n


def stay_level(ds, m, p, t):
    sid, L, te, y = ds["stay_id"][m], ds["landmark"][m], ds["t_event"][m], ds["y24"][m]
    o = np.lexsort((L, sid))
    sid, L, te, y, a = sid[o], L[o], te[o], y[o], (p >= t)[o]
    pos = det = 0; leads = []; n_alert = []
    for s_ in np.unique(sid):
        k = np.where(sid == s_)[0]
        n_alert.append(a[k].sum())
        if y[k].any():
            pos += 1
            first_pos = k[np.argmax(y[k])]
            al = k[(a[k]) & (L[k] <= L[first_pos])]
            if len(al):
                det += 1; leads.append(te[first_pos] + L[first_pos] - L[al[0]])   # t_event is time from its own landmark
    return {"stays": int(len(np.unique(sid))), "positive_stays": pos, "detected": det, "stay_sensitivity": det / max(pos, 1),
            "median_lead_h": float(np.median(leads)) if leads else None, "alerts_per_stay": float(np.mean(n_alert))}


out = {}
for tgt, groups in GROUPS.items():
    ds, m, cl = (e, low, e["site"][low]) if tgt == "E" else (h, np.ones(len(h["y24"]), bool), h["subject_id"])
    y = ds["y24"][m].astype(int)
    idx = [i for i in boot_idx(cl, 1000, np.random.RandomState(0)) if 0 < y[i].sum() < len(i)]
    R = {}
    for g, tag in groups.items():
        po, p = preds(tag, tgt)
        r = {"n": int(len(y)), "events": int(y.sum()), "calibration": calib(y, p)}
        r["deployable"] = {}
        for q in QS:
            t = float(np.quantile(po, 1 - q))
            r["deployable"][str(q)] = {"threshold": t, **at(y, p, t), "stay": stay_level(ds, m, p, t)}
        r["oracle_rate"] = {str(q): at(y, p, float(np.quantile(p, 1 - q))) for q in QS}
        r["oracle_sens"] = {}
        for s_ in SENS:
            t = float(np.quantile(p[y == 1], 1 - s_))
            r["oracle_sens"][str(s_)] = at(y, p, t)
        r["net_benefit"] = {str(pt): nb(y, p, pt) for pt in PTS}
        r["_p"], r["_t10"] = p, float(np.quantile(po, 0.9))
        R[g] = r
        dp = r["deployable"]["0.1"]
        print(f"[{tgt}] {g:15s} cal slope {r['calibration']['slope']:.2f} int {r['calibration']['intercept']:+.2f} Brier {r['calibration']['brier']:.4f} O/E {r['calibration']['oe']:.2f} | "
              f"deployable 10%: alert {100 * dp['alert_rate']:.1f}% sens {dp['sensitivity']:.3f} PPV {dp['ppv']:.3f}; stay sens {dp['stay']['stay_sensitivity']:.3f} lead {dp['stay']['median_lead_h']} h | "
              f"oracle 10%: sens {r['oracle_rate']['0.1']['sensitivity']:.3f} PPV {r['oracle_rate']['0.1']['ppv']:.3f} | sens 0.8 needs alerts {r['oracle_sens']['0.8']['alerts_per_100']:.1f}/100 | "
              f"NB@0.05 {r['net_benefit']['0.05']:.4f}", flush=True)
    prev = y.mean()
    treat_all = {str(pt): prev - (1 - prev) * pt / (1 - pt) for pt in PTS}
    # bootstrap: proposed minus backbone
    pb, pp = R["backbone"]["_p"], R["proposed"]["_p"]
    tb, tp_ = R["backbone"]["_t10"], R["proposed"]["_t10"]
    diffs = {}
    for name, f_b, f_p in (("deployable10_sensitivity", lambda yy, s: ((s >= tb) & (yy == 1)).sum() / max(yy.sum(), 1), lambda yy, s: ((s >= tp_) & (yy == 1)).sum() / max(yy.sum(), 1)),
                           ("deployable10_ppv", lambda yy, s: ((s >= tb) & (yy == 1)).sum() / max((s >= tb).sum(), 1), lambda yy, s: ((s >= tp_) & (yy == 1)).sum() / max((s >= tp_).sum(), 1)),
                           ("deployable10_alert_rate", lambda yy, s: (s >= tb).mean(), lambda yy, s: (s >= tp_).mean()),
                           ("oracle10_sensitivity", lambda yy, s: ((s >= np.quantile(s, 0.9)) & (yy == 1)).sum() / max(yy.sum(), 1), None),
                           ("net_benefit_0.05", lambda yy, s: nb(yy, s, 0.05), None), ("net_benefit_0.10", lambda yy, s: nb(yy, s, 0.10), None),
                           ("brier", lambda yy, s: np.mean((s - yy) ** 2), None)):
        f_p = f_p or f_b
        bs = boot_metric(f_p, y, pp, idx) - boot_metric(f_b, y, pb, idx)
        diffs[name] = [float(f_p(y, pp) - f_b(y, pb)), *np.percentile(bs, [2.5, 97.5]).tolist()]
        print(f"   proposed - backbone {name}: {diffs[name][0]:+.4f} [{diffs[name][1]:+.4f},{diffs[name][2]:+.4f}]", flush=True)
    for g in R:
        R[g].pop("_p"); R[g].pop("_t10")
    out[tgt] = {"models": R, "treat_all_net_benefit": treat_all, "prevalence": float(prev), "diff_proposed_minus_backbone": diffs}
np.savez(cache, **HP)
json.dump(out, open(f"{T.ROOT}/results/utility_W.json", "w"), indent=1, default=float)
print("written results/utility_W.json")
