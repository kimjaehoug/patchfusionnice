"""Analyses requested in review round 5 (hidden size 256, five seeds).
(a) Fidelity of imagined records: descriptive observation statistics of the actual target records, unchanged development
    records, the rate-level variant and the three-scale profile (target profiles, as used for the target half of training),
    per hospital (Hallym; low-testing eICU-CRD hospitals with >= 500 windows), averaged over hospitals.
(b) Seed variability: per-seed 24-h AUROC/AUPRC (mean, SD, range) of every method.
(c) Hospital-level change: per-hospital 24-h AUROC and AUPRC of the proposed model minus the backbone (five-seed
    ensembles), low-testing eICU-CRD hospitals with >= 10 deaths within 24 h, and each Hallym hospital.
Writes results/v11_W.json."""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
import train as T
import obsprofile as PR
from eval_hallym import frozen_preds
from link_eval import low_tertile
from regime import VARS, VIT

S = (2, 3, 4, 5, 6)
rng = np.random.RandomState(0)
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"])
e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
out = {}

# ---------------- (a) fidelity
i = rng.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i], d["yhosp"][i]); dv = PR.estimate(d["x"][i], d["m"][i], acu)
r_dev = PR.obs_rate(d["m"][i])
pool = rng.permutation(np.setdiff1d(np.arange(len(d["x"])), i))
LAB = PR.LAB


def stats(x, m):
    o = m > 0; lab = o[:, :, LAB]
    act = np.cumsum(o.any(2), 1) > 0
    tested = lab.any(1)                                                    # [N, L]
    r = {"lab_window_presence": float(tested.mean()),
         "lab_hourly_rate": float((lab & act[..., None]).sum() / max(act.sum() * len(LAB), 1)),
         "spo2_hourly_rate": float((o[:, :, VIT[3]] & act).sum() / max(act.sum(), 1))}
    gaps, runs = [], []
    for n in range(len(o)):
        t = np.where(lab[n].any(1))[0]
        if len(t) > 1:
            gaps.append(np.diff(t).mean())
        a = np.where(act[n])[0]
        if len(a):
            seg = ~lab[n, a[0]:].any(1); best = cur = 0
            for v in seg:
                cur = cur + 1 if v else 0; best = max(best, cur)
            runs.append(best)
    r["lab_mean_interval_h"] = float(np.mean(gaps)) if gaps else float("nan")
    r["lab_longest_gap_h"] = float(np.mean(runs))
    th = lab.any(2)
    r["lab_co_measurement"] = float(((lab.sum(2) >= 2) & th).sum() / max(th.sum(), 1))
    with np.errstate(all="ignore"):
        aw = np.nanmean(acu(x), 1)
    ok = ~np.isnan(aw); q = np.nanquantile(aw[ok], [1 / 3, 2 / 3])
    r["lab_presence_low_acuity"] = float(tested[ok & (aw <= q[0])].mean())
    r["lab_presence_high_acuity"] = float(tested[ok & (aw > q[1])].mean())
    return r


fid = {}
for tname, label in (("hallym_dyn:sites", "Hallym"), ("eicu_dyn:low:sites", "eICU-CRD low testing")):
    per = {k: [] for k in ("target", "development", "rate_level", "three_scale")}
    ptr = 0
    for sx, sm in PR.target_sites(tname):
        k = min(len(sx), 3000); ri = rng.permutation(len(sx))[:k]
        di = pool[ptr:ptr + k]; ptr += k
        xd, md = d["x"][di], d["m"][di]
        tp = PR.fit_window(PR.estimate(sx, sm, acu, beta=dv.beta), dv)
        per["target"].append(stats(sx[ri], sm[ri])); per["development"].append(stats(xd, md))
        per["rate_level"].append(stats(*PR.imagine_rate(xd, md, r_dev, PR.obs_rate(sm), rng)))
        per["three_scale"].append(stats(*PR.imagine_two(xd, md, acu, dv, tp, parts=["sel", "cad", "vitcad", "doccad"], rng=rng)))
    fid[label] = {g: {m_: float(np.nanmean([r[m_] for r in v])) for m_ in v[0]} for g, v in per.items()}
    print(f"\n[fidelity {label}]")
    for m_ in fid[label]["target"]:
        print(f"  {m_:26s} " + "  ".join(f"{g} {fid[label][g][m_]:.3f}" for g in fid[label]), flush=True)
out["fidelity"] = fid

# ---------------- (b) seed variability
G = {"E": {"backbone": "W_base", "reweighting": "W_rwE", "generic": "W_gen", "selection_only": "W_selE", "rate_level": "W_thinE", "proposed": "W_propE"},
     "H": {"backbone": "W_base", "reweighting": "W_rwH", "generic": "W_gen", "selection_only": "W_selH", "rate_level": "W_thinH", "proposed": "W_propH"}}
sv, HPS = {}, {}
for t, tags in G.items():
    sv[t] = {}
    for g, tg in tags.items():
        vals = []
        for s in S:
            if t == "E":
                p = np.load(f"{T.ROOT}/results/preds/{tg}_s{s}.npz")["ext"][low, 0]; y = e["y24"][low]
            else:
                p = frozen_preds(f"{tg}_s{s}", h, st, "cuda")[:, 0]; y = h["y24"]
            vals.append([A(y, p), AP(y, p)])
        v = np.array(vals)
        sv[t][g] = {"per_seed": v.tolist(), "mean": v.mean(0).tolist(), "sd": v.std(0, ddof=1).tolist(), "min": v.min(0).tolist(), "max": v.max(0).tolist()}
        print(f"[seeds {t}] {g:15s} AUROC {v[:, 0].mean():.3f} ± {v[:, 0].std(ddof=1):.3f} ({v[:, 0].min():.3f}-{v[:, 0].max():.3f})  AUPRC {v[:, 1].mean():.3f} ± {v[:, 1].std(ddof=1):.3f}", flush=True)
out["seed_variability"] = sv

# ---------------- (c) hospital-level change
HPc = dict(np.load(f"{T.ROOT}/results/hallym_W_preds.npz"))
pe = {g: np.mean([np.load(f"{T.ROOT}/results/preds/{tg}_s{s}.npz")["ext"] for s in S], 0)[:, 0] for g, tg in (("backbone", "W_base"), ("proposed", "W_propE"))}
rows = []
for s_ in np.unique(e["site"][low]):
    k = low & (e["site"] == s_); y = e["y24"][k]
    if y.sum() >= 10:
        rows.append([str(s_), int(k.sum()), int(y.sum()), A(y, pe["proposed"][k]) - A(y, pe["backbone"][k]), AP(y, pe["proposed"][k]) - AP(y, pe["backbone"][k])])
dA, dP = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
hosp = {"eicu_low": {"hospitals": len(rows), "auroc_improved": int((dA > 0).sum()), "auprc_improved": int((dP > 0).sum()),
                     "auroc_median": float(np.median(dA)), "auroc_iqr": np.percentile(dA, [25, 75]).tolist(),
                     "auprc_median": float(np.median(dP)), "auprc_iqr": np.percentile(dP, [25, 75]).tolist(), "rows": rows}}
print(f"\n[hospitals eICU low, >=10 deaths] n={len(rows)} AUROC improved {(dA > 0).sum()} median {np.median(dA):+.3f} IQR {np.percentile(dA, 25):+.3f} to {np.percentile(dA, 75):+.3f} | AUPRC improved {(dP > 0).sum()} median {np.median(dP):+.3f}")
hr = []
pb, pp = HPc["W_base"][:, 0], HPc["W_propH"][:, 0]
for s_ in np.unique(h["site"]):
    k = h["site"] == s_; y = h["y24"][k]
    r = [str(s_), int(k.sum()), int(y.sum())]
    r += [A(y, pp[k]) - A(y, pb[k]), AP(y, pp[k]) - AP(y, pb[k])] if 0 < y.sum() < k.sum() else [None, None]
    hr.append(r); print(f"[hospital Hallym {s_}] n={r[1]} deaths24={r[2]} dAUROC {r[3]} dAUPRC {r[4]}")
hosp["hallym"] = hr
out["hospital_change"] = hosp
json.dump(out, open(f"{T.ROOT}/results/v11_W.json", "w"), indent=1, default=float)
print("written results/v11_W.json")
