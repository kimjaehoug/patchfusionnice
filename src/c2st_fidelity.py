"""Label-free fidelity of imagined observation processes: classifier two-sample test (C2ST) between real target
records and development records re-observed for that target (per hospital, size-proportional). Features describe the
observation process only (no values); 'structure+acuity' adds the window vital-sign acuity so that acuity-dependent
selection can be detected. Variables the target records more often than the development hospital (which no
imagination can add) are excluded from the rate, presence and patch features. AUROC 0.5 = indistinguishable. 5-fold CV, HistGradientBoosting. Writes results/c2st_W.json."""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
import train as T
import obsprofile as PR
from regime import VARS, VIT

rng = np.random.RandomState(0)
d = T.load("mimic_dyn"); i = rng.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i], d["yhosp"][i]); dv = PR.estimate(d["x"][i], d["m"][i], acu)
pool = rng.permutation(np.setdiff1d(np.arange(len(d["x"])), i))
r_dev = PR.obs_rate(d["m"][i])


def feats(x, m, keep=None):
    o = m > 0
    keep = np.arange(o.shape[2]) if keep is None else keep
    act = np.cumsum(o.any(2), 1) > 0; na = np.maximum(act.sum(1), 1)
    rate = (o & act[..., None]).sum(1) / na[:, None]                    # hourly rate per variable
    pres = o.any(1).astype(float)                                       # recorded in window
    N, Tt, C = o.shape
    P6 = o[:, :Tt // 6 * 6].reshape(N, Tt // 6, 6, C).any(2)            # per 6-h patch
    actp = act[:, :Tt // 6 * 6].reshape(N, Tt // 6, 6).any(2); npch = np.maximum(actp.sum(1), 1)
    ppatch = (P6 & actp[..., None]).sum(1) / npch[:, None]              # share of patches with a record
    lab = o[:, :, PR.LAB]
    co = ((lab.sum(2) >= 2).sum(1) / np.maximum(lab.any(2).sum(1), 1))[:, None]   # co-measured lab hours
    gap = np.zeros((N, 1)); h = lab.any(2)
    for n in range(N):
        t = np.where(h[n])[0]; gap[n] = np.diff(t).mean() if len(t) > 1 else 48
    S = np.concatenate([rate[:, keep], pres[:, keep], ppatch[:, keep], co, gap / 48], 1)
    with np.errstate(all="ignore"):
        aw = np.nan_to_num(np.nanmean(acu(x), 1), nan=0.0)[:, None]
    return S, np.concatenate([S, aw, pres[:, PR.LAB] * aw], 1)



def c2st(Fr, Fi, seed=0):
    X = np.concatenate([Fr, Fi]); y = np.r_[np.ones(len(Fr)), np.zeros(len(Fi))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


out = {}
for tname in ("hallym_dyn:sites", "eicu_dyn:low:sites"):
    sites = PR.target_sites(tname)
    r_t = PR.obs_rate(np.concatenate([sm for _, sm in sites]))
    keep = np.where(r_t <= r_dev * 1.05)[0]
    print(f"[{tname}] reachable variables: {[VARS[c] for c in keep]}", flush=True)
    nt = sum(len(sx) for sx, _ in sites); cap = min(1.0, 8000 / nt)
    real_s, real_a = [], []
    gen = {k: ([], []) for k in ("development", "thinning", "proposed", "proposed_nokeepone")}
    ptr = 0
    for sx, sm in sites:
        k = max(1, int(round(len(sx) * cap)))
        ri = rng.permutation(len(sx))[:k]
        a, b = feats(sx[ri], sm[ri], keep); real_s.append(a); real_a.append(b)
        di = pool[ptr:ptr + k]; ptr += k
        xd, md = d["x"][di], d["m"][di]
        tp = PR.fit_window(PR.estimate(sx, sm, acu, beta=dv.beta), dv)
        views = {"development": (xd, md), "thinning": PR.imagine_rate(xd, md, r_dev, PR.obs_rate(sm), rng)}
        PR.KEEP_ONE = True; views["proposed"] = PR.imagine_two(xd, md, acu, dv, tp, parts=["sel", "cad", "vitcad", "doccad"], rng=rng)
        PR.KEEP_ONE = False; views["proposed_nokeepone"] = PR.imagine_two(xd, md, acu, dv, tp, parts=["sel", "cad", "vitcad", "doccad"], rng=rng)
        PR.KEEP_ONE = True
        for g, (xv, mv) in views.items():
            a, b = feats(xv, mv, keep); gen[g][0].append(a); gen[g][1].append(b)
    Rs, Ra = np.concatenate(real_s), np.concatenate(real_a)
    res = {"n_real": int(len(Rs)), "hospitals": len(sites), "reachable": [VARS[c] for c in keep]}
    for g, (fs, fa) in gen.items():
        Fs, Fa = np.concatenate(fs), np.concatenate(fa)
        res[g] = {"structure": [c2st(Rs, Fs, s) for s in (0, 1, 2)], "structure+acuity": [c2st(Ra, Fa, s) for s in (0, 1, 2)]}
        print(f"[{tname}] {g:20s} C2ST AUROC structure {np.mean(res[g]['structure']):.3f} (sd {np.std(res[g]['structure']):.3f})  "
              f"structure+acuity {np.mean(res[g]['structure+acuity']):.3f}", flush=True)
    out[tname] = res
json.dump(out, open(f"{T.ROOT}/results/c2st_W.json", "w"), indent=1)
print("written results/c2st_W.json")
