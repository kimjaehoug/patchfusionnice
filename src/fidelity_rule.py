"""Label-free, per-hospital choice between three-scale and rate-level re-observation by fidelity (C2ST).
Plan: paper/notes/fidelity_rule_prereg.md. Writes results/fidelity_rule_W.json."""
import json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP, roc_auc_score
import train as T
import obsprofile as PR
from eval_hallym import boot_idx, frozen_preds
from bootstrap_par import boot_metric
rng = np.random.RandomState(0)
d = T.load("mimic_dyn"); i0 = rng.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i0], d["yhosp"][i0]); dv = PR.estimate(d["x"][i0], d["m"][i0], acu)
pool = rng.permutation(np.setdiff1d(np.arange(len(d["x"])), i0)); r_dev = PR.obs_rate(d["m"][i0])
src = open(f"{T.ROOT}/src/c2st_fidelity.py").read()
exec(src[src.index("def feats"):src.index("out = {}")])          # feats() and c2st() exactly as Supplementary S15
S5 = (2, 3, 4, 5, 6); st = T.fit_stats(d["x"])


def tertile(e, name):
    site = np.asarray(e["site"]).astype(str); crit = [7, 13, 14, 16, 20, 21]
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    t = np.full(len(hr), None, object); t[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    return t == name


e = T.load("eicu_dyn"); h = T.load("hallym_dyn")
SRC = {"low": (e, tertile(e, "low"), "W_propTE", "W_thinE"), "mid": (e, tertile(e, "mid"), "W_propTM", "W_thinM"),
       "hallym": (h, np.ones(len(h["y24"]), bool), "W_propTH", "W_thinH")}
out = {}; ptr = 0
for name, (ds, mask, t3, tr) in SRC.items():
    site = np.asarray(ds["site"]).astype(str)
    sites = [s for s in np.unique(site[mask]) if (mask & (site == s)).sum() >= 500]
    adv = {}
    for s in sites:
        ii = np.where(mask & (site == s))[0]; sx, sm = ds["x"][ii], ds["m"][ii]
        k = min(1500, len(ii)); ri = rng.permutation(len(ii))[:k]
        r_t = PR.obs_rate(sm); keep = np.where(r_t <= r_dev * 1.05)[0]
        di = pool[ptr:ptr + k]; ptr = (ptr + k) % (len(pool) - 2000)
        xd, md = d["x"][di], d["m"][di]
        tp = PR.fit_window(PR.estimate(sx, sm, acu, beta=dv.beta), dv)
        PR.KEEP_ONE = True
        views = {"rate": PR.imagine_rate(xd, md, r_dev, r_t, rng), "three": PR.imagine_two(xd, md, acu, dv, tp, parts=["sel", "cad", "vitcad", "doccad"], rng=rng)}
        real = feats(sx[ri], sm[ri], keep)[1]
        c = {g: c2st(real, feats(xv, mv, keep)[1], 0) for g, (xv, mv) in views.items()}
        adv[s] = c["rate"] - c["three"]
    print(f"[{name}] {len(sites)} hospitals; three-scale closer at {sum(v > 0 for v in adv.values())}; advantage median {np.median(list(adv.values())):+.4f}", flush=True)
    m2 = mask & np.isin(site, sites)
    if name != "hallym":
        ld = lambda tag: np.mean([np.load(f"{T.ROOT}/results/preds/{tag}_s{s}.npz")["ext"] for s in S5], 0)
    else:
        ld = lambda tag: np.mean([frozen_preds(f"{tag}_s{s}", h, st, "cuda") for s in S5], 0)
    P3, PRt = ld(t3)[m2], ld(tr)[m2]
    use3 = np.array([adv[s] > 0 for s in site[m2]])
    P = {"three": P3, "rate": PRt, "guided": np.where(use3[:, None], P3, PRt)}
    cl = site[m2] if name != "hallym" else ds["subject_id"][m2]
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    R = {"hospitals": len(sites), "three_chosen": int(sum(v > 0 for v in adv.values())), "advantage": adv}
    for j, ep in ((0, "y24"), (1, "y48")):
        y = ds[ep][m2]; jj = [q for q in idx if 0 < y[q].sum() < len(q)]
        r = {g: [A(y, p[:, j]), AP(y, p[:, j])] for g, p in P.items()}
        for a_, b_ in (("guided", "three"), ("guided", "rate"), ("three", "rate")):
            for mn, f in (("auroc", A), ("auprc", AP)):
                bs = boot_metric(f, y, P[a_][:, j], jj) - boot_metric(f, y, P[b_][:, j], jj)
                r[f"{a_}-{b_}_{mn}"] = [f(y, P[a_][:, j]) - f(y, P[b_][:, j]), *np.percentile(bs, [2.5, 97.5]).tolist()]
        R[ep] = r
        cc = lambda k: f"{r[k][0]:+.4f} [{r[k][1]:+.4f},{r[k][2]:+.4f}]"
        print(f"  {ep}: three {r['three'][0]:.4f}/{r['three'][1]:.4f} rate {r['rate'][0]:.4f}/{r['rate'][1]:.4f} guided {r['guided'][0]:.4f}/{r['guided'][1]:.4f} | "
              f"guided-three AUROC {cc('guided-three_auroc')} | guided-rate AUROC {cc('guided-rate_auroc')} AUPRC {cc('guided-rate_auprc')} | three-rate AUROC {cc('three-rate_auroc')}", flush=True)
    if name != "hallym":
        rows = []
        for s in sites:
            q = site[m2] == s; y = ds["y24"][m2][q]
            if y.sum() >= 10:
                rows.append((adv[s], A(y, P3[q, 0]) - A(y, PRt[q, 0])))
        if len(rows) > 5:
            rho = pd.Series([a for a, _ in rows]).corr(pd.Series([b for _, b in rows]), method="spearman")
            R["spearman_adv_vs_auroc_diff"] = [float(rho), len(rows)]
            print(f"  Spearman(advantage, per-hospital 24h AUROC three-rate) {rho:+.3f} over {len(rows)} hospitals", flush=True)
    else:
        print("  Hallym advantage per hospital", {s: round(v, 4) for s, v in adv.items()}, flush=True)
    out[name] = R
json.dump(out, open(f"{T.ROOT}/results/fidelity_rule_W.json", "w"), indent=1, default=float)
print("written results/fidelity_rule_W.json")
