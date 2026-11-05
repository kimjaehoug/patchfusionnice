"""Reset x patch-scale imagination: does the patch-scale component (full vs selection-only imagination) help more with
the reset encoder than without it? Hallym (patient-cluster bootstrap) and eICU-CRD low-testing tertile (hospital-cluster
bootstrap), seed ensembles; per-seed differences are printed as well.

Usage: python link_eval.py --pfx X --seeds 2 3 4 5 [--targets H E] [--write]
  tags: {pfx}{R|N}{full|sel}{H|E}_s{seed}   (R reset, N no reset)
"""
from bootstrap_par import boot_diff
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from eval_hallym import frozen_preds, boot_idx

EPS = ["y24", "y48", "y168", "yhosp"]


def low_tertile(e):
    site = e["site"]
    crit = [7, 13, 14, 16, 20, 21]
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    tert = np.full(len(hr), None, object)
    tert[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    return tert == "low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfx", default="X")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--targets", nargs="+", default=["H", "E"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="link_summary.json")
    a = ap.parse_args()
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    out = {}
    for tg in a.targets:
        if tg == "H":
            h = T.load("hallym_dyn")
            ys, cl, get = h, h["subject_id"], (lambda t: frozen_preds(t, h, stats, "cuda"))
            msk = np.ones(len(h["y24"]), bool)
        else:
            e = T.load("eicu_dyn")
            msk = low_tertile(e)
            ys, cl = e, e["site"][msk]
            get = lambda t: np.load(f"{T.ROOT}/results/preds/{t}.npz")["ext"]
        P = {}
        for r in "RN":
            for p in ("full", "sel"):
                tags = [f"{a.pfx}{r}{p}{tg}_s{s}" for s in a.seeds if os.path.exists(f"{T.ROOT}/results/preds/{a.pfx}{r}{p}{tg}_s{s}.npz")]
                P[r + p] = [get(t)[msk] for t in tags]
        n = min(len(v) for v in P.values())
        if n == 0:
            print(tg, "no complete runs")
            continue
        res = {"n_seeds": n}
        for j, ep in enumerate(EPS):
            y = ys[ep][msk]
            per = {k: [(A(y, v[i][:, j]), AP(y, v[i][:, j])) for i in range(n)] for k, v in P.items()}
            ens = {k: np.mean([v[i][:, j] for i in range(n)], 0) for k, v in P.items()}
            dR = [per["Rfull"][i][0] - per["Rsel"][i][0] for i in range(n)]
            dN = [per["Nfull"][i][0] - per["Nsel"][i][0] for i in range(n)]
            r = {k: [A(y, v), AP(y, v), y.sum() / v.sum()] for k, v in ens.items()}
            r["seed_dAUROC_R"], r["seed_dAUROC_N"] = dR, dN
            if ep in ("y24", "yhosp"):
                idx = [i for i in boot_idx(cl, a.B, np.random.RandomState(0)) if 0 < y[i].sum() < len(i)]
                for k, f in (("auroc", A), ("auprc", AP)):
                    bR = boot_diff(f, y, ens["Rfull"], ens["Rsel"], idx)
                    bN = boot_diff(f, y, ens["Nfull"], ens["Nsel"], idx)
                    pR = f(y, ens["Rfull"]) - f(y, ens["Rsel"])
                    pN = f(y, ens["Nfull"]) - f(y, ens["Nsel"])
                    r[f"d_{k}_R"] = [pR, *np.percentile(bR, [2.5, 97.5])]
                    r[f"d_{k}_N"] = [pN, *np.percentile(bN, [2.5, 97.5])]
                    r[f"inter_{k}"] = [pR - pN, *np.percentile(bR - bN, [2.5, 97.5]), float((bR - bN > 0).mean())]
            res[ep] = r
            line = f"[{tg} {ep} n={n}] AUROC sel/full  R {r['Rsel'][0]:.3f}->{r['Rfull'][0]:.3f}  N {r['Nsel'][0]:.3f}->{r['Nfull'][0]:.3f}" \
                   f" | per-seed dR {np.round(dR, 3).tolist()} dN {np.round(dN, 3).tolist()}"
            if "inter_auroc" in r:
                line += f" | inter AUROC {r['inter_auroc'][0]:+.3f} [{r['inter_auroc'][1]:+.3f},{r['inter_auroc'][2]:+.3f}] P>0={r['inter_auroc'][3]:.2f}" \
                        f" AUPRC {r['inter_auprc'][0]:+.3f} [{r['inter_auprc'][1]:+.3f},{r['inter_auprc'][2]:+.3f}]"
            print(line, flush=True)
        out[tg] = res
    if a.write:
        json.dump(out, open(f"{T.ROOT}/results/{a.out}", "w"), indent=1, default=float)
        print("written results/" + a.out)


if __name__ == "__main__":
    main()
