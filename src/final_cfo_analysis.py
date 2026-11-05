"""Final comparison for the counterfactual-observation study (hour-12 data, 3-seed ensembles, frozen eICU)."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = ["rr", "temp", "hr", "spo2", "sbp", "dbp", "bun", "lactate", "wbc", "hb", "na", "k", "cl", "ph", "be", "ag",
     "ne", "vaso", "fio2", "vent", "uo", "gcs"]
S = lambda tag: [f"{tag}_s{s}" for s in (2, 3, 4)]
GROUPS = {"base(avail)": S("hpf_noq_av"), "TTA outside": [f"hpf_noq_av_s{s}_tta" for s in (2, 3, 4)],
          "whole-seq aug (reset)": S("gaug"), "whole-seq aug (carry)": S("gaugcarry"),
          "patch-wise in-model aug": S("cfoaug"), "patch-wise + CF tokens (CFO)": S("cfo"),
          "hybrid granularity": S("hyb")}
CONTRASTS = [("patch-wise in-model aug", "base(avail)"), ("patch-wise + CF tokens (CFO)", "base(avail)"),
             ("hybrid granularity", "base(avail)"), ("patch-wise in-model aug", "whole-seq aug (reset)"),
             ("hybrid granularity", "whole-seq aug (reset)"), ("patch-wise + CF tokens (CFO)", "patch-wise in-model aug"),
             ("whole-seq aug (reset)", "whole-seq aug (carry)"), ("patch-wise in-model aug", "TTA outside")]


def main():
    e = np.load(f"{ROOT}/data/eicu_tensor.npz", allow_pickle=True)
    site = pd.read_parquet(f"{ROOT}/data/eicu_cohort.parquet").sort_values("stay_id")["hospitalid"].values
    crit = [V.index(v) for v in ["lactate", "ph", "be", "ne", "uo", "gcs"]]
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    tert = np.full(len(hr), None, object)
    tert[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    P = {}
    for g, tags in GROUPS.items():
        tags = [t for t in tags if os.path.exists(f"{ROOT}/results/preds/{t}.npz")]
        if tags:
            P[g] = 1 / (1 + np.exp(-np.mean([lg(np.load(f"{ROOT}/results/preds/{t}.npz")["ext"]) for t in tags], 0)))
            P[g + "#n"] = len(tags)
    print("== 3-seed ensembles, eICU (AUROC/AUPRC, O/E; tertiles = hospital testing intensity)")
    for g in GROUPS:
        if g not in P:
            continue
        out = []
        for j, ep in [(0, "y24"), (1, "y48"), (3, "yhosp")]:
            y, p = e[ep], P[g][:, j]
            low = tert == "low"
            out.append(f"{ep} {A(y, p):.3f}/{AP(y, p):.3f} O/E {p.mean() / y.mean():.2f} low {AP(y[low], p[low]):.3f} (O/E {p[low].mean() / y[low].mean():.2f})")
        print(f"  {g:30s} n={P[g + '#n']} | " + " | ".join(out))
    hs = np.unique(site)
    idx = {h: np.where(site == h)[0] for h in hs}
    rng = np.random.RandomState(0)
    draws = [np.concatenate([idx[h] for h in rng.choice(hs, len(hs), replace=True)]) for _ in range(300)]
    print("\n== hospital-cluster bootstrap (300), Δ = first − second")
    for a, b in CONTRASTS:
        if a not in P or b not in P:
            continue
        out = []
        for j, ep in [(0, "y24"), (1, "y48"), (3, "yhosp")]:
            y = e[ep]
            dA = np.array([A(y[i], P[a][i, j]) - A(y[i], P[b][i, j]) for i in draws])
            dP = np.array([AP(y[i], P[a][i, j]) - AP(y[i], P[b][i, j]) for i in draws])
            out.append(f"{ep} ΔAUROC {dA.mean():+.4f} [{np.percentile(dA, 2.5):+.4f},{np.percentile(dA, 97.5):+.4f}] "
                       f"ΔAUPRC {dP.mean():+.4f} [{np.percentile(dP, 2.5):+.4f},{np.percentile(dP, 97.5):+.4f}]")
        print(f"  {a} vs {b}:\n     " + "\n     ".join(out))


if __name__ == "__main__":
    main()
