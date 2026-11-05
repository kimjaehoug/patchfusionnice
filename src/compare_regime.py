"""eICU evaluation by hospital testing-intensity tertile (discrimination + calibration-in-the-large).

Usage: python compare_regime.py group=tag1,tag2,... [group=...]   (tags in a group are averaged as an ensemble)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

ROOT = os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = ["rr", "temp", "hr", "spo2", "sbp", "dbp", "bun", "lactate", "wbc", "hb", "na", "k", "cl", "ph", "be", "ag",
     "ne", "vaso", "fio2", "vent", "uo", "gcs"]
EPS = ["y24", "y48", "y168", "yhosp"]


def main(groups):
    DATA = os.environ.get("PF_DATA", "")                       # "" = hour-12 landmark, "_dyn" = dynamic landmarks
    e = np.load(f"{ROOT}/data/eicu{DATA}_tensor.npz", allow_pickle=True)
    site = e["site"] if "site" in e.files else \
        pd.read_parquet(f"{ROOT}/data/eicu_cohort.parquet").sort_values("stay_id")["hospitalid"].values
    crit = [V.index(v) for v in ["lactate", "ph", "be", "ne", "uo", "gcs"]]
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    tert = np.full(len(hr), None, object)
    tert[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    runs = {}
    for l in open(f"{ROOT}/results/runs.jsonl"):
        r = json.loads(l)
        runs[r["tag"]] = r
    for g in groups:
        name, tags = g.split("=")
        tags = [t for t in tags.split(",") if os.path.exists(f"{ROOT}/results/preds/{t}.npz")]
        if not tags:
            continue
        P = np.mean([np.load(f"{ROOT}/results/preds/{t}.npz")["ext"] for t in tags], 0)
        rows = [[runs[t][f"oof_{ep}_auprc"] for ep in EPS] for t in tags if t in runs]
        oof = " ".join(f"{ep} {v:.3f}" for ep, v in zip(EPS, np.mean(rows, 0))) if rows else "n/a"
        print(f"\n{name} (n={len(tags)}: {','.join(tags)})  OOF AUPRC " + oof)
        for j, ep in [(0, "y24"), (1, "y48"), (3, "yhosp")]:
            y = e[ep]
            s = [f"all {A(y, P[:, j]):.3f}/{AP(y, P[:, j]):.3f} O/E {P[:, j].mean() / y.mean():.2f}"]
            for t in ["low", "mid", "high"]:
                i = tert == t
                s.append(f"{t} {AP(y[i], P[i, j]):.3f} O/E {P[i, j].mean() / y[i].mean():.2f}")
            print(f"  {ep:5s} " + " | ".join(s))


if __name__ == "__main__":
    main(sys.argv[1:])
