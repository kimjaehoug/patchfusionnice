"""Capture-gap stress test: erase the channels that the Hallym source never records (lactate, AG, NE, FiO2, ventilation,
urine output, GCS) from eICU dynamic landmarks (values NaN, raw masks 0) and score frozen models (no retraining).

Usage: python erase_test.py [--n 60000] group=tag1,tag2,... [group=...]
"""
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from eval_hallym import frozen_preds

V = ["rr", "temp", "hr", "spo2", "sbp", "dbp", "bun", "lactate", "wbc", "hb", "na", "k", "cl", "ph", "be", "ag",
     "ne", "vaso", "fio2", "vent", "uo", "gcs"]
GONE = [V.index(v) for v in ["lactate", "ag", "ne", "fio2", "vent", "uo", "gcs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("groups", nargs="+")
    a = ap.parse_args()
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    e = T.load("eicu_dyn")
    i = np.random.RandomState(0).permutation(len(e["y24"]))[:a.n]
    e = {k: (v[i] if hasattr(v, "__len__") and len(v) == len(e["y24"]) else v) for k, v in e.items()}
    xe, me = e["x"].copy(), e["m"].copy()
    xe[..., GONE] = np.nan
    me[..., GONE] = 0
    er = dict(e, x=xe, m=me)
    for g in a.groups:
        name, tags = g.split("=")
        tags = tags.split(",")
        p0 = np.mean([frozen_preds(t, e, stats, "cuda") for t in tags], 0)
        p1 = np.mean([frozen_preds(t, er, stats, "cuda") for t in tags], 0)
        row = []
        for j, ep in [(0, "y24"), (1, "y48"), (2, "y168"), (3, "yhosp")]:
            y = e[ep]
            row.append(f"{ep} {A(y, p0[:, j]):.3f}/{AP(y, p0[:, j]):.3f} -> {A(y, p1[:, j]):.3f}/{AP(y, p1[:, j]):.3f} "
                       f"(drop {A(y, p1[:, j]) - A(y, p0[:, j]):+.3f}/{AP(y, p1[:, j]) - AP(y, p0[:, j]):+.3f}, "
                       f"O/E {y.sum() / p0[:, j].sum():.2f}->{y.sum() / p1[:, j].sum():.2f})")
        print(f"\n{name}:\n  " + "\n  ".join(row))


if __name__ == "__main__":
    main()
