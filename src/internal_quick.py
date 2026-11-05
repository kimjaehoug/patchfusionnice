"""Internal (MIMIC-IV) check for quick models trained with --sub_frac: score frozen models on development patients
that were NOT in the training subset (same RandomState(7) patient draw as train.py).

Usage: python internal_quick.py [--frac 0.2] [--n 60000] group=tags ...
"""
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from eval_hallym import frozen_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("groups", nargs="+")
    a = ap.parse_args()
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    g = d["subject_id"]
    us = np.unique(g)
    keep = np.random.RandomState(7).choice(us, int(a.frac * len(us)), replace=False)
    held = np.where(~np.isin(g, keep))[0]
    held = held[np.random.RandomState(0).permutation(len(held))[:a.n]]
    e = {k: (v[held] if hasattr(v, "__len__") and len(v) == len(g) else v) for k, v in d.items()}
    print(f"held-out development samples: {len(held)}")
    for grp in a.groups:
        name, tags = grp.split("=")
        p = np.mean([frozen_preds(t, e, stats, "cuda") for t in tags.split(",")], 0)
        print(f"  {name:16s} " + "  ".join(f"{ep} {A(e[ep], p[:, j]):.3f}/{AP(e[ep], p[:, j]):.3f}"
                                           for j, ep in enumerate(["y24", "y48", "y168", "yhosp"])))


if __name__ == "__main__":
    main()
