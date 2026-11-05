"""Charting-fidelity stress test: add independent noise to measured vital signs of eICU dynamic landmarks so that the
hour-to-hour change of measured vitals rises from the monitor-derived level toward the Hallym level, re-apply
carry-forward, and score frozen models (no retraining).

Usage: python noise_test.py [--n 60000] group=tag1,tag2,... [group=...]
"""
import os
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from eval_hallym import frozen_preds

VIT = list(range(6))                     # rr temp hr spo2 sbp dbp


def jitter(x, m, mu, sd):
    z = (x[..., VIT] - mu[VIT]) / sd[VIT]
    both = (m[:, 1:, VIT] > 0) & (m[:, :-1, VIT] > 0)
    return float(np.nanmean(np.where(both, np.abs(z[:, 1:] - z[:, :-1]), np.nan)))


def add_noise(x, m, sd, sigma, rs):
    x = x.copy()
    for c in VIT:
        meas = m[..., c] > 0
        x[..., c][meas] += rs.normal(0, sigma * sd[c], meas.sum()).astype(np.float32)
        for t in range(1, x.shape[1]):                     # carried cells repeat the (noisy) last measurement
            car = (~meas[:, t]) & ~np.isnan(x[:, t, c]) & ~np.isnan(x[:, t - 1, c])
            x[car, t, c] = x[car, t - 1, c]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("groups", nargs="+")
    a = ap.parse_args()
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    mu, sd = stats
    e = T.load("eicu_dyn")
    i = np.random.RandomState(0).permutation(len(e["y24"]))[:a.n]
    e = {k: (v[i] if hasattr(v, "__len__") and len(v) == len(e["y24"]) else v) for k, v in e.items()}
    rs = np.random.RandomState(1)
    levels = {}
    for sigma in (0.0, 0.3, 0.6, 0.9):
        xn = e["x"] if sigma == 0 else add_noise(e["x"], e["m"], sd, sigma, rs)
        levels[sigma] = (xn, jitter(xn, e["m"], mu, sd))
    print("noise sigma -> hour-to-hour |change| of measured vitals (SD units; eICU ~0.46, Hallym ~1.09): "
          + "  ".join(f"{s}: {j:.2f}" for s, (_, j) in levels.items()))
    for g in a.groups:
        name, tags = g.split("=")
        tags = tags.split(",")
        res = {}
        for s, (xn, _) in levels.items():
            ext = dict(e, x=xn)
            res[s] = np.mean([frozen_preds(t, ext, stats, "cuda") for t in tags], 0)
        line = []
        for j, ep in [(0, "y24"), (2, "y168"), (3, "yhosp")]:
            y = e[ep]
            base = (A(y, res[0.0][:, j]), AP(y, res[0.0][:, j]))
            line.append(f"{ep} clean {base[0]:.3f}/{base[1]:.3f} drop " + " ".join(
                f"{A(y, res[s][:, j]) - base[0]:+.3f}/{AP(y, res[s][:, j]) - base[1]:+.3f}" for s in levels if s > 0))
        print(f"\n{name}:\n  " + "\n  ".join(line))


if __name__ == "__main__":
    main()
