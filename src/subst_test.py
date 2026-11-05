"""Test-time patch substitution: patches whose observation pattern (per-variable measured / available share inside the
reset-confined patch) is atypical for the development data (Mahalanobis distance above the MIMIC q-quantile) get their
factual token replaced by the imagined full-observation token mu_k. No retraining, no target labels.

Usage: python subst_test.py [--q 0.95] tag1,tag2,...
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
import models as M
from diag_patch import load_model

N = 22


def patch_obs(X, starts, P):
    """X [n,T,88] -> [n,K,44] measured share and available share per variable inside each patch"""
    return np.stack([np.concatenate([X[:, s:s + P, 2 * N:3 * N].mean(1), X[:, s:s + P, 3 * N:4 * N].mean(1)], -1)
                     for s in starts], 1)


@torch.no_grad()
def predict(model, X, sub, bs=2048):
    out = []
    for i in range(0, len(X), bs):
        model.enc._subst = None if sub is None else torch.tensor(sub[i:i + bs], device="cuda")
        out.append(M.cr_probs(model(torch.tensor(X[i:i + bs], device="cuda")))[1].cpu().numpy())
    model.enc._subst = None
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("tags")
    a = ap.parse_args()
    tags = a.tags.split(",")
    d = T.load("mimic")
    stats = T.fit_stats(d["x"])
    T.AVAIL = True
    m0 = load_model(tags[0], "cuda")
    starts, P = m0.enc.starts, m0.enc.patch
    Xm = T.tensors(d, np.random.RandomState(0).permutation(len(d["y24"]))[:40000], stats, "cpu")[0].numpy()
    F = patch_obs(Xm, starts, P).reshape(-1, 2 * N)
    mu = F.mean(0)
    icov = np.linalg.inv(np.cov(F.T) + 1e-3 * np.eye(2 * N))
    dist = lambda f: np.sqrt(np.einsum("...i,ij,...j->...", f - mu, icov, f - mu))
    thr = np.quantile(dist(F), a.q)
    sets = {}
    e = T.load("eicu")
    site = pd.read_parquet(f"{T.ROOT}/data/eicu_cohort.parquet").sort_values("stay_id")["hospitalid"].values
    crit = [6 + 1, 13, 14, 16, 20, 21]                     # lactate ph be ne uo gcs (as in compare_regime)
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    tert = np.full(len(hr), None, object)
    tert[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    sets["eicu"] = (e, tert)
    sets["hallym"] = (T.load("hallym"), None)
    for nm, (ds, tt) in sets.items():
        X = T.tensors(ds, np.arange(len(ds["y24"])), stats, "cpu")[0].numpy()
        D = dist(patch_obs(X, starts, P))
        subs = {"none": None, "atypical": D > thr, "all": np.ones_like(D, bool)}
        print(f"\n{nm}: share of patches flagged atypical {np.mean(D > thr):.3f} (MIMIC {1 - a.q:.2f} by construction)")
        P_ = {k: np.mean([predict(load_model(t, "cuda"), X, s) for t in tags], 0) for k, s in subs.items()}
        eps = [(0, "y24"), (1, "y48"), (3, "yhosp")] if nm == "eicu" else [(1, "y48"), (2, "y168"), (3, "yhosp")]
        for k, p in P_.items():
            row = []
            for j, ep in eps:
                y = ds[ep]
                row.append(f"{ep} {A(y, p[:, j]):.3f}/{AP(y, p[:, j]):.3f} O/E {y.sum() / p[:, j].sum():.2f}")
            if tt is not None:
                lo = tt == "low"
                row.append(f"| low-testing y24 AUPRC {AP(ds['y24'][lo], p[lo, 0]):.3f}")
            print(f"  {k:9s} " + "  ".join(row))


if __name__ == "__main__":
    main()
