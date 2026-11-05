"""Why does patch-level restoration hurt? (D1) drop restored patch tokens at inference; (D2) linear probe of the
laboratory-measured share of each 3-h block, trained on factual MIMIC tokens, applied to factual and restored tokens
at each site.

Usage: python diag_pimag.py tag1,tag2,tag3
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
import models as M
from diag_patch import load_model

N, LAB = 22, list(range(6, 16))


@torch.no_grad()
def run(model, X, drop=False, diag=False, bs=2048):
    enc = model.enc
    enc._drop_pimag, enc._keep_diag = drop, diag
    P, Z, MU = [], [], []
    for i in range(0, len(X), bs):
        o = model(X[i:i + bs])
        P.append(M.cr_probs(o)[1].cpu().numpy())
        if diag:
            Z.append(enc._diag[0].cpu().numpy())
            MU.append(enc._diag[1].cpu().numpy())
    enc._drop_pimag, enc._keep_diag = False, False
    return np.concatenate(P), (np.concatenate(Z) if diag else None), (np.concatenate(MU) if diag else None)


def block_lab_share(xin, starts, ends):
    m = xin[..., 2 * N:3 * N][..., LAB]                       # raw-observation masks of labs
    return np.stack([m[:, s:e + 1].mean((1, 2)) for s, e in zip(starts, ends)], 1)


def main():
    tags = sys.argv[1].split(",")
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    T.AVAIL = True
    us = np.unique(d["subject_id"])
    keep = np.random.RandomState(7).choice(us, int(0.2 * len(us)), replace=False)
    held = np.where(~np.isin(d["subject_id"], keep))[0]
    held = held[np.random.RandomState(0).permutation(len(held))[:30000]]
    sets = {"MIMIC (held-out)": {k: (v[held] if hasattr(v, "__len__") and len(v) == len(d["y24"]) else v) for k, v in d.items()}}
    e = T.load("eicu_dyn")
    ie = np.random.RandomState(0).permutation(len(e["y24"]))[:60000]
    sets["eICU"] = {k: (v[ie] if hasattr(v, "__len__") and len(v) == len(e["y24"]) else v) for k, v in e.items()}
    sets["Hallym"] = T.load("hallym_dyn")
    for tag in tags:
        m = load_model(tag, "cuda")
        starts, ends = m.enc.dyn.starts, m.enc.dyn.ends
        print(f"\n=== {tag}")
        probe = None
        for nm, ds in sets.items():
            X = T.tensors(ds, np.arange(len(ds["y24"])), stats, "cuda")[0]
            p_full, Z, MU = run(m, X, diag=True)
            p_drop, _, _ = run(m, X, drop=True)
            share = block_lab_share(X.cpu().numpy(), starts, ends)
            K, dd = Z.shape[1], Z.shape[2]
            if probe is None:                                  # fit on MIMIC factual tokens
                probe = Ridge(alpha=1.0).fit(Z.reshape(-1, dd), share.reshape(-1))
            pz = probe.predict(Z.reshape(-1, dd)).reshape(-1, K)
            pm = probe.predict(MU.reshape(-1, dd)).reshape(-1, K)
            sparse = share < np.quantile(share, 0.25) if share.std() > 0 else np.ones_like(share, bool)
            row = []
            for j, ep in [(0, "y24"), (3, "yhosp")]:
                y = ds[ep]
                row.append(f"{ep} full {A(y, p_full[:, j]):.3f}/{AP(y, p_full[:, j]):.3f} O/E {y.sum() / p_full[:, j].sum():.2f}"
                           f" | drop {A(y, p_drop[:, j]):.3f}/{AP(y, p_drop[:, j]):.3f} O/E {y.sum() / p_drop[:, j].sum():.2f}")
            print(f"  {nm:17s} D1 " + "  ||  ".join(row))
            print(f"  {'':17s} D2 lab-measured share: actual {share.mean():.3f} | read from factual token {pz.mean():.3f}"
                  f" | read from restored token {pm.mean():.3f}   (blocks with no lab test: actual {share[share == 0].mean() if (share == 0).any() else float('nan'):.3f},"
                  f" factual {pz[share == 0].mean():.3f}, restored {pm[share == 0].mean():.3f}; share of such blocks {(share == 0).mean():.2f})")
            del X
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
