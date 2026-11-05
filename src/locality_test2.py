"""Locality with full re-preprocessing. Raw laboratory observations in one 3-h block are deleted, and carry-forward,
first differences, raw-observation masks and availability indicators are recomputed. Measured: (1) the last hour whose
input changed (the carry-forward horizon); (2) where the recurrent-path tokens (before attention) change: in the block's
own reset patch, in the patches whose hours had changed input, and elsewhere. Reset vs no-reset encoders.
Usage: python locality_test2.py reset_tag noreset_tag"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
import obsprofile as PR
from diag_patch import load_model


@torch.no_grad()
def tokens(m, X, bs=1024):
    """recurrent-path tokens before global attention: tok(dyn(proj(x)))."""
    m.eval(); enc = m.enc; out = []
    for k in range(0, len(X), bs):
        x = X[k:k + bs]
        out.append(enc.tok(enc.dyn(enc.proj(x), x)).cpu())
    return torch.cat(out).numpy()

LAB = list(range(6, 16))


def delete_block(x, m, a, b):
    """delete raw laboratory observations in hours a..b and redo carry-forward exactly as in imagination."""
    raw = np.where(m > 0, x, np.nan).astype(np.float32)
    keep = np.ones_like(raw, bool)
    keep[:, a:b + 1, LAB] = False
    raw = np.where(keep, raw, np.nan)
    xs, ms = PR.refill(raw)
    first = np.cumsum(m > 0, 1) == 0
    carried_in = (m == 0) & ~np.isnan(x) & first                       # values carried in from before the window
    fill = carried_in & keep.any(1, keepdims=True) & np.isnan(xs)
    xs = np.where(fill, x, xs)
    return xs.astype(np.float32), ms.astype(np.float32)


def main():
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    e = T.load("eicu_dyn")
    i = np.where(e["landmark"] >= 48)[0]
    i = i[np.random.RandomState(0).permutation(len(i))[:4000]]
    ds = {k: (v[i] if hasattr(v, "__len__") and len(v) == len(e["y24"]) else v) for k, v in e.items()}
    T.AVAIL = True
    X0 = T.tensors(ds, np.arange(len(i)), stats, "cuda")[0]
    out = {}
    for tag in sys.argv[1:]:
        m = load_model(tag, "cuda")
        st, en, seg = m.enc.dyn.starts, m.enc.dyn.ends, m.enc.dyn.seg
        z0 = tokens(m, X0)
        res = []
        for j in [3, 7, 11]:
            a, b = st[j], en[j]
            xs, ms = delete_block(ds["x"], ds["m"], a, b)
            X1 = T.tensors(ds, np.arange(len(i)), stats, "cuda", xm=(xs, ms))[0]
            changed = (X1 - X0).abs().amax(dim=2).cpu().numpy() > 1e-6        # [N, T] hours whose input changed
            hrs = np.where(changed.any(0))[0]
            last = int(hrs.max()) if len(hrs) else b
            reach = np.array([changed[:, h].mean() for h in range(changed.shape[1])])
            z1 = tokens(m, X1)
            dzs = np.linalg.norm(z1 - z0, axis=-1)                           # [N, K] per sample and token
            dz = dzs.mean(0)
            P6 = lambda k: ((st[k] // 6) * 6, (st[k] // 6) * 6 + 5)           # common 6-h grid for both encoders
            touched = np.stack([changed[:, P6(k)[0]:P6(k)[1] + 1].any(1) for k in range(len(st))], 1)   # [N, K] per sample
            before = np.array([P6(k)[1] < a for k in range(len(st))])       # tokens of patches entirely before the block
            own = np.array([P6(k) == P6(j) for k in range(len(st))])
            tot = dzs.sum()
            r = {"block": [int(a), int(b)], "last_changed_hour": last,
                 "share_hours_changed_after_block": {int(h): float(reach[h]) for h in range(b + 1, min(b + 14, 48)) if reach[h] > 0},
                 "own_patch": float(dzs[:, own].sum() / tot), "input_changed_patches": float((dzs * touched).sum() / tot),
                 "outside_changed_input": float((dzs * ~touched).sum() / tot), "before_block": float(dzs[:, before].sum() / tot),
                 "per_token": np.round(dz, 4).tolist()}
            print(f"{tag} (reset every {seg} h) block h{a}-{b}: input changed up to hour {last}; token change: own 6-h patch {r['own_patch']:.2f}, "
                  f"patches whose input changed (per sample) {r['input_changed_patches']:.3f}, patches with unchanged input {r['outside_changed_input']:.3f} "
                  f"(of which before the block {r['before_block']:.3f})", flush=True)
            res.append(r)
        out[tag] = res
    json.dump(out, open(f"{T.ROOT}/results/locality2_W.json", "w"), indent=1)
    print("written results/locality2_W.json")


if __name__ == "__main__":
    main()
