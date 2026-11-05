"""Does the reset make observation effects attributable to their time block? Erase laboratory observations
(value, delta, mask, availability) in one 3-h block of the 48-h window and measure where the recurrent-path tokens
(before attention) change, for a reset and a no-reset model trained with the same pipeline.

Usage: python locality_test.py reset_tag noreset_tag
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from diag_patch import load_model

N, LAB = 22, list(range(6, 16))


@torch.no_grad()
def tokens(m, X, bs=1024):
    m.enc._keep_tok = True
    out = []
    for i in range(0, len(X), bs):
        m(X[i:i + bs])
        out.append(m.enc._tok.cpu())
    m.enc._keep_tok = False
    return torch.cat(out).numpy()


def main():
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    e = T.load("eicu_dyn")
    i = np.where(e["landmark"] >= 48)[0]                                  # full 48-h window available
    i = i[np.random.RandomState(0).permutation(len(i))[:4000]]
    ds = {k: (v[i] if hasattr(v, "__len__") and len(v) == len(e["y24"]) else v) for k, v in e.items()}
    for tag in sys.argv[1:]:
        m = load_model(tag, "cuda")
        T.AVAIL = True
        X = T.tensors(ds, np.arange(len(i)), stats, "cuda")[0]
        st, en, seg = m.enc.dyn.starts, m.enc.dyn.ends, m.enc.dyn.seg
        z0 = tokens(m, X)
        rows = []
        for j in [3, 7, 11]:                                              # perturbed 3-h block (hours 9-11, 21-23, 33-35)
            a, b = st[j], en[j]
            Xp = X.clone()
            for blk in range(4):
                Xp[:, a:b + 1, [blk * N + c for c in LAB]] = 0.0
            z1 = tokens(m, Xp)
            dz = np.linalg.norm(z1 - z0, axis=-1).mean(0)                 # per token
            seg_lo, seg_hi = (a // seg) * seg, (a // seg) * seg + seg - 1 if seg < 48 else 47
            inside = np.array([(st[k] >= seg_lo) and (en[k] <= seg_hi) for k in range(len(st))]) if seg < 48 else \
                np.array([k == j for k in range(len(st))])
            own = np.array([k == j for k in range(len(st))])
            rows.append((j, dz[own].sum() / dz.sum(), dz[~inside].sum() / dz.sum(), np.round(dz, 3)))
        print(f"\n{tag} (reset every {seg} h)")
        for j, own, out, dz in rows:
            print(f"  erase labs in block {j} (h {st[j]}-{en[j]}): share of token change in that block's token {own:.2f}; "
                  f"outside its 6-h patch {out:.2f}\n    per-token |dz|: {dz}")


if __name__ == "__main__":
    main()
