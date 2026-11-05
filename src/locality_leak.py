"""Is the observation-indicator leakage diagnosis temporally localisable? For patch-restoration models with and without
reset: (a) erase laboratory observations in one 3-h block and measure where the injected testing signal
(probe(restored token) - probe(factual token)) changes; (b) how well the injected signal of block k identifies blocks
without any laboratory test (AUROC), and how well the factual-token probe recovers the block's own testing share (R2).
The probe (ridge) is fit per model on factual tokens of held-out MIMIC-IV development patients.

Usage: python locality_leak.py name=tag1,tag2,... [name=...]
"""
import os
import sys
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score as A, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from diag_patch import load_model

N, LAB = 22, list(range(6, 16))


@torch.no_grad()
def tok_mu(m, X, bs=1024):
    m.enc._keep_diag = True
    Z, MU = [], []
    for i in range(0, len(X), bs):
        m(X[i:i + bs])
        Z.append(m.enc._diag[0].cpu())
        MU.append(m.enc._diag[1].cpu())
    m.enc._keep_diag = False
    return torch.cat(Z).numpy(), torch.cat(MU).numpy()


def share(X, st, en):
    mk = X[..., 2 * N:3 * N][..., LAB]
    return np.stack([mk[:, a:b + 1].float().mean((1, 2)).cpu().numpy() for a, b in zip(st, en)], 1)


def sub(ds, idx, n):
    return {k: (v[idx] if hasattr(v, "__len__") and len(v) == n else v) for k, v in ds.items()}


def main():
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    us = np.unique(d["subject_id"])
    keep = np.random.RandomState(7).choice(us, int(0.2 * len(us)), replace=False)
    held = np.where(~np.isin(d["subject_id"], keep))[0]
    held = held[np.random.RandomState(0).permutation(len(held))[:20000]]
    dm = sub(d, held, len(d["y24"]))
    e = T.load("eicu_dyn")
    ne = len(e["y24"])
    full = np.where(e["landmark"] >= 48)[0]
    full = full[np.random.RandomState(0).permutation(len(full))[:4000]]
    anyl = np.random.RandomState(1).permutation(ne)[:20000]
    ef, ea = sub(e, full, ne), sub(e, anyl, ne)
    T.AVAIL = True
    Xm = T.tensors(dm, np.arange(len(held)), stats, "cuda")[0]
    Xf = T.tensors(ef, np.arange(len(full)), stats, "cuda")[0]
    Xa = T.tensors(ea, np.arange(len(anyl)), stats, "cuda")[0]
    for g in sys.argv[1:]:
        name, tags = g.split("=")
        res = {"own": [], "out": [], "auc_inj": [], "r2_fact": [], "auc_fact": []}
        for tag in tags.split(","):
            m = load_model(tag, "cuda")
            st, en, seg = m.enc.dyn.starts, m.enc.dyn.ends, 6        # attribution judged against 6-h patches
            Zm, _ = tok_mu(m, Xm)
            sm = share(Xm, st, en)
            K, D = Zm.shape[1], Zm.shape[2]
            probe = Ridge(alpha=1.0).fit(Zm.reshape(-1, D), sm.reshape(-1))
            # (b) accuracy on eICU, all landmarks
            Za, Ma = tok_mu(m, Xa)
            sa = share(Xa, st, en)
            pz = probe.predict(Za.reshape(-1, D)).reshape(-1, K)
            inj = probe.predict(Ma.reshape(-1, D)).reshape(-1, K) - pz
            res["r2_fact"].append(r2_score(sa.reshape(-1), pz.reshape(-1)))
            res["auc_fact"].append(A((sa > 0).reshape(-1), pz.reshape(-1)))
            res["auc_inj"].append(A((sa == 0).reshape(-1), inj.reshape(-1)))
            # (a) locality of the injected signal
            Z0, M0 = tok_mu(m, Xf)
            i0 = probe.predict(M0.reshape(-1, D)).reshape(-1, K) - probe.predict(Z0.reshape(-1, D)).reshape(-1, K)
            for j in (3, 7, 11):
                a, b = st[j], en[j]
                Xp = Xf.clone()
                for blk in range(4):
                    Xp[:, a:b + 1, [blk * N + c for c in LAB]] = 0.0
                Z1, M1 = tok_mu(m, Xp)
                i1 = probe.predict(M1.reshape(-1, D)).reshape(-1, K) - probe.predict(Z1.reshape(-1, D)).reshape(-1, K)
                dz = np.abs(i1 - i0).mean(0)
                lo = (a // seg) * seg
                inside = np.array([(st[k] >= lo) and (en[k] <= lo + seg - 1) for k in range(K)])
                res["own"].append(dz[j] / dz.sum())
                res["out"].append(dz[~inside].sum() / dz.sum())
            del m
            torch.cuda.empty_cache()
        print(f"{name:10s} injected-signal change: own block {np.mean(res['own']):.2f}, outside 6-h patch {np.mean(res['out']):.2f}"
              f" | AUROC(injection -> block without lab test) {np.mean(res['auc_inj']):.3f}"
              f" | factual probe: R2 {np.mean(res['r2_fact']):.3f}, AUROC(test present) {np.mean(res['auc_fact']):.3f}"
              f"  [per seed AUROC inj {np.round(res['auc_inj'], 3)}]")


if __name__ == "__main__":
    main()
