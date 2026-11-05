"""Compare multi-horizon models on frozen eICU (paired bootstrap vs reference) and decedent timing discrimination.

Usage: python compare_lap.py ref_tag tag1 [tag2 ...]
"""
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPS = ["y24", "y48", "y168", "yhosp"]


def main(ref, tags, B=1000):
    e = np.load(f"{ROOT}/data/eicu_tensor.npz", allow_pickle=True)
    d = np.load(f"{ROOT}/data/mimic_tensor.npz", allow_pickle=True)
    P = {t: np.load(f"{ROOT}/results/preds/{t}.npz") for t in [ref] + tags}
    rng = np.random.RandomState(0)
    n = len(e["y24"])
    idx = [rng.randint(0, n, n) for _ in range(B)]
    for j, ep in enumerate(EPS):
        y = e[ep]
        print(f"\n{ep}  (eICU frozen; AUROC / AUPRC; Δ vs {ref} with 95% paired-bootstrap CI)")
        r = P[ref]["ext"][:, j]
        for t in [ref] + tags:
            p = P[t]["ext"][:, j]
            line = f"  {t:14s} {roc_auc_score(y, p):.4f} / {average_precision_score(y, p):.4f}"
            if t != ref:
                da = np.array([roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], r[i]) for i in idx])
                dp = np.array([average_precision_score(y[i], p[i]) - average_precision_score(y[i], r[i]) for i in idx])
                line += (f"   ΔAUROC {da.mean():+.4f} [{np.percentile(da, 2.5):+.4f},{np.percentile(da, 97.5):+.4f}]"
                         f"   ΔAUPRC {dp.mean():+.4f} [{np.percentile(dp, 2.5):+.4f},{np.percentile(dp, 97.5):+.4f}]")
            print(line)
    print("\nOOF (MIMIC) AUROC/AUPRC per endpoint")
    for t in [ref] + tags:
        o = P[t]["oof"]
        print(f"  {t:14s} " + "  ".join(f"{ep} {roc_auc_score(d[ep], o[:, j]):.4f}/{average_precision_score(d[ep], o[:, j]):.4f}"
                                       for j, ep in enumerate(EPS)))
    print("\nTiming among decedents (eICU): AUROC early vs later death, own-horizon output vs severity(yhosp) output")
    me = e["event"] == 1
    for t in [ref] + tags:
        x = P[t]["ext"]
        s = []
        for cut, j in [(24, 0), (48, 1)]:
            yt = e["t_event"][me] <= cut
            s.append(f"<= {cut}h: {roc_auc_score(yt, x[me, j]):.3f} (severity-only {roc_auc_score(yt, x[me, 3]):.3f})")
        print(f"  {t:14s} " + " | ".join(s))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
