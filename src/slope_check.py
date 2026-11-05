"""Window test-selection slope on acuity: development (MIMIC) vs each Hallym hospital fitted freely (label-free)."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import train as T, obsprofile as PR
from regime import VARS
rng = np.random.RandomState(0)
d = T.load("mimic_dyn"); i = rng.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i], d["yhosp"][i]); dv = PR.estimate(d["x"][i], d["m"][i], acu)
rows = []
for k, (sx, sm) in enumerate(PR.target_sites("hallym_dyn:sites")):
    tp = PR.estimate(sx, sm, acu, beta=dv.beta, n=len(sx))
    r = {}
    for c in PR.LAB:
        y = tp._tested[c]
        r[c] = (LogisticRegression().fit(tp._aw[:, None], y).coef_[0, 0] if 0 < y.mean() < 1 else np.nan, y.mean())
    rows.append(r)
print(f"{'lab':8s} {'dev slope':>9s} | " + " | ".join(f"H{k} slope / tested" for k in range(len(rows))))
for c in PR.LAB:
    print(f"{VARS[c]:8s} {dv.beta_w[c]:9.2f} | " + " | ".join(f"{r[c][0]:6.2f} / {r[c][1]:.2f}" for r in rows))
