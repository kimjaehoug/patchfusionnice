"""Where does the larger model lose 7-day discrimination at the Hallym hospitals? W + imagination vs 2.53M + imagination
(3-seed ensembles): 7-day AUROC of deaths by timing vs 7-day survivors, by landmark and by hospital; calibration."""
import numpy as np, train as T
from eval_hallym import frozen_preds, boot_idx
from sklearn.metrics import roc_auc_score as A
S = (2, 3, 4); d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn")
print("hallym keys", list(h.keys()))
G = {"small": "F_propH", "W": "W_propH", "W4L": "WL4_propH"}
P = {g: np.mean([frozen_preds(f"{p}_s{s}", h, st, "cuda") for s in S], 0)[:, 2] for g, p in G.items()}
y = h["y168"].astype(bool); te = h["t_event"]; ev = h["event"]; L = h["landmark"]
site = np.asarray(h["site"]).astype(str) if "site" in h else None
idx = list(boot_idx(h["subject_id"], 500, np.random.RandomState(0)))
def cmp(mask_case, mask_ctrl, label):
    m = mask_case | mask_ctrl; yy = mask_case[m]
    if yy.sum() < 5: print(f"  {label}: too few"); return
    a = {g: A(yy, P[g][m]) for g in G}
    bs = []
    for i in idx:
        i = i[m[i]]; yi = mask_case[i]
        if 0 < yi.sum() < len(i): bs.append(A(yi, P["W"][i]) - A(yi, P["small"][i]))
    print(f"  {label:38s} cases {int(yy.sum()):4d} | small {a['small']:.3f}  W {a['W']:.3f}  W4L {a['W4L']:.3f} | W-small {a['W'] - a['small']:+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
surv = ~y
print("7-day deaths by time from landmark (vs 7-day survivors):")
dt = te                      # t_event is already the time from the landmark
for lo, hi in ((0, 24), (24, 48), (48, 96), (96, 168)):
    cmp(y & (dt > lo) & (dt <= hi), surv, f"death in ({lo},{hi}] h")
print("by landmark:")
for l in sorted(set(L)): cmp(y & (L == l), surv & (L == l), f"landmark {int(l)}")
if site is not None:
    print("by hospital:")
    for s_ in np.unique(site): cmp(y & (site == s_), surv & (site == s_), f"hospital {s_}")
print("all", end=""); cmp(y, surv, "")
print("O/E 7-day:", {g: round(y.sum() / P[g].sum(), 2) for g in G}, " mean risk:", {g: round(P[g].mean(), 3) for g in G})
