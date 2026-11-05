"""Forensics of the reset's source-only loss at the Hallym hospitals (24-h AUROC, hidden 256): where it sits and which
input drives it. Frozen models, no retraining: reset W_base vs no reset W_noreset (seeds 2-4), and the 30%-trained pair
WXRbase / WXNbase (seeds 2-5) as a reproduction check. Diagnostic use of Hallym outcomes only; any fix must be decided
without them. Writes results/reset_gap_W.json."""
import json
import numpy as np
import pandas as pd
import train as T
import obsprofile as PR
from eval_hallym import frozen_preds, boot_idx
from sklearn.metrics import roc_auc_score as A

VARS = ["rr", "temp", "hr", "spo2", "sbp", "dbp", "bun", "lactate", "wbc", "hb", "na", "k", "cl", "ph", "be", "ag",
        "ne", "vaso", "fio2", "vent", "uo", "gcs"]
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn")
y = h["y24"].astype(int); S = (2, 3, 4)
ens = lambda tag, ds, seeds=S: np.mean([frozen_preds(f"{tag}_s{s}", ds, st, "cuda") for s in seeds], 0)[:, 0]
out = {}

# ---- A. where the gap sits ----
pR, pN = ens("W_base", h), ens("W_noreset", h)
out["auroc"] = {"reset": A(y, pR), "noreset": A(y, pN), "gap": A(y, pR) - A(y, pN)}
print("overall 24h AUROC reset %.4f noreset %.4f gap %+.4f" % (A(y, pR), A(y, pN), A(y, pR) - A(y, pN)), flush=True)
per_seed = [A(y, frozen_preds(f"W_base_s{s}", h, st, "cuda")[:, 0]) - A(y, frozen_preds(f"W_noreset_s{s}", h, st, "cuda")[:, 0]) for s in S]
out["gap_per_seed"] = per_seed; print("gap per seed", np.round(per_seed, 4), flush=True)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
rk = lambda p: (np.searchsorted(np.sort(p[neg]), p[pos], side="left") / len(neg))          # share of negatives below each positive
dpos = rk(pR) - rk(pN)                                                                         # per positive: rank change
rkn = lambda p: 1 - np.searchsorted(np.sort(p[pos]), p[neg], side="right") / len(pos)          # share of positives above each negative
dneg = rkn(pR) - rkn(pN)                                                                       # per negative: < 0 = reset ranks it above more positives
m, x = h["m"], h["x"]
feat = pd.DataFrame({"site": h["site"].astype(str), "L": h["landmark"].astype(float),
                     "hours_monitored": (m[:, :, [0, 2, 3]].max(2) > 0).sum(1)})
for v in ("temp", "spo2", "hr", "rr", "sbp", "bun", "wbc", "na"):
    feat[f"n_{v}"] = (m[:, :, VARS.index(v)] > 0).sum(1)
feat["labs_any"] = (m[:, :, PR.LAB] > 0).any((1, 2)).astype(int)
fp, fn = feat.iloc[pos].assign(d=dpos), feat.iloc[neg].assign(d=dneg)
out["positives"] = {"mean_rank_change": float(dpos.mean()), "share_lower_with_reset": float((dpos < 0).mean())}
out["negatives"] = {"mean_rank_change": float(dneg.mean())}
print("positives: mean rank change %+.4f (lower with reset in %.0f%%) | negatives: mean %+.4f" % (dpos.mean(), 100 * (dpos < 0).mean(), dneg.mean()), flush=True)
for col in ("site", "labs_any"):
    t = pd.DataFrame({"pos_rank_change": fp.groupby(col)["d"].mean(), "neg_rank_change": fn.groupby(col)["d"].mean(),
                      "n_pos": fp.groupby(col).size(), "n_neg": fn.groupby(col).size()})
    out[f"by_{col}"] = t.to_dict(); print(t.round(4).to_string(), flush=True)
fn["Lb"] = pd.cut(fn["L"], [-1, 23, 47, 71, 1e9]); fp["Lb"] = pd.cut(fp["L"], [-1, 23, 47, 71, 1e9])
t = pd.DataFrame({"pos": fp.groupby("Lb", observed=True)["d"].mean(), "neg": fn.groupby("Lb", observed=True)["d"].mean()}); print(t.round(4).to_string(), flush=True)
# which negative features go with the reset over-ranking them (Spearman)
cor = {c: float(pd.Series(fn[c]).corr(fn["d"], method="spearman")) for c in fn.columns if c not in ("site", "d", "Lb")}
out["neg_rank_change_spearman"] = cor; print("negatives: Spearman of rank change with", {k: round(v, 3) for k, v in cor.items()}, flush=True)
cor = {c: float(pd.Series(fp[c]).corr(fp["d"], method="spearman")) for c in fp.columns if c not in ("site", "d", "Lb")}
out["pos_rank_change_spearman"] = cor; print("positives: Spearman of rank change with", {k: round(v, 3) for k, v in cor.items()}, flush=True)

# ---- B. which input drives it: remove one variable / thin variables Hallym records more often than MIMIC ----
rate_h, rate_d = PR.obs_rate(h["m"]), PR.obs_rate(d["m"])
out["rates"] = {VARS[c]: [float(rate_d[c]), float(rate_h[c])] for c in range(22)}
print("hourly rate MIMIC vs Hallym:", {VARS[c]: (round(float(rate_d[c]), 3), round(float(rate_h[c]), 3)) for c in range(22) if rate_h[c] > 0}, flush=True)
present = [c for c in range(22) if rate_h[c] > 0]
out["remove"], out["thin"] = {}, {}
for c in present:
    xs, ms = h["x"].copy(), h["m"].copy()
    xs[:, :, c] = np.nan; ms[:, :, c] = 0
    ds = {**h, "x": xs, "m": ms}
    r_, n_ = ens("W_base", ds), ens("W_noreset", ds)
    out["remove"][VARS[c]] = {"reset": A(y, r_), "noreset": A(y, n_), "gap": A(y, r_) - A(y, n_)}
    print(f"remove {VARS[c]:8s} reset {A(y, r_):.4f} noreset {A(y, n_):.4f} gap {A(y, r_) - A(y, n_):+.4f}", flush=True)
for c in [c for c in present if rate_h[c] > rate_d[c]]:
    r_t = rate_h.copy(); r_t[c] = rate_d[c]
    gaps = []
    for k in range(3):
        xs, ms = PR.imagine_rate(h["x"], h["m"], rate_h, r_t, np.random.RandomState(k))
        ds = {**h, "x": xs, "m": ms}
        r_, n_ = ens("W_base", ds), ens("W_noreset", ds)
        gaps.append([A(y, r_), A(y, n_), A(y, r_) - A(y, n_)])
    g = np.mean(gaps, 0)
    out["thin"][VARS[c]] = {"reset": g[0], "noreset": g[1], "gap": g[2]}
    print(f"thin {VARS[c]:8s} to MIMIC rate: reset {g[0]:.4f} noreset {g[1]:.4f} gap {g[2]:+.4f}", flush=True)

# ---- C. does the 30%-trained pair show the same gap at Hallym? ----
S30 = (2, 3, 4, 5)
r30, n30 = ens("WXRbase", h, S30), ens("WXNbase", h, S30)
out["pair30"] = {"reset": A(y, r30), "noreset": A(y, n30), "gap": A(y, r30) - A(y, n30)}
print("30%%-trained pair at Hallym: reset %.4f noreset %.4f gap %+.4f" % (A(y, r30), A(y, n30), A(y, r30) - A(y, n30)), flush=True)
json.dump(out, open(f"{T.ROOT}/results/reset_gap_W.json", "w"), indent=1, default=float)
print("written results/reset_gap_W.json")
