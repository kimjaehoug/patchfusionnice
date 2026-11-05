"""Observation-regime simulation and label-free site descriptors.

Two regimes of missing sparse variables are distinguished:
  selective testing - a variable is measured preferentially in sicker patients (higher testing threshold);
  capture gaps      - a variable is missing independently of acuity (charting / interface differences).
A site is described per variable by (log measurement-rate ratio to the development cohort, selectivity), where
selectivity = AUROC(vital-sign acuity -> ever measured) - 0.5. Acuity comes from the always-observed vital signs only,
so the descriptor needs no outcome labels at the deployment site.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

VARS = ["rr", "temp", "hr", "spo2", "sbp", "dbp", "bun", "lactate", "wbc", "hb", "na", "k", "cl", "ph", "be", "ag",
        "ne", "vaso", "fio2", "vent", "uo", "gcs"]
CARRY = {"rr": 12, "temp": 12, "hr": 12, "spo2": 12, "sbp": 12, "dbp": 12, "fio2": 12,
         "ph": 4, "be": 4, "lactate": 4, "bun": 8, "na": 8, "k": 8, "cl": 8, "ag": 8, "wbc": 12, "hb": 12,
         "ne": 0, "vaso": 0, "vent": 0, "uo": 0, "gcs": 0}
VIT = [VARS.index(v) for v in ["rr", "temp", "hr", "spo2", "sbp", "dbp"]]
SIM = [c for c in range(len(VARS)) if c not in VIT]
DESC_DIM = 2 * len(VARS)


def _vit_feats(x):
    v = x[:, :, VIT]
    f = np.concatenate([np.nanmean(v, 1), np.nanmin(v, 1), np.nanmax(v, 1), v[:, -1]], 1)
    return f


class Acuity:
    """vital-sign-only risk score fitted on development training data (yhosp)."""

    def fit(self, x, y):
        f = _vit_feats(x)
        self.med = np.nanmedian(f, 0)
        f = np.where(np.isnan(f), self.med, f)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-6
        self.lr = LogisticRegression(max_iter=2000).fit((f - self.mu) / self.sd, y)
        return self

    def __call__(self, x):
        f = _vit_feats(x)
        f = np.where(np.isnan(f), self.med, f)
        return self.lr.decision_function((f - self.mu) / self.sd)


def measured(m):
    return m.sum(1) > 0                                   # [N, C] ever measured in the window


def descriptor(meas, acu, ref_rate):
    rate = meas.mean(0)
    sel = np.zeros(meas.shape[1])
    for c in range(meas.shape[1]):
        y = meas[:, c]
        sel[c] = roc_auc_score(y, acu) - 0.5 if 0 < y.sum() < len(y) else 0.0
    return np.concatenate([np.log((rate + 1e-2) / (ref_rate + 1e-2)), sel]).astype(np.float32)


def site_descriptors(m, acu, site, ref_rate):
    """per-sample descriptor of the sample's site, from unlabeled data only."""
    meas = measured(m)
    out = np.zeros((len(m), DESC_DIM), np.float32)
    for s in np.unique(site):
        i = site == s
        out[i] = descriptor(meas[i], acu[i], ref_rate)
    return out


def refill(raw):
    """re-apply the persistence rules of build_tensor.py to raw (NaN = not observed) values."""
    m = (~np.isnan(raw)).astype(np.float32)
    x = raw.copy()
    N, T, _ = raw.shape
    for c, v in enumerate(VARS):
        w = CARRY[v]
        if w == 0:
            continue
        last = np.full(N, np.nan, np.float32)
        age = np.full(N, 10 ** 6)
        for t in range(T):
            o = m[:, t, c] > 0
            last[o] = raw[o, t, c]
            age[o] = 0
            f = (~o) & (age <= w) & ~np.isnan(last)
            x[f, t, c] = last[f]
            age = age + 1
    return x, m


def simulate(x, m, acu, ref_rate, rng, n_sites=8):
    """split the cohort into n_sites pseudo-sites (the first keeps the native regime); in each, every sparse variable
    gets a target keep rate q ~ logU(0.15, 1) and either capture-gap (random) or selective (acuity-ranked) removal of
    all its observations for a patient. Returns refilled x, m and each sample's site descriptor."""
    N = len(x)
    order = rng.permutation(N)
    xs, ms = x.copy(), m.copy()
    desc = np.zeros((N, DESC_DIM), np.float32)
    for k, idx in enumerate(np.array_split(order, n_sites)):
        if k > 0:
            ar = acu[idx].argsort().argsort() / max(len(idx) - 1, 1)
            drop = np.zeros((len(idx), len(VARS)), bool)
            for c in SIM:
                q = np.exp(rng.uniform(np.log(0.15), 0.0))
                if rng.rand() < 0.5:
                    keep = rng.rand(len(idx)) < q
                else:
                    beta = rng.uniform(2, 10)
                    lo, hi = -30.0, 30.0
                    for _ in range(40):
                        mid = (lo + hi) / 2
                        p = 1 / (1 + np.exp(-beta * (ar - 0.5) - mid))
                        lo, hi = (mid, hi) if p.mean() < q else (lo, mid)
                    keep = rng.rand(len(idx)) < p
                drop[:, c] = ~keep
            raw = np.where(m[idx] > 0, x[idx], np.nan)
            raw = np.where(drop[:, None, :], np.nan, raw)
            xs[idx], ms[idx] = refill(raw)
        desc[idx] = descriptor(measured(ms[idx]), acu[idx], ref_rate)
    return xs, ms, desc


# ---------------------------------------------------------------- observation-process-bounded inputs
# fixed clinical critical values (raw units); None = no anchor on that side
ANCHORS = {"rr": (8, 30), "temp": (35, 39), "hr": (50, 130), "spo2": (90, None), "sbp": (90, 180), "dbp": (40, 110),
           "bun": (None, 50), "lactate": (None, 4), "wbc": (2, 20), "hb": (7, None), "na": (125, 155), "k": (3, 6),
           "cl": (90, 120), "ph": (7.25, 7.55), "be": (-10, None), "ag": (None, 20), "ne": (None, 0.25),
           "vaso": (None, 0.5), "fio2": (None, 60), "vent": (None, 0.5), "uo": (30, None), "gcs": (8, None)}


def anchors_z(mu, sd):
    """critical values on the standardised scale used by the network inputs."""
    lo = np.array([-1e4 if ANCHORS[v][0] is None else (ANCHORS[v][0] - mu[i]) / sd[i] for i, v in enumerate(VARS)], np.float32)
    hi = np.array([1e4 if ANCHORS[v][1] is None else (ANCHORS[v][1] - mu[i]) / sd[i] for i, v in enumerate(VARS)], np.float32)
    return lo, hi


def evidence_bounds(x, m, y, q_keep=(1.0, 0.75, 0.5), min_sel=0.05):
    """per-variable bound (logit units) on the evidence a measurement indicator may contribute.
    For each variable, testing thresholds are raised by keeping tests only in the most acute q of tested patients;
    the indicator's adjusted log-odds ratio (outcome ~ acuity + indicator) is re-estimated under each threshold and the
    bound is the smallest magnitude if the sign is stable, else 0. Variables whose testing does not depend on acuity
    (capture/documentation gaps, selectivity < min_sel) get bound 0."""
    acu = Acuity().fit(x, y)
    a = acu(x)
    meas = measured(m)
    za = (a - a.mean()) / a.std()
    b = np.zeros(len(VARS), np.float32)
    for c in range(len(VARS)):
        t = meas[:, c]
        if not (0.01 < t.mean() < 0.99):
            continue
        if roc_auc_score(t, a) - 0.5 < min_sel:
            continue
        gs = []
        for q in q_keep:
            keep = t.copy()
            if q < 1.0:
                cut = np.quantile(a[t], 1 - q)
                keep = t & (a >= cut)
            X = np.column_stack([za, keep.astype(float)])
            gs.append(LogisticRegression(max_iter=1000).fit(X, y).coef_[0, 1])
        gs = np.array(gs)
        b[c] = np.abs(gs).min() if (np.all(gs > 0) or np.all(gs < 0)) else 0.0
    return b
