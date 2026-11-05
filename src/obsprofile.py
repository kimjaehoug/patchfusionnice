"""Label-free observation profile of a deployment site and profile-matched observation imagination.

Profile (no outcome labels at the target site):
  hour scale  - for each laboratory variable v, P(v measured in 3-h block b) = sigmoid(beta_v * a_b + alpha_v), with
                block acuity a_b from vital signs only; beta_v is fitted on the development data and shared, the site
                intercept alpha_v (testing threshold) is estimated by maximum likelihood at each site;
  stay scale  - for each documentation stream (GCS, urine output, norepinephrine, vasopressin, ventilation), the share
                of windows in which the stream is recorded at all;
  site scale  - variables never recorded at the site.
Imagination re-observes development samples as the target site would: laboratory observations are kept in block b with
probability min(1, p_target(a_b) / p_dev(a_b)) ("patch" mode, block acuity) or with the same probability computed from the
window acuity and applied to all observations of the variable ("seq" mode); documentation streams are removed per window
with probability 1 - rho_target / rho_dev; absent variables are removed. Values are re-imputed with the persistence rules.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from regime import VARS, VIT, refill

LAB = list(range(6, 16))                                   # bun .. ag
CAD_HOURLY = False
KEEP_ONE = True                                            # cadence thinning keeps one tested block per window
FREE_SLOPE = False                                         # window-selection slope fitted at the target
RAND_VITALS_FIXED = False                                  # random profiles keep development vital cadence
DOC = [VARS.index(v) for v in ["ne", "vaso", "fio2", "vent", "uo", "gcs"]]
BLK = 3


def _block_vit(x):
    """[N,T,22] -> [N,K,24] block vital features (mean, min, max, last of 6 vitals), NaN-robust."""
    N, T, _ = x.shape
    K = T // BLK
    v = x[:, :K * BLK, VIT].reshape(N, K, BLK, len(VIT))
    with np.errstate(all="ignore"):
        f = np.concatenate([np.nanmean(v, 2), np.nanmin(v, 2), np.nanmax(v, 2), v[:, :, -1]], -1)
    return f


class BlockAcuity:
    """vital-sign-only acuity of each 3-h block, fitted on development data (in-hospital death of the stay)."""

    def fit(self, x, y, n=40000, seed=0):
        i = np.random.RandomState(seed).permutation(len(x))[:n]
        f = _block_vit(x[i])
        K = f.shape[1]
        f = f.reshape(-1, f.shape[-1])
        yy = np.repeat(y[i], K)
        ok = ~np.isnan(f).all(1)
        self.med = np.nanmedian(f[ok], 0)
        g = np.where(np.isnan(f[ok]), self.med, f[ok])
        self.mu, self.sd = g.mean(0), g.std(0) + 1e-6
        self.lr = LogisticRegression(max_iter=2000).fit((g - self.mu) / self.sd, yy[ok])
        return self

    def __call__(self, x):
        f = _block_vit(x)
        N, K, D = f.shape
        f = f.reshape(-1, D)
        none = np.isnan(f).all(1)
        g = np.where(np.isnan(f), self.med, f)
        a = self.lr.decision_function((g - self.mu) / self.sd)
        a[none] = np.nan                                    # blocks before admission
        return a.reshape(N, K)


def _block_measured(m):
    N, T, _ = m.shape
    K = T // BLK
    return m[:, :K * BLK].reshape(N, K, BLK, -1).max(2) > 0      # [N,K,22]


def _fit_intercept(y, a, beta, iters=60):
    """MLE of alpha in sigmoid(beta*a + alpha) with beta fixed: the score equation sum(y - p) = 0, solved by bisection."""
    ybar = y.mean()
    lo, hi = -30.0, 30.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-np.clip(beta * a + mid, -40, 40)))
        lo, hi = (mid, hi) if p.mean() < ybar else (lo, mid)
    return (lo + hi) / 2


class Profile:
    def __init__(self, beta, alpha, rho, absent, n, beta_w=None, alpha_w=None, cad=None):
        self.beta, self.alpha, self.rho, self.absent, self.n = beta, alpha, rho, absent, n
        self.beta_w, self.alpha_w, self.cad = beta_w, alpha_w, cad      # window selection and within-window cadence

    def summary(self):
        lab = " ".join(f"{VARS[c]}:{self.alpha[c]:+.2f}" for c in LAB)
        doc = " ".join(f"{VARS[c]}:{self.rho[c]:.2f}" for c in DOC)
        return f"n={self.n} | lab intercepts {lab} | documentation rates {doc} | absent {[VARS[c] for c in self.absent]}"


def estimate(x, m, acu, beta=None, n=40000, seed=0):
    """profile of one site from its (x, raw mask) samples; beta None -> fit beta (development site)."""
    i = np.random.RandomState(seed).permutation(len(x))[:n]
    a = acu(x[i])
    bm = _block_measured(m[i])
    ok = ~np.isnan(a)
    C = len(VARS)
    b = np.zeros(C) if beta is None else beta.copy()
    al = np.zeros(C)
    for c in LAB:
        y = bm[..., c][ok].astype(float)
        aa = a[ok]
        if beta is None:
            lr = LogisticRegression(max_iter=1000).fit(aa[:, None], y)
            b[c], al[c] = lr.coef_[0, 0], lr.intercept_[0]
        else:
            al[c] = _fit_intercept(y, aa, b[c]) if y.sum() > 0 else -30.0
    win = m[i].max(1) > 0                                   # [n,22] recorded anywhere in the window
    rho = win.mean(0)
    absent = [c for c in range(C) if c not in VIT and rho[c] == 0]
    # two-part hour scale: (1) which windows are tested, as a function of window acuity; (2) cadence = share of
    # observed blocks that carry a test, among tested windows (acuity-independent within the window)
    with np.errstate(all="ignore"):
        aw = np.nanmean(a, 1)
    okw = ~np.isnan(aw)
    nb = ok.sum(1)
    bw = np.zeros(C) if beta is None else (getattr(beta, "beta_w", None) if False else None)
    bw = np.zeros(C)
    alw = np.zeros(C)
    cad = np.zeros(C)
    for c in LAB:
        tested = bm[..., c].sum(1) > 0
        yw, aa = tested[okw].astype(float), aw[okw]
        if beta is None:
            lr = LogisticRegression(max_iter=1000).fit(aa[:, None], yw)
            bw[c], alw[c] = lr.coef_[0, 0], lr.intercept_[0]
        cad[c] = (bm[..., c].sum(1)[tested] / np.maximum(nb[tested], 1)).mean() if tested.any() else 0.0
    prof = Profile(b, al, rho, absent, len(i), bw, alw, cad)
    # documentation cadence: share of monitored hours with a record, among windows in which the stream is documented
    mon = m[i][:, :, VIT].max(2) > 0
    dcad = np.zeros(C)
    for c in DOC:
        rec = m[i][:, :, c] > 0
        doc = rec.any(1)
        dcad[c] = (rec.sum(1)[doc] / np.maximum(mon.sum(1)[doc], 1)).mean() if doc.any() else 0.0
    prof.dcad = dcad
    vcad = np.zeros(C)                                      # vital-sign hourly cadence among monitored hours
    for c in VIT:
        vcad[c] = (m[i][:, :, c] > 0).sum() / max(mon.sum(), 1)
    prof.vcad = vcad
    prof._aw, prof._tested = aw[okw], {c: (bm[..., c].sum(1) > 0)[okw] for c in LAB}
    return prof


def fit_window(tgt, dev):
    """site intercepts of the window-selection model with the development slope fixed."""
    tgt.beta_w = dev.beta_w.copy()
    for c in LAB:
        y = tgt._tested[c].astype(float)
        if FREE_SLOPE and 20 <= y.sum() <= len(y) - 20:     # target's own acuity slope (label-free)
            lr = LogisticRegression(max_iter=1000).fit(tgt._aw[:, None], y)
            tgt.beta_w[c], tgt.alpha_w[c] = lr.coef_[0, 0], lr.intercept_[0]
        else:
            tgt.alpha_w[c] = _fit_intercept(y, tgt._aw, tgt.beta_w[c]) if y.sum() > 0 else -30.0
    return tgt


def imagine_two(x, m, acu, dev, tgt, parts=("sel", "cad"), rng=None):
    """two-part hour-scale imagination: 'sel' keeps a laboratory variable in the window with probability
    p_target(a_w) / p_dev(a_w) (window acuity); 'cad' keeps each tested 3-h block with probability
    cad_target / cad_dev (at least one tested block is kept in a kept window). Documentation and absent
    variables as in imagine()."""
    rng = rng or np.random
    N, T, C = x.shape
    K = T // BLK
    raw = np.where(m > 0, x, np.nan).astype(np.float32)
    a = acu(x)
    with np.errstate(all="ignore"):
        aw = np.nan_to_num(np.nanmean(a, 1), nan=0.0)
    bm = _block_measured(m)                                 # [N,K,22]
    keep = np.ones((N, T, C), bool)
    for c in LAB:
        kb = np.ones((N, K), bool)
        if "sel" in parts:
            pd_ = 1 / (1 + np.exp(-(dev.beta_w[c] * aw + dev.alpha_w[c])))
            pt_ = 1 / (1 + np.exp(-(tgt.beta_w[c] * aw + tgt.alpha_w[c])))
            kw = rng.rand(N) < np.clip(pt_ / np.maximum(pd_, 1e-6), 0, 1)
            kb &= kw[:, None]
        kb_sel = kb.copy()
        if "cad" in parts and dev.cad[c] > 0:
            r = min(1.0, tgt.cad[c] / dev.cad[c])
            kc = rng.rand(N, K) < r
            tested = bm[..., c]
            none = ~(kc & tested).any(1) & tested.any(1)      # keep one tested block if all were dropped
            if none.any() and KEEP_ONE:
                pick = np.array([rng.choice(np.where(tested[n])[0]) for n in np.where(none)[0]])
                kc[np.where(none)[0], pick] = True
            kb &= kc
        keep[:, :K * BLK, c] = np.repeat(kb, BLK, 1)
        if CAD_HOURLY and "cad" in parts and dev.cad[c] > 0:     # smoke: cadence thinned per hour, not per 3-h block
            keep[:, :K * BLK, c] = np.repeat(kb_sel, BLK, 1) & (rng.rand(N, K * BLK) < min(1.0, tgt.cad[c] / dev.cad[c]))
    explicit = "explicit" in parts                          # absent / streams only when named (damage-ranked subsets)
    for c in DOC:
        if dev.rho[c] > 0:
            u = rng.rand(N, 1)                              # drawn either way: identical random stream
            if not explicit or "streams" in parts:
                keep[:, :, c] &= (u < min(1.0, tgt.rho[c] / dev.rho[c]))
        if "doccad" in parts and getattr(dev, "dcad", None) is not None and dev.dcad[c] > 0:
            r = min(1.0, tgt.dcad[c] / dev.dcad[c])                 # hourly documentation cadence
            rec = m[:, :, c] > 0
            kh = rng.rand(N, T) < r
            none = ~(kh & rec).any(1) & rec.any(1)
            for n in np.where(none)[0]:
                kh[n, rng.choice(np.where(rec[n])[0])] = True
            keep[:, :, c] &= kh | ~rec
    if "vitcad" in parts and getattr(dev, "vcad", None) is not None:
        for c in VIT:                                       # hourly vital-sign cadence (can only be thinned)
            if dev.vcad[c] > 0 and tgt.vcad[c] < dev.vcad[c]:
                keep[:, :, c] &= rng.rand(N, T) < tgt.vcad[c] / dev.vcad[c]
    for c in (tgt.absent if not explicit or "absent" in parts else []):
        keep[:, :, c] = False
    raw = np.where(keep, raw, np.nan)
    xs, ms = refill(raw)
    first = np.cumsum(m > 0, 1) == 0
    carried_in = (m == 0) & ~np.isnan(x) & first
    fill = carried_in & keep.any(1, keepdims=True) & np.isnan(xs)
    xs = np.where(fill, x, xs)
    return xs.astype(np.float32), ms.astype(np.float32)


def imagine(x, m, acu, dev, tgt, mode="patch", rng=None):
    """re-observe development samples (x filled, m raw mask) as the target site would."""
    rng = rng or np.random
    N, T, C = x.shape
    K = T // BLK
    raw = np.where(m > 0, x, np.nan).astype(np.float32)
    a = acu(x)                                              # [N,K] block acuity
    if mode == "seq":                                       # one acuity per window, broadcast to its blocks
        with np.errstate(all="ignore"):
            aw = np.nanmean(a, 1, keepdims=True)
        a = np.where(np.isnan(a), np.nan, np.repeat(aw, K, 1))
    if mode != "seq":
        aw = None
    a0 = np.nan_to_num(a, nan=0.0)
    keep = np.ones((N, T, C), bool)
    for c in LAB:
        pd_ = 1 / (1 + np.exp(-(dev.beta[c] * a0 + dev.alpha[c])))
        pt_ = 1 / (1 + np.exp(-(dev.beta[c] * a0 + tgt.alpha[c])))
        q = np.clip(pt_ / np.maximum(pd_, 1e-6), 0, 1)       # [N,K]
        if mode == "patch":                                 # independent decision per 3-h block
            kb = rng.rand(N, K) < q
        else:                                               # one decision per window and variable, window acuity
            aw0 = np.nan_to_num(aw[:, 0], nan=0.0)
            pdw = 1 / (1 + np.exp(-(dev.beta[c] * aw0 + dev.alpha[c])))
            ptw = 1 / (1 + np.exp(-(dev.beta[c] * aw0 + tgt.alpha[c])))
            qw = np.clip(ptw / np.maximum(pdw, 1e-6), 0, 1)
            kb = np.repeat((rng.rand(N) < qw)[:, None], K, 1)
        keep[:, :K * BLK, c] = np.repeat(kb, BLK, 1)
    for c in DOC:
        if dev.rho[c] > 0:
            qd = min(1.0, tgt.rho[c] / dev.rho[c])
            keep[:, :, c] &= (rng.rand(N, 1) < qd)
    for c in tgt.absent:
        keep[:, :, c] = False
    raw = np.where(keep, raw, np.nan)
    xs, ms = refill(raw)
    # values carried into the window from before its start are not observations; keep them unless the variable is
    # removed for the whole window
    first = np.cumsum(m > 0, 1) == 0
    carried_in = (m == 0) & ~np.isnan(x) & first
    whole = keep.any(1, keepdims=True)
    fill = carried_in & whole & np.isnan(xs)
    xs = np.where(fill, x, xs)
    return xs.astype(np.float32), ms.astype(np.float32)


def target_sites(name, min_n=500):
    """list of (x, m) per hospital of a target source: 'hallym_dyn:sites' or 'eicu_dyn:low:sites' (hospitals of a
    testing-intensity tertile with at least min_n samples)."""
    import train as T
    parts = name.split(":")
    e = T.load(parts[0])
    site = np.asarray(e["site"]).astype(str)
    keep = np.ones(len(site), bool)
    if len(parts) == 3:                                     # restrict to a tertile
        import pandas as pd
        crit = [VARS.index(v) for v in ["lactate", "ph", "be", "ne", "uo", "gcs"]]
        hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
        big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
        t = np.full(len(hr), None, object)
        t[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
        keep = t == parts[1]
    out = []
    for s in np.unique(site[keep]):
        i = keep & (site == s)
        if i.sum() >= min_n:
            out.append((e["x"][i], e["m"][i]))
    return out


def target_samples(name):
    """unlabeled target data: 'hallym_dyn' or 'eicu_dyn:low|mid|high' (hospital testing-intensity tertile)."""
    import pandas as pd
    import train as T
    src, _, tert = name.partition(":")
    e = T.load(src)
    if not tert:
        return e["x"], e["m"]
    site = e["site"]
    crit = [VARS.index(v) for v in ["lactate", "ph", "be", "ne", "uo", "gcs"]]
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    t = np.full(len(hr), None, object)
    t[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    i = np.where(t == tert)[0]
    return e["x"][i], e["m"][i]


def perturbed(dev, tgt, kind, rng):
    """'random': a random profile around the development site (generic imagination);
    'jitter': the target profile with estimation noise (threshold +-, rates x, absent variables kept with p=0.2)."""
    import copy
    p = copy.deepcopy(tgt)
    C = len(VARS)
    if kind == "random":
        p.alpha = dev.alpha + rng.uniform(-2.5, 0.0, C)
        if dev.alpha_w is not None:
            p.alpha_w = dev.alpha_w + rng.uniform(-2.5, 0.0, C)
            p.cad = dev.cad * rng.uniform(0.5, 1.0, C)
        if getattr(dev, "dcad", None) is not None:
            p.dcad = dev.dcad * rng.uniform(0.2, 1.0, C)
        if getattr(dev, "vcad", None) is not None:
            p.vcad = dev.vcad.copy() if RAND_VITALS_FIXED else dev.vcad * np.where(rng.rand(C) < 0.2, rng.uniform(0.1, 1.0, C), 1.0)
        p.rho = dev.rho * rng.uniform(0.3, 1.0, C)
        p.absent = [c for c in range(C) if c not in VIT and rng.rand() < 0.25]
    else:
        p.alpha = tgt.alpha + rng.normal(0, 0.5, C)
        p.rho = np.clip(tgt.rho * rng.uniform(0.7, 1.3, C), 0, 1)
        p.absent = [c for c in tgt.absent if rng.rand() < 0.8]
    return p


def imagine_mixed(x, m, acu, dev, tgt, mode="patch", scheme="mix", rng=None, chunk=2000, parts=None, tfrac=0.5):
    """imagination with a fresh profile per chunk of samples: scheme 'mix' = half target / half random profiles,
    'jitter' = jittered target profiles."""
    rng = rng or np.random
    xs, ms = np.empty_like(x), np.empty_like(m)
    for i in range(0, len(x), chunk):
        tg_i = tgt[rng.randint(len(tgt))] if isinstance(tgt, list) else tgt     # several site profiles
        if scheme == "mix":
            pr = tg_i if rng.rand() < tfrac else perturbed(dev, tg_i, "random", rng)
        else:
            pr = perturbed(dev, tg_i, "jitter", rng)
        if parts:
            xs[i:i + chunk], ms[i:i + chunk] = imagine_two(x[i:i + chunk], m[i:i + chunk], acu, dev, pr, parts=parts, rng=rng)
        else:
            xs[i:i + chunk], ms[i:i + chunk] = imagine(x[i:i + chunk], m[i:i + chunk], acu, dev, pr, mode=mode, rng=rng)
    return xs, ms


def obs_rate(m):
    """hourly observation share of every variable (all windows, all hours)."""
    return (m > 0).mean((0, 1))


def imagine_naive(x, m, r_dev, r_tgt, rng=None):
    """naive re-masking (comparator, as for static features): every available entry of the hourly input (recorded or
    carried forward) is dropped independently with probability 1 - min(1, r_tgt[v] / r_dev[v]); masks follow; values are
    NOT re-imputed, so carried-forward runs keep values whose recording was removed and get holes."""
    rng = rng or np.random
    N, T, C = x.shape
    q = np.clip(np.asarray(r_tgt) / np.maximum(np.asarray(r_dev), 1e-9), 0, 1)
    keep = rng.rand(N, T, C) < q[None, None, :]
    xs = np.where(keep, x, np.nan).astype(np.float32)
    ms = np.where(keep, m, 0).astype(np.float32)
    return xs, ms


def imagine_rate(x, m, r_dev, r_tgt, rng=None):
    """rate-matched thinning (comparator): every recorded value of variable v is kept independently with probability
    min(1, r_tgt[v] / r_dev[v]); no acuity-dependent selection, no stay- or patch-level structure. Re-imputed as in
    imagine_two."""
    rng = rng or np.random
    N, T, C = x.shape
    raw = np.where(m > 0, x, np.nan).astype(np.float32)
    q = np.clip(np.asarray(r_tgt) / np.maximum(np.asarray(r_dev), 1e-9), 0, 1)
    keep = rng.rand(N, T, C) < q[None, None, :]
    raw = np.where(keep, raw, np.nan)
    xs, ms = refill(raw)
    first = np.cumsum(m > 0, 1) == 0
    carried_in = (m == 0) & ~np.isnan(x) & first
    fill = carried_in & (q[None, None, :] > 0) & np.isnan(xs)
    xs = np.where(fill, x, xs)
    return xs.astype(np.float32), ms.astype(np.float32)
