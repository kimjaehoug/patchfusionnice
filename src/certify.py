"""Interval bound propagation (IBP) through a trained HorizonPF (PatchFusion encoder, queries=False) to certify the
predicted cumulative incidence of death against observation-process changes.

Threat model (observation process only, physiology untouched): any subset of the sparse-variable measurements inside a
time window may be missing; values are then re-imputed with the persistence rules. Every input cell therefore takes one of
a finite set of values; the input box is the hull of {original, every value the re-imputation could produce, absent}.

With per-patch state reset the perturbation reaches only the patches that overlap the window plus the carry horizon of the
affected variables (the footprint); all other patch tokens stay exact points. With a carried state, every token after the
window start is affected. Bounds are propagated with standard interval arithmetic (eval mode, no dropout).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

import regime as RG
import train as T
import models as M


class Box:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = lo, (lo if hi is None else hi)

    @property
    def c(self):
        return (self.lo + self.hi) / 2

    @property
    def r(self):
        return (self.hi - self.lo) / 2

    def linear(self, lin):
        c = F.linear(self.c, lin.weight, lin.bias)
        r = F.linear(self.r, lin.weight.abs())
        return Box(c - r, c + r)

    def mono(self, fn):                      # monotone non-decreasing elementwise function
        return Box(fn(self.lo), fn(self.hi))

    def __add__(self, o):
        return Box(self.lo + o.lo, self.hi + o.hi) if isinstance(o, Box) else Box(self.lo + o, self.hi + o)

    def mul(self, o):                        # elementwise interval product
        cands = torch.stack([self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi])
        return Box(cands.min(0).values, cands.max(0).values)

    def slice(self, *idx):
        return Box(self.lo[idx], self.hi[idx])


def box_cat(boxes, dim):
    return Box(torch.cat([b.lo for b in boxes], dim), torch.cat([b.hi for b in boxes], dim))


def lstm_dir(lstm, x, reverse):
    """x: Box [N,P,d_in]; one direction of a single-layer nn.LSTM. Returns final hidden Box [N,H]."""
    sfx = "_reverse" if reverse else ""
    Wih, Whh = getattr(lstm, "weight_ih_l0" + sfx), getattr(lstm, "weight_hh_l0" + sfx)
    b = getattr(lstm, "bias_ih_l0" + sfx) + getattr(lstm, "bias_hh_l0" + sfx)
    N, P, _ = x.lo.shape
    H = Whh.shape[1]
    h = Box(x.lo.new_zeros(N, H))
    c = Box(x.lo.new_zeros(N, H))
    steps = range(P - 1, -1, -1) if reverse else range(P)
    for t in steps:
        xt = Box(x.lo[:, t], x.hi[:, t])
        gc = F.linear(xt.c, Wih) + F.linear(h.c, Whh) + b
        gr = F.linear(xt.r, Wih.abs()) + F.linear(h.r, Whh.abs())
        g = Box(gc - gr, gc + gr)
        i, f, gg, o = [g.slice(slice(None), slice(k * H, (k + 1) * H)) for k in range(4)]
        i, f, o = i.mono(torch.sigmoid), f.mono(torch.sigmoid), o.mono(torch.sigmoid)
        gg = gg.mono(torch.tanh)
        c = f.mul(c) + i.mul(gg)
        h = o.mul(c.mono(torch.tanh))
    return h


def layernorm_box(ln, x):
    """per-token LayerNorm; exact for point tokens, sound but loose otherwise."""
    d = x.lo.shape[-1]
    mu = Box(x.lo.mean(-1, keepdim=True), x.hi.mean(-1, keepdim=True))
    cen = Box(x.lo - mu.hi, x.hi - mu.lo)                               # x - mean
    sq = torch.maximum(cen.lo ** 2, cen.hi ** 2)
    sq_lo = torch.where((cen.lo <= 0) & (cen.hi >= 0), torch.zeros_like(sq), torch.minimum(cen.lo ** 2, cen.hi ** 2))
    var = Box(sq_lo.mean(-1, keepdim=True), sq.mean(-1, keepdim=True))
    inv = Box(1 / torch.sqrt(var.hi + ln.eps), 1 / torch.sqrt(var.lo + ln.eps))
    y = cen.mul(inv)
    bound = math.sqrt(d - 1)                                            # |normalised coord| <= sqrt(d-1)
    y = Box(y.lo.clamp(-bound, bound), y.hi.clamp(-bound, bound))
    pt = (x.hi - x.lo).abs().amax(-1, keepdim=True) < 1e-6             # exact for unperturbed tokens
    ex = F.layer_norm((x.lo + x.hi) / 2, (d,), None, None, ln.eps)
    y = Box(torch.where(pt, ex, y.lo), torch.where(pt, ex, y.hi))
    g, b = ln.weight, ln.bias
    c = y.c * g + b
    r = y.r * g.abs()
    return Box(c - r, c + r)


def rope_box(x, pos):
    """x Box [B,H,L,Dh]; RoPE is a fixed rotation per position."""
    dh = x.lo.shape[-1]
    half = dh // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=x.lo.device, dtype=x.lo.dtype) / half))
    ang = pos[:, None].to(x.lo.dtype) * inv[None]
    cos, sin = ang.cos(), ang.sin()
    c, r = x.c, x.r
    c1, c2, r1, r2 = c[..., :half], c[..., half:], r[..., :half], r[..., half:]
    cc = torch.cat([c1 * cos - c2 * sin, c1 * sin + c2 * cos], -1)
    rr = torch.cat([r1 * cos.abs() + r2 * sin.abs(), r1 * sin.abs() + r2 * cos.abs()], -1)
    return Box(cc - rr, cc + rr)


def attention_box(att, x, pos, bias):
    B, L, D = x.lo.shape
    qkv = x.linear(att.qkv)
    sh = lambda t: t.view(B, L, 3, att.h, att.dh).permute(2, 0, 3, 1, 4)
    lo, hi = sh(qkv.lo), sh(qkv.hi)
    q, k, v = Box(lo[0], hi[0]), Box(lo[1], hi[1]), Box(lo[2], hi[2])
    if pos is not None:
        q, k = rope_box(q, pos), rope_box(k, pos)
    s = 1.0 / math.sqrt(att.dh)
    # scores = q k^T: interval matrix product
    qc, qr, kc, kr = q.c, q.r, k.c, k.r
    sc = qc @ kc.transpose(-1, -2)
    sr = qr @ kc.abs().transpose(-1, -2) + qc.abs() @ kr.transpose(-1, -2) + qr @ kr.transpose(-1, -2)
    S = Box((sc - sr) * s, (sc + sr) * s)
    if bias is not None:
        S = S + bias[:, None, None, :]
    # softmax bounds: w_j in [e^{lo_j}/(e^{lo_j}+sum_{i!=j} e^{hi_i}), e^{hi_j}/(e^{hi_j}+sum_{i!=j} e^{lo_i})]
    Lk = S.lo.shape[-1]
    eye = torch.eye(Lk, dtype=torch.bool, device=S.lo.device)
    # log-space: w_lo_j = exp(lo_j - lse({lo_j} U {hi_i: i != j})), w_hi_j = exp(hi_j - lse({hi_j} U {lo_i: i != j}))
    A = torch.where(eye, S.lo.unsqueeze(-1), S.hi.unsqueeze(-2))       # [..., j, i]
    Bm = torch.where(eye, S.hi.unsqueeze(-1), S.lo.unsqueeze(-2))
    wlo = torch.exp(S.lo - torch.logsumexp(A, -1))
    whi = torch.exp(S.hi - torch.logsumexp(Bm, -1))
    # output = sum_j w_j v_j with sum w = 1: per coordinate, worst case over the weight box
    vlo, vhi = v.lo.unsqueeze(-3), v.hi.unsqueeze(-3)                  # [B,H,1,L,Dh]
    wl, wh = wlo.unsqueeze(-1), whi.unsqueeze(-1)                      # [B,H,L,L,1]
    out_lo = _weighted_extreme(wl, wh, vlo, minimise=True)
    out_hi = _weighted_extreme(wl, wh, vhi, minimise=False)
    o = Box(out_lo.transpose(1, 2).reshape(B, L, D), out_hi.transpose(1, 2).reshape(B, L, D))
    return o.linear(att.o)


def _weighted_extreme(wl, wh, val, minimise):
    """min (or max) of sum_j w_j val_j subject to wl<=w<=wh, sum w = 1 (greedy on sorted values)."""
    v = val.expand(*wl.shape[:-1], val.shape[-1])                     # [B,H,Lq,Lk,Dh]
    order = v.argsort(dim=-2, descending=not minimise)
    vs = v.gather(-2, order)
    wls = wl.expand_as(v).gather(-2, order)
    whs = wh.expand_as(v).gather(-2, order)
    budget = 1 - wls.sum(-2, keepdim=True)                             # mass to distribute beyond the lower bounds
    extra = (whs - wls)
    cum = torch.cumsum(extra, -2) - extra
    take = torch.clamp(torch.minimum(extra, budget - cum), min=0)
    w = wls + take
    return (w * vs).sum(-2)


def block_box(blk, z, pos, bias):
    a = attention_box(blk.att, layernorm_box(blk.n1, z), pos, bias)
    z = z + a
    y = layernorm_box(blk.n2, z).linear(blk.ff[0])
    y = Box(_gelu_lo(y.lo, y.hi), _gelu_hi(y.lo, y.hi))
    y = y.linear(blk.ff[3])
    return z + y


GELU_MIN_X, GELU_MIN = -0.7517916, -0.1699792


def _gelu_lo(lo, hi):
    g = torch.minimum(F.gelu(lo), F.gelu(hi))
    inside = (lo <= GELU_MIN_X) & (hi >= GELU_MIN_X)
    return torch.where(inside, torch.full_like(g, GELU_MIN), g)


def _gelu_hi(lo, hi):
    return torch.maximum(F.gelu(lo), F.gelu(hi))


@torch.no_grad()
def bound_cif(model, Xbox):
    """Xbox: Box [B,T,C] input interval -> (lo, hi) of CIF [B,4]."""
    enc = model.enc
    assert not enc.obsb and not enc.streams and not enc.ctx_dim and not enc.ta_cuts and not enc.ple_bins
    B = Xbox.lo.shape[0]
    cont = enc.contamination(Xbox.lo) if enc.cattn else None
    if cont is not None:                                               # contamination itself may change: bound it
        c2 = enc.contamination(Xbox.hi)
        cont = Box(torch.minimum(cont, c2), torch.maximum(cont, c2))
    h = Xbox.linear(enc.proj)
    if enc.state == "carry":
        raise NotImplementedError("use footprint_all for carried state")
    toks = []
    for s in enc.starts:
        p = Box(h.lo[:, s:s + enc.patch], h.hi[:, s:s + enc.patch])
        hf = lstm_dir(enc.lstm, p, reverse=False)
        hb = lstm_dir(enc.lstm, p, reverse=True)
        toks.append(box_cat([hf, hb], -1))
    hn = Box(torch.stack([t.lo for t in toks], 1), torch.stack([t.hi for t in toks], 1))
    z = hn.linear(enc.tok)
    pos = torch.arange(enc.K, device=z.lo.device, dtype=torch.float32) if enc.use_rope else None
    bias = None
    if cont is not None:
        k = enc.cattn_logb.exp()
        bias_box = Box(-k * cont.hi, -k * cont.lo)
        bias = bias_box.c                                              # bias interval handled via score box below
        raise NotImplementedError("contamination-biased models: bound the bias jointly (todo)")
    for blk in enc.blocks:
        z = block_box(blk, z, pos, bias)
    zp = Box(z.lo.mean(1), z.hi.mean(1))
    lg = zp.linear(model.head.out[1]).lo.view(B, 4, 3), zp.linear(model.head.out[1]).hi.view(B, 4, 3)
    return cif_bounds(*lg)


def cif_bounds(llo, lhi):
    """interval logits [B,J,3] (death, discharge, remain) -> CIF interval [B,J]; last interval has no 'remain'."""
    llo, lhi = llo.clone(), lhi.clone()
    llo[:, -1, 2] = -1e4
    lhi[:, -1, 2] = -1e4
    def p_bounds(k):
        others = [i for i in range(3) if i != k]
        lo = torch.exp(llo[..., k] - torch.logsumexp(torch.stack([llo[..., k]] + [lhi[..., i] for i in others], -1), -1))
        hi = torch.exp(lhi[..., k] - torch.logsumexp(torch.stack([lhi[..., k]] + [llo[..., i] for i in others], -1), -1))
        return lo, hi
    pd_lo, pd_hi = p_bounds(0)
    pr_lo, pr_hi = p_bounds(2)
    ones = torch.ones_like(pr_lo[:, :1])
    s_lo = torch.cumprod(torch.cat([ones, pr_lo[:, :-1]], 1), 1)
    s_hi = torch.cumprod(torch.cat([ones, pr_hi[:, :-1]], 1), 1)
    return torch.cumsum(s_lo * pd_lo, 1), torch.cumsum(s_hi * pd_hi, 1)


# ---------------------------------------------------------------- threat sets -> input boxes
SPARSE = [c for c in range(len(RG.VARS)) if c not in RG.VIT]


def window_threat(x, m, t0, t1, variables=SPARSE):
    """all measurements of `variables` with hour in [t0, t1) may each be missing. Returns raw-scale (lo, hi) of the
    filled values and of the mask / availability channels after re-imputation, by enumerating per cell the values any
    subset deletion can yield: the original value, any earlier observation still inside the carry window, or absent."""
    N, Tn, C = x.shape
    raw = np.where(m > 0, x, np.nan)
    vlo, vhi = x.copy(), x.copy()
    mlo, mhi = m.copy(), m.copy()
    absent_possible = np.zeros_like(m, dtype=bool)
    for c in variables:
        w = RG.CARRY[RG.VARS[c]]
        del_obs = (m[:, :, c] > 0) & (np.arange(Tn)[None] >= t0) & (np.arange(Tn)[None] < t1)
        if not del_obs.any():
            continue
        mlo[:, :, c] = np.where(del_obs, 0, m[:, :, c])
        for t in range(Tn):
            # values reachable at hour t: original filled value, or any observation at hour s in [t-w, t] that is kept
            cand_lo, cand_hi = vlo[:, t, c].copy(), vhi[:, t, c].copy()
            can_change = np.zeros(N, bool)
            for s in range(max(0, t - w), t + 1):
                dels = del_obs[:, s]
                can_change |= dels
                v = raw[:, s, c]
                ok = ~np.isnan(v)
                cand_lo = np.where(ok, np.fmin(cand_lo, v), cand_lo)
                cand_hi = np.where(ok, np.fmax(cand_hi, v), cand_hi)
            if w == 0:
                can_change = del_obs[:, t]
            vlo[:, t, c] = np.where(can_change, cand_lo, vlo[:, t, c])
            vhi[:, t, c] = np.where(can_change, cand_hi, vhi[:, t, c])
            absent_possible[:, t, c] = can_change
    return vlo, vhi, mlo, mhi, absent_possible


def input_box(vlo, vhi, mlo, mhi, absent, stats, avail):
    """raw-scale value hull + mask bounds -> network input Box, mirroring train.make_input exactly:
    standardised value (NaN -> 0), first difference (0 whenever either neighbour is missing), mask, availability."""
    mu, sd = stats
    zl = np.clip((vlo - mu) / sd, -10, 10)
    zh = np.clip((vhi - mu) / sd, -10, 10)
    nan = np.isnan(zl)
    posabs = absent | nan                                             # value may be missing (-> 0)
    v_lo, v_hi = np.nan_to_num(zl, nan=0.0), np.nan_to_num(zh, nan=0.0)
    v_lo = np.where(posabs, np.minimum(v_lo, 0.0), v_lo)
    v_hi = np.where(posabs, np.maximum(v_hi, 0.0), v_hi)
    d_lo = np.zeros_like(v_lo)
    d_hi = np.zeros_like(v_hi)
    with np.errstate(invalid="ignore"):
        a_lo = zl[:, 1:] - zh[:, :-1]
        a_hi = zh[:, 1:] - zl[:, :-1]
    a_lo = np.nan_to_num(a_lo, nan=0.0)
    a_hi = np.nan_to_num(a_hi, nan=0.0)
    either = posabs[:, 1:] | posabs[:, :-1]
    d_lo[:, 1:] = np.where(either, np.minimum(a_lo, 0.0), a_lo)
    d_hi[:, 1:] = np.where(either, np.maximum(a_hi, 0.0), a_hi)
    both_nan = nan[:, 1:] | nan[:, :-1]                               # certain missing neighbour -> exactly 0
    certain = ~(absent[:, 1:] | absent[:, :-1])
    d_lo[:, 1:] = np.where(both_nan & certain, 0.0, d_lo[:, 1:])
    d_hi[:, 1:] = np.where(both_nan & certain, 0.0, d_hi[:, 1:])
    lo = [v_lo, d_lo, mlo]
    hi = [v_hi, d_hi, mhi]
    if avail:
        av = (~nan).astype(np.float32)
        lo.append(np.where(absent, 0.0, av))
        hi.append(av)
    return Box(torch.tensor(np.concatenate(lo, -1), dtype=torch.float32), torch.tensor(np.concatenate(hi, -1), dtype=torch.float32))


# ---------------------------------------------------------------- bound-friendly (CertBlock) models, differentiable
def lstm_seq(lstm, x, reverse):
    """all per-step hidden Boxes of one LSTM direction over x Box [N,T,d]; returns list indexed by time."""
    sfx = "_reverse" if reverse else ""
    Wih, Whh = getattr(lstm, "weight_ih_l0" + sfx), getattr(lstm, "weight_hh_l0" + sfx)
    b = getattr(lstm, "bias_ih_l0" + sfx) + getattr(lstm, "bias_hh_l0" + sfx)
    N, Tn, _ = x.lo.shape
    H = Whh.shape[1]
    h, c = Box(x.lo.new_zeros(N, H)), Box(x.lo.new_zeros(N, H))
    out = [None] * Tn
    for t in (range(Tn - 1, -1, -1) if reverse else range(Tn)):
        gc = F.linear(x.c[:, t], Wih) + F.linear(h.c, Whh) + b
        gr = F.linear(x.r[:, t], Wih.abs()) + F.linear(h.r, Whh.abs())
        g = Box(gc - gr, gc + gr)
        i, f, gg, o = [g.slice(slice(None), slice(k * H, (k + 1) * H)) for k in range(4)]
        i, f, o, gg = i.mono(torch.sigmoid), f.mono(torch.sigmoid), o.mono(torch.sigmoid), gg.mono(torch.tanh)
        c = f.mul(c) + i.mul(gg)
        h = o.mul(c.mono(torch.tanh))
        out[t] = h
    return out


def l2norm_box(b):
    """per-row (last dim) L2 normalisation of an interval vector; exact for points."""
    lo, hi = b.lo, b.hi
    dist = torch.where((lo <= 0) & (hi >= 0), torch.zeros_like(lo), torch.minimum(lo.abs(), hi.abs()))
    nmin = dist.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
    nmax = torch.maximum(lo.abs(), hi.abs()).pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
    ylo = torch.where(lo >= 0, lo / nmax, lo / nmin).clamp(-1, 1)
    yhi = torch.where(hi >= 0, hi / nmin, hi / nmax).clamp(-1, 1)
    pt = (hi - lo).abs().amax(-1, keepdim=True) < 1e-6                 # float-noise tolerance
    ex = F.normalize((lo + hi) / 2, dim=-1)
    return Box(torch.where(pt, ex, ylo), torch.where(pt, ex, yhi))


def cert_block_box(blk, z, pos):
    B, L, D = z.lo.shape
    qkv = z.linear(blk.qkv)
    sh = lambda t: t.view(B, L, 3, blk.h, blk.dh).permute(2, 0, 3, 1, 4)
    lo, hi = sh(qkv.lo), sh(qkv.hi)
    q, k, v = Box(lo[0], hi[0]), Box(lo[1], hi[1]), Box(lo[2], hi[2])
    if pos is not None:
        q, k = rope_box(q, pos), rope_box(k, pos)
    q, k = l2norm_box(q), l2norm_box(k)
    qc, qr, kc, kr = q.c, q.r, k.c, k.r
    sc = qc @ kc.transpose(-1, -2)
    sr = qr @ kc.abs().transpose(-1, -2) + qc.abs() @ kr.transpose(-1, -2) + qr @ kr.transpose(-1, -2)
    S = Box(((sc - sr) * blk.tau).clamp(-blk.tau, blk.tau), ((sc + sr) * blk.tau).clamp(-blk.tau, blk.tau))
    Lk = S.lo.shape[-1]
    eye = torch.eye(Lk, dtype=torch.bool, device=S.lo.device)
    A = torch.where(eye, S.lo.unsqueeze(-1), S.hi.unsqueeze(-2))
    Bm = torch.where(eye, S.hi.unsqueeze(-1), S.lo.unsqueeze(-2))
    wlo = torch.exp(S.lo - torch.logsumexp(A, -1))
    whi = torch.exp(S.hi - torch.logsumexp(Bm, -1))
    vlo, vhi = v.lo.unsqueeze(-3), v.hi.unsqueeze(-3)
    wl, wh = wlo.unsqueeze(-1), whi.unsqueeze(-1)
    y = Box(_weighted_extreme(wl, wh, vlo, True), _weighted_extreme(wl, wh, vhi, False))
    y = Box(y.lo.transpose(1, 2).reshape(B, L, D), y.hi.transpose(1, 2).reshape(B, L, D))
    z = z + y.linear(blk.o)
    f = z.linear(blk.ff1).mono(F.relu).linear(blk.ff2)
    return z + f


def cert_bounds(model, xb, return_logits=False):
    """differentiable CIF bounds for a CertBlock HorizonPF (queries=False), reset or carried state."""
    enc = model.enc
    assert enc.cert
    B = xb.lo.shape[0]
    h = xb.linear(enc.proj)
    if enc.state == "carry":
        fw, bw = lstm_seq(enc.lstm, h, False), lstm_seq(enc.lstm, h, True)
        toks = [box_cat([fw[s + enc.patch - 1], bw[s]], -1) for s in enc.starts]
    else:
        toks = []
        for s in enc.starts:
            p = Box(h.lo[:, s:s + enc.patch], h.hi[:, s:s + enc.patch])
            toks.append(box_cat([lstm_dir(enc.lstm, p, False), lstm_dir(enc.lstm, p, True)], -1))
    hn = Box(torch.stack([t.lo for t in toks], 1), torch.stack([t.hi for t in toks], 1))
    z = hn.linear(enc.tok)
    pos = torch.arange(enc.K, device=z.lo.device, dtype=torch.float32) if enc.use_rope else None
    for blk in enc.blocks:
        z = cert_block_box(blk, z, pos)
    zp = Box(z.lo.mean(1), z.hi.mean(1)).linear(model.head.out[1])
    lo, hi = cif_bounds(zp.lo.view(B, 4, 3), zp.hi.view(B, 4, 3))
    lo, hi = lo.clamp(0, 1), hi.clamp(0, 1)                            # CIF is a probability
    return (lo, hi, zp) if return_logits else (lo, hi)
