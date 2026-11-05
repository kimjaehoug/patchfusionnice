"""Models: PatchFusion (manuscript spec), baselines, and horizon-coherent extensions."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_HORIZON = 4  # (12,24], (24,48], (48,168], (168, discharge]


# ---------------------------------------------------------------- RoPE attention
def rope(x, pos):
    """x [B,H,L,Dh], pos [L] float -> rotated x."""
    dh = x.shape[-1]
    half = dh // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=x.dtype) / half))
    ang = pos[:, None].to(x.dtype) * inv[None]          # [L, half]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class RoPEAttention(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.drop = dropout

    def forward(self, x, pos=None, bias=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if pos is not None:
            q, k = rope(q, pos), rope(k, pos)
        mask = None if bias is None else bias[:, None, None, :].expand(B, self.h, L, L).to(q.dtype)   # key bias
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    """pre-norm transformer layer; same parameter count as nn.TransformerEncoderLayer(norm_first=True)."""

    def __init__(self, d, h, ff, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = RoPEAttention(d, h, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, pos=None, bias=None):
        x = x + self.d1(self.att(self.n1(x), pos, bias))
        return x + self.d2(self.ff(self.n2(x)))


LAB_IDX = list(range(6, 16))                 # bun..ag (selective-testing laboratory variables)
CAP_IDX = [16, 20, 21]                       # ne, uo, gcs (capture / documentation gaps)
SRC_IDX = list(range(6, 22))                 # labs, ne, vaso, fio2, vent, uo, gcs (may be absent from a whole source)
CARRY_POS = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0], dtype=torch.bool)   # carry>0 per var


def counterfactual_slice(p, regime, n=22):
    """p: [N,L,88] (standardised value, delta, raw-obs mask, availability). Returns the slice re-expressed under a
    counterfactual observation process, re-imputed inside the slice only:
      'lab' - each laboratory variable keeps only its last observation in the slice (lower testing intensity);
      'cap' - all GCS / urine output / norepinephrine observations are missing (documentation gap);
      'src' - each non-vital channel is absent for the whole stay with probability 0.5 (variable not recorded at the site).
    Values carried into the slice from before its start are left untouched (they are not in-slice observations)."""
    if regime == "src":                                   # source-level absence: channels never recorded at this site
        drop = torch.rand(p.shape[0], len(SRC_IDX), device=p.device) < 0.5
        keep = torch.ones(p.shape[0], n, device=p.device)
        keep[:, SRC_IDX] = (~drop).float()
        keep = keep[:, None, :].repeat(1, 1, 4)                # same channel in value / delta / mask / availability
        return p * keep
    v, dv, m, av = p[..., :n], p[..., n:2 * n], p[..., 2 * n:3 * n], p[..., 3 * n:4 * n]
    G = LAB_IDX if regime == "lab" else CAP_IDX
    vg, mg, ag = v[..., G], m[..., G] > 0, av[..., G] > 0
    L = p.shape[1]
    if regime == "lab":
        later = torch.flip(torch.cumsum(torch.flip(mg.float(), [1]), 1), [1]) - mg.float()   # obs strictly after t
        keep = mg & (later == 0)
    else:
        keep = torch.zeros_like(mg)
    carry = CARRY_POS.to(p.device)[G]
    vs, as_, ms = torch.zeros_like(vg), torch.zeros_like(ag), keep.clone()
    for t in range(L):
        kept = mg[:, t] & keep[:, t]
        carried_in = (~mg[:, t]) & ag[:, t] & (t == 0)
        refill = (mg[:, t] & ~keep[:, t]) | ((~mg[:, t]) & ag[:, t] & (t > 0))
        prev_ok = as_[:, t - 1] & carry if t > 0 else torch.zeros_like(kept)
        vt = torch.where(kept | carried_in, vg[:, t], torch.zeros_like(vg[:, t]))
        at = kept | carried_in
        vt = torch.where(refill & prev_ok, vs[:, t - 1] if t > 0 else vt, vt)
        at = at | (refill & prev_ok)
        vs[:, t], as_[:, t] = vt, at
    dvs = torch.zeros_like(vs)
    dvs[:, 1:] = torch.where(as_[:, 1:] & as_[:, :-1], vs[:, 1:] - vs[:, :-1], torch.zeros_like(vs[:, 1:]))
    same0 = (as_[:, 0] == ag[:, 0]) & (vs[:, 0] == vg[:, 0])
    dvs[:, 0] = torch.where(same0, dv[:, 0][..., G], torch.zeros_like(dvs[:, 0]))
    out = p.clone()
    out[..., G], out[..., [n + g for g in G]] = vs, dvs
    out[..., [2 * n + g for g in G]], out[..., [3 * n + g for g in G]] = ms.float(), as_.float()
    return out


class CertBlock(nn.Module):
    """bound-friendly global attention layer: per-head L2-normalised queries/keys (after RoPE) with a fixed temperature,
    so every attention score lies in [-tau, tau]; no LayerNorm; ReLU feed-forward."""

    def __init__(self, d, h, ff, tau=4.0):
        super().__init__()
        self.h, self.dh, self.tau = h, d // h, tau
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x, pos=None, bias=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if pos is not None:
            q, k = rope(q, pos), rope(k, pos)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        s = self.tau * (q @ k.transpose(-1, -2))
        if bias is not None:
            s = s + bias[:, None, None, :]
        y = torch.softmax(s, -1) @ v
        x = x + self.o(y.transpose(1, 2).reshape(B, L, D))
        return x + self.ff2(F.relu(self.ff1(x)))


# ---------------------------------------------------------------- PatchFusion encoder
class DynReset(nn.Module):
    """Observation-dynamics-gated continuous reset (dynamic patching).

    A detector reads, at every hour, the size of physiological change (|dv|) and of changes in the observation process
    (|delta measured-mask|, |delta availability|) and emits s_t in (0,1). Branch b resets the bidirectional recurrent state
    with strength r_t = kappa_b * s_t:  state <- (1 - r_t) * state  (toward the zero initial state).
    r = 0 everywhere is one uninterrupted sequence (no patching); r = 1 at fixed hours is hard patching. Two branches with
    learnable kappa (init 0.2 short-horizon, 0.8 long-horizon) give two patching degrees. Tokens every `stride` hours:
    forward state at the block end, backward state at the block start.
    """

    def __init__(self, d, H, n_val, stride, T, kappa=(0.2, 0.8), mode="learn", r_fix=None, penc="lstm"):
        super().__init__()
        # patch-encoder variants (hard mode): the within-patch recurrence is replaced by a non-recurrent map of the same
        # receptive field (forward: segment start..block end, backward: block start..segment end), or hours are shuffled
        self.penc = penc
        if penc in ("cnn", "mlp", "linear", "mean"):
            def mk():
                if penc == "cnn":
                    return nn.Conv1d(d, H, 3, padding=1)
                if penc == "mlp":
                    return nn.Sequential(nn.Linear(6 * d, H), nn.GELU(), nn.Linear(H, H))
                if penc == "linear":
                    return nn.Linear(6 * d, H)
                return nn.Linear(d, H)
            self.pf_f, self.pf_b = mk(), mk()
        self.r_fix = r_fix                                    # dial: fixed hourly reset strength for both branches
        self.mode = mode                                      # learn | const (s = 0.5 everywhere: falsification) | boost
        if penc not in ("cnn", "mlp", "linear", "mean"):
            self.cf, self.cb = nn.LSTMCell(d, H), nn.LSTMCell(d, H)
        self.det = nn.Sequential(nn.Linear(3 * n_val, 32), nn.GELU(), nn.Linear(32, 1))
        self.kl = nn.Parameter(torch.logit(torch.tensor(kappa)))
        # hard / handoff: reset every `seg` hours; handoff passes a bottlenecked summary of the finished segment as the next
        # segment's initial state (observation invariance of that summary is enforced in the two-view loss, inv_loss)
        self.seg, self.inv_loss = 6, None
        if mode == "handoff":
            nb = 32
            self.hb_f, self.hb_b = nn.Linear(H, nb), nn.Linear(H, nb)
            self.hi_f, self.hi_b = nn.Linear(nb, 2 * H), nn.Linear(nb, 2 * H)
        if mode == "boost":                                   # wider initial detector response (lr x10 set in train.py)
            nn.init.normal_(self.det[-1].weight, std=0.5)
        self.n, self.H = n_val, H
        self.ends = list(range(stride - 1, T, stride))
        self.starts = [e - stride + 1 for e in self.ends]
        self._s = None

    def _ff_tokens(self, h):
        B, T, D = h.shape
        seg = min(self.seg, T)
        fi, bi = [], []
        for s_, e in zip(self.starts, self.ends):
            S = (s_ // seg) * seg
            E = min(S + seg, T) - 1
            f = list(range(S, e + 1))
            b = list(range(s_, E + 1))
            fi.append([-1] * (seg - len(f)) + f)              # left-padded: forward context ends at the block end
            bi.append(b + [-1] * (seg - len(b)))              # right-padded: backward context starts at the block start
        out = []
        for idx, mod in ((fi, self.pf_f), (bi, self.pf_b)):
            ix = torch.tensor(idx, device=h.device)            # [K,seg]
            mk = (ix >= 0).float()[None, :, :, None]           # [1,K,seg,1]
            g = h[:, ix.clamp(min=0)] * mk                     # [B,K,seg,D]
            K = g.shape[1]
            if self.penc == "mean":
                o = mod(g.sum(2) / mk.sum(2))
            elif self.penc == "cnn":
                c = F.gelu(mod(g.reshape(B * K, seg, D).transpose(1, 2))).view(B, K, -1, seg)
                o = (c - (1 - mk[:, :, :, 0])[:, :, None] * 1e4).max(-1).values
            else:
                o = mod(g.reshape(B, K, seg * D))
            out.append(o)
        return torch.cat(out, -1)

    def _segments(self, h, x):
        B, T, H, seg = h.shape[0], h.shape[1], self.H, self.seg
        if self.penc in ("cnn", "mlp", "linear", "mean"):
            return self._ff_tokens(h)
        if self.penc == "shuffle":                            # order inside each reset patch destroyed
            gen = None if self.training else torch.Generator(device="cpu").manual_seed(0)
            perm = torch.cat([S + torch.randperm(min(seg, T - S), generator=gen) for S in range(0, T, seg)]).to(h.device)
            h = h[:, perm]
        ho = self.mode == "handoff"
        if not ho and self.penc == "lstm" and (getattr(self, "stagger", False) or hasattr(self, "h0")):
            tk = self._grid(h, 0, x)
            if getattr(self, "stagger", False):
                tk = 0.5 * (tk + self._grid(h, seg // 2, x))
            self.inv_loss = None
            return tk if getattr(self, "clean", False) else torch.cat([tk, tk], 1)
        z = h.new_zeros(B, H)
        hf, cf, of, bf = z, z, [], []
        for t in range(T):
            if t > 0 and t % seg == 0:
                if ho:
                    b = self.hb_f(hf)
                    bf.append(b)
                    hf, cf = torch.tanh(self.hi_f(b)).chunk(2, -1)
                else:
                    hf, cf = z, z
            hf, cf = self.cf(h[:, t], (hf, cf))
            of.append(hf)
        hb, cb, ob = z, z, [None] * T
        for t in range(T - 1, -1, -1):
            hb, cb = self.cb(h[:, t], (hb, cb))
            ob[t] = hb
            if t > 0 and t % seg == 0:
                if ho:
                    b = self.hb_b(hb)
                    bf.append(b)
                    hb, cb = torch.tanh(self.hi_b(b)).chunk(2, -1)
                else:
                    hb, cb = z, z
        self.inv_loss = None
        if ho and self.training and self.twoview:             # factual first half, counterfactual second half
            bb = torch.stack(bf, 1)                           # [B, n_boundaries, nb]
            h2 = B // 2
            fa, cfv = bb[:h2].detach(), bb[h2:]
            self.inv_loss = ((cfv - fa) ** 2).mean() / (fa.var() + 1e-4)
        tk = torch.stack([torch.cat([of[e], ob[s_]], -1) for s_, e in zip(self.starts, self.ends)], 1)
        if getattr(self, "clean", False):
            return tk
        return torch.cat([tk, tk], 1)                         # same token set for both head branches

    def _grid(self, h, off, x=None):
        """hard reset at hours t with (t - off) % seg == 0; tokens: forward state at block end, backward at block start."""
        B, T, seg = h.shape[0], h.shape[1], self.seg
        init = [h.new_zeros(B, self.H)] * 4 if not hasattr(self, "h0") else [p[None].expand(B, -1) for p in self.h0]
        if hasattr(self, "h0_net") and x is not None:      # absence-conditioned start: which variables exist in the window
            ex = (x[..., 3 * self.n:4 * self.n].amax(1) > 0).float()        # [B, n] stay-level, unchanged by thinning
            init = [i + z for i, z in zip(init, torch.tanh(self.h0_net(ex)).chunk(4, -1))]
        hf, cf = init[0], init[1]
        of = []
        for t in range(T):
            if t > 0 and (t - off) % seg == 0:
                hf, cf = init[0], init[1]
            hf, cf = self.cf(h[:, t], (hf, cf))
            of.append(hf)
        hb, cb, ob = init[2], init[3], [None] * T
        for t in range(T - 1, -1, -1):
            hb, cb = self.cb(h[:, t], (hb, cb))
            ob[t] = hb
            if t > 0 and (t - off) % seg == 0:
                hb, cb = init[2], init[3]
        return torch.stack([torch.cat([of[e], ob[s_]], -1) for s_, e in zip(self.starts, self.ends)], 1)

    def forward(self, h, x):
        if self.mode in ("hard", "handoff"):
            return self._segments(h, x)
        n, B, T = self.n, h.shape[0], h.shape[1]
        dv, m, av = x[..., n:2 * n], x[..., 2 * n:3 * n], x[..., 3 * n:4 * n]
        dm = torch.cat([torch.zeros_like(m[:, :1]), (m[:, 1:] - m[:, :-1]).abs()], 1)
        da = torch.cat([torch.zeros_like(av[:, :1]), (av[:, 1:] - av[:, :-1]).abs()], 1)
        s = torch.sigmoid(self.det(torch.cat([dv.abs(), dm, da], -1))).squeeze(-1)          # [B,T]
        if self.mode == "const":
            s = torch.full_like(s, 0.5)
        kap = torch.sigmoid(self.kl)
        self._s, self._kappa = s.detach(), kap.detach()
        hh = torch.cat([h, h])                                                              # branch-major [2B,T,d]
        r = torch.cat([kap[0] * s, kap[1] * s])[..., None]                                  # [2B,T,1]
        if self.r_fix is not None:
            r = torch.full_like(r, float(self.r_fix))
        z = hh.new_zeros(2 * B, self.H)
        hf, cf, of = z, z, []
        for t in range(T):                                    # boundary at t: reset before consuming hour t
            hf, cf = self.cf(hh[:, t], ((1 - r[:, t]) * hf, (1 - r[:, t]) * cf))
            of.append(hf)
        hb, cb, ob = z, z, [None] * T
        for t in range(T - 1, -1, -1):                        # segment starting at t ends the backward run
            hb, cb = self.cb(hh[:, t], (hb, cb))
            ob[t] = hb
            hb, cb = (1 - r[:, t]) * hb, (1 - r[:, t]) * cb
        tk = torch.stack([torch.cat([of[e], ob[s_]], -1) for s_, e in zip(self.starts, self.ends)], 1)   # [2B,K,2H]
        return torch.cat([tk[:B], tk[B:]], 1)                                               # [B,2K,2H]


class PatchFusionEncoder(nn.Module):
    """[B,12,66] -> trajectory tokens [B,K,256] after RoPE global attention (manuscript Table S4b)."""

    def __init__(self, in_dim=66, d=256, lstm_h=189, patch=6, stride=3, T=12, layers=2, heads=8, ff=512,
                 dropout=0.2, bidir=True, use_rope=True, use_global=True, state="reset", windows=None,
                 ple_bins=0, n_val=22, ta_cuts=0, ta_keep=False, ta_q=(0.02, 0.1, 0.9, 0.98), ctx_dim=0,
                 streams=None, stream_h=109, stream_d=128, obsb=False, obsb_exc=True, obsb_ind_in_enc=False,
                 cattn=0.0, cattn_learn=False, cattn_mask_off=44, cert=False, cert_tau=4.0,
                 cfo=(), cfo_aug=0.0, cfo_gaug=0.0, cfo_aug_reg=("lab", "cap"), cfo_gaug_reg=("lab", "cap"),
                 imagine=0.0, imag2=0.0, imag_m=0.99, imag_ctx=False, imag_stay=False, imag_wpool=False, imag_wshared=False, pgate=False, twoview=False, dynpatch=False, dyn_mode="learn", dyn_r=None, inv_w=0.0, dyn_seg=6, pimag="", pimag_target="token", pimag_sg=False, clean=False, dyn_penc="lstm", dyn_pre_attn=False, dyn_stagger=False, dyn_h0=False):
        """state='reset': recurrent state restarts in every patch (manuscript).
        state='carry': one BiLSTM over the whole 12 h; token k = forward state at the end of patch k and
                       backward state at its start (same parameters and token count, state not confined).
        windows=[(s, e), ...]: landmark-anchored patches given explicitly (e.g. nested (9,12),(6,12),(0,12));
                       each window is encoded with its own reset recurrent state, as in 'reset'."""
        super().__init__()
        self.state = state
        self.patch, self.stride = patch, stride
        self.starts = list(range(0, T - patch + 1, stride))
        self.windows = [tuple(w) for w in windows] if windows else None
        self.K = len(self.windows) if self.windows else len(self.starts)
        self.ple_bins, self.n_val = ple_bins, n_val
        if ple_bins:                                          # piecewise-linear encoding of the value channels
            self.register_buffer("edges", torch.linspace(-2, 2, ple_bins + 1).repeat(n_val, 1))
            in_dim = in_dim - n_val + n_val * ple_bins
        self.ta_cuts, self.ta_keep, self.ta_q = ta_cuts, ta_keep, tuple(ta_q)
        if ta_cuts:                                           # learned temporal abstraction (needs avail channels 66:88)
            self.ta_c0 = nn.Parameter(torch.full((n_val, 1), -2.0))
            self.ta_dc = nn.Parameter(torch.zeros(n_val, ta_cuts - 1))
            self.ta_logk = nn.Parameter(torch.full((n_val, ta_cuts), math.log(4.0)))
            in_dim = n_val * (ta_cuts + 2) + 2 * n_val + (n_val if ta_keep else 0)   # states(+unavailable) + dv + mask (+ raw values)
        self.obsb, self.obsb_exc, self.obsb_ind_in_enc = obsb, obsb_exc, obsb_ind_in_enc
        if obsb:                                              # observation-process-bounded input (needs avail channels 66:88)
            self.register_buffer("anc_lo", torch.full((n_val,), -1e4))
            self.register_buffer("anc_hi", torch.full((n_val,), 1e4))
            in_dim = n_val * (2 + (2 if obsb_exc else 0) + (2 if obsb_ind_in_enc else 0))
            self._ind = None
        self.streams = [list(g) for g in streams] if streams else None
        if self.streams:                                      # observation-process confinement: one reset BiLSTM per stream
            assert not (ple_bins or ta_cuts), "streams use raw channel blocks"
            n_blk = in_dim // n_val                           # v, dv, m (, avail)
            self.s_idx = [[b * n_val + c for b in range(n_blk) for c in g] for g in self.streams]
            self.s_proj = nn.ModuleList([nn.Linear(len(ix), stream_d) for ix in self.s_idx])
            self.s_lstm = nn.ModuleList([nn.LSTM(stream_d, stream_h, batch_first=True, bidirectional=bidir) for _ in self.streams])
            self.s_tok = nn.ModuleList([nn.Linear(stream_h * (2 if bidir else 1), d) for _ in self.streams])
            self.s_emb = nn.Parameter(torch.zeros(len(self.streams), d))
        # contamination-biased global attention: patch tokens are reset-confined, so the share of non-fresh evidence
        # inside each token's receptive field is exactly computable from the observation masks (no learning)
        self.cattn, self.cattn_mask_off = float(cattn), cattn_mask_off
        if cattn:
            self.cattn_logb = nn.Parameter(torch.tensor(math.log(cattn)), requires_grad=cattn_learn)
        self._pool_w = None
        self.ctx_dim = ctx_dim
        if ctx_dim:                                           # site observation-regime descriptor -> one context token
            self.ctx = nn.Sequential(nn.Linear(ctx_dim, d), nn.GELU(), nn.Linear(d, d))
        self.proj = nn.Linear(in_dim, d)
        self.lstm = nn.LSTM(d, lstm_h, batch_first=True, bidirectional=bidir)
        self.tok = nn.Linear(lstm_h * (2 if bidir else 1), d)
        # counterfactual observation patch encoding: every reset-confined patch is also encoded under counterfactual
        # observation processes (simulated inside the slice); factual and counterfactual tokens share global attention
        self.cfo, self.cfo_aug, self.cfo_gaug = list(cfo), float(cfo_aug), float(cfo_gaug)
        self.cfo_aug_reg, self.cfo_gaug_reg = list(cfo_aug_reg), list(cfo_gaug_reg)
        # latent imagination: when a patch is seen under a thinned observation process, its token is trained to recover the
        # (stop-gradient) token of the same patch under the full development observation process
        self.imagine = float(imagine)
        self.imag_loss = None
        # imagination v2: separate probabilistic head predicts the EMA-teacher token of the same patch under the full
        # observation process; the imagined (mean, log-variance) enters global attention as an additional token
        self.imag2, self.imag_m, self.imag_scale = float(imag2), float(imag_m), 1.0
        if imag2:
            self.imag_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * d))
            self.imag_proj = nn.Linear(2 * d, d)
            self.imag_type = nn.Parameter(torch.zeros(d))
        # context-grounded patch imagination: each patch's full-observation token is imagined from the patch token after
        # one global self-attention layer over all reset-confined patch tokens (capture gaps are stay-global, so the
        # evidence for imagining a patch comes from the whole stay while the target stays patch-specific)
        self.imag_ctx = bool(imag_ctx)
        # stay-level imagination: one token imagines the EMA teacher's stay representation (global attention over its patch
        # tokens, mean-pooled) under the full observation process, from the student's patch tokens after one global layer
        self.imag_stay = bool(imag_stay)
        if imag2 and imag_stay:
            self.imag_ctx_blk = Block(d, heads, ff, dropout)
        # adaptive patch weighting: the stay-level imagination draws on patches with input-dependent softmax weights
        self.imag_wpool = bool(imag_wpool)
        if imag2 and imag_stay and imag_wpool:
            self.imag_score = nn.Linear(d, 1)
        self._imag_w = None
        # shared adaptive patch weights: the same input-dependent weights pool the trajectory tokens for prediction (so they
        # are learned from the outcome) and define the teacher's full-observation stay summary that is imagined
        self.imag_wshared = bool(imag_wshared)
        # observation-gated patch tokens: from each reset-confined patch's own observation status (share of measured and of
        # available cells per variable) a gate moves the patch token toward a version corrected by the imagined stay token;
        # well-observed patches keep their factual token, poorly observed ones lean on the imagination
        self.pgate = bool(pgate)
        if imag2 and imag_stay and pgate:
            self.gate_net = nn.Sequential(nn.Linear(2 * n_val, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.constant_(self.gate_net[-1].bias, -2.0)
            self.gate_mix = nn.Linear(2 * d, d)
        self._gate = None
        # two-view training: the batch arrives duplicated; first half factual, second half counterfactual
        self.twoview = bool(twoview)
        # dynamic patching: observation-dynamics-gated continuous reset, two patching degrees (see DynReset)
        self.dynpatch = bool(dynpatch)
        if dynpatch:
            self.dyn = DynReset(d, lstm_h, n_val, stride, T, mode=dyn_mode, r_fix=dyn_r, penc=dyn_penc)
            self.pre_attn = bool(dyn_pre_attn)               # reversal: global attention over hours before the patches
            self.dyn.twoview = bool(twoview)
            self.dyn.seg = int(dyn_seg)                       # hard-reset segment length (>= T: no reset)
            self.Kd = len(self.dyn.ends)
            self.dyn.clean = self.clean = bool(clean)       # clean: K tokens, no duplicate branch, mean pooling
            self.dyn.stagger = bool(dyn_stagger)             # second reset grid offset by seg/2; tokens averaged
            if dyn_h0:                                        # learned (not zero) initial state of every reset patch
                self.dyn.h0 = nn.Parameter(torch.zeros(4, lstm_h))
            if dyn_h0 == "avail":                             # + start conditioned on the window's variable existence
                self.dyn.h0_net = nn.Linear(n_val, 4 * lstm_h)
                nn.init.zeros_(self.dyn.h0_net.weight); nn.init.zeros_(self.dyn.h0_net.bias)
        # scale-matched restoration: patch-level probabilistic restoration of each reset-confined patch token under the
        # factual development observation process, supervised only where the counterfactual acted at the hour scale
        # (pimag="lab": selective-testing thinning) or for every counterfactual (pimag="all", control)
        self.pimag, self._reg = pimag, None
        # pimag_target="values": restore the standardised laboratory values measured in each 3-h block (physiology, not the
        # observation process); pimag_sg: stop the restoration gradient at the factual token (protect the shared encoder)
        self.pimag_target, self.pimag_sg = pimag_target, bool(pimag_sg)
        if pimag:
            od = len(LAB_IDX) if pimag_target == "values" else d
            self.pimag_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * od))
            self.pimag_proj = nn.Linear(2 * od, d)
            self.pimag_type = nn.Parameter(torch.zeros(d))
        if imag2 and imag_stay and imag_wshared:
            self.imag_wpool = True
            self.imag_score = nn.Linear(d, 1)
        if imag2 and imag_ctx:
            self.imag_ctx_blk = Block(d, heads, ff, dropout)
        if self.cfo:
            self.cfo_type = nn.Parameter(torch.zeros(1 + len(self.cfo), d))
        self.cert = cert
        if use_global:
            self.blocks = nn.ModuleList([CertBlock(d, heads, ff, cert_tau) if cert else Block(d, heads, ff, dropout)
                                         for _ in range(layers)])
        else:
            self.blocks = None
        self.use_rope = use_rope
        self.d = d

    def set_edges(self, x, m):
        """x [N,T,n_val] standardised values, m [N,T,n_val] raw-observation mask -> per-channel quantile edges."""
        q = torch.linspace(0, 1, self.ple_bins + 1, device=x.device)
        for c in range(self.n_val):
            v = x[..., c][m[..., c] > 0]
            v = v[torch.randperm(len(v), device=v.device)[:200000]] if len(v) else x[..., c].flatten()[:1000]
            e = torch.quantile(v.float(), q)
            e = torch.maximum(e, torch.cat([e[:1], e[:-1] + 1e-3]).cummax(0).values)   # strictly increasing
            self.edges[c] = e

    def contamination(self, x):
        """[B,T,88] -> [B,K]: within each (reset-confined) patch, the share of available values that are carried forward
        rather than freshly observed; patches with no available value (e.g. before ICU admission) count as fully contaminated."""
        n, o = self.n_val, self.cattn_mask_off
        m, av = x[..., o:o + n], x[..., o + n:o + 2 * n]
        car, avl = av * (1 - m), av
        wins = self.windows or [(s, s + self.patch) for s in self.starts]
        num = torch.stack([car[:, s:e].sum((1, 2)) for s, e in wins], 1)
        den = torch.stack([avl[:, s:e].sum((1, 2)) for s, e in wins], 1)
        return torch.where(den > 0, num / den.clamp_min(1), torch.ones_like(num))

    def set_anchors(self, lo, hi):
        self.anc_lo.copy_(torch.as_tensor(lo, dtype=self.anc_lo.dtype)); self.anc_hi.copy_(torch.as_tensor(hi, dtype=self.anc_hi.dtype))

    def _obsb_input(self, x):
        """absence-neutral values (+ one-sided excursions beyond fixed clinical critical values); measurement indicators are
        withheld from the encoder and exposed only to the bounded evidence pathway (patient-level ever-measured)."""
        n = self.n_val
        v, dv, m, av = x[..., :n], x[..., n:2 * n], x[..., 2 * n:3 * n], x[..., 3 * n:4 * n]
        parts = [v * av, dv * av]
        if self.obsb_exc:
            parts += [F.relu(self.anc_lo - v) * av, F.relu(v - self.anc_hi) * av]
        if self.obsb_ind_in_enc:
            parts += [m, av]
        self._ind = m.amax(1)                                 # [B, n_val]
        return torch.cat(parts, -1)

    def _encode_slices(self, sl):
        """sl: [N,P,in] raw patch slices -> tokens [N,d] (reset state per slice)."""
        _, (hn, _) = self.lstm(self.proj(sl))
        return self.tok(hn.transpose(0, 1).reshape(sl.shape[0], -1))

    def _cfo_forward(self, x, c):
        B = x.shape[0]
        sl = torch.stack([x[:, s:s + self.patch] for s in self.starts], 1)            # [B,K,P,in]
        flat = sl.reshape(B * self.K, self.patch, -1)
        if self.cfo_aug and self.training:                                           # augmentation-only control
            r = torch.rand(B * self.K, device=x.device)
            nr = len(self.cfo_aug_reg)
            for j, reg in enumerate(self.cfo_aug_reg):
                sel = (r >= j * self.cfo_aug / nr) & (r < (j + 1) * self.cfo_aug / nr)
                if sel.any():
                    flat = flat.clone()
                    flat[sel] = counterfactual_slice(flat[sel], reg)
        views = [flat] + [counterfactual_slice(flat, reg) for reg in self.cfo]
        z = torch.stack([self._encode_slices(v).view(B, self.K, -1) for v in views], 1)   # [B,V,K,d]
        V = z.shape[1]
        if self.cfo:
            z = z + self.cfo_type[None, :, None, :]
        z = z.reshape(B, V * self.K, -1)
        pos = torch.arange(self.K, device=x.device, dtype=torch.float32).repeat(V) if self.use_rope else None
        for b in self.blocks:
            z = b(z, pos)
        return z

    def _stream_forward(self, x, c):
        B, toks, pos = x.shape[0], [], []
        for g, ix in enumerate(self.s_idx):
            h = self.s_proj[g](x[..., ix])
            p = torch.stack([h[:, s:s + self.patch] for s in self.starts], 1)          # [B,K,P,ds]
            _, (hn, _) = self.s_lstm[g](p.reshape(B * self.K, self.patch, -1))      # state confined to (stream, patch)
            toks.append(self.s_tok[g](hn.transpose(0, 1).reshape(B, self.K, -1)) + self.s_emb[g])
            pos.append(torch.arange(self.K, device=x.device, dtype=torch.float32))
        z, pos = torch.cat(toks, 1), torch.cat(pos)
        if c is not None:
            z = torch.cat([z, self.ctx(c)[:, None]], 1)
            pos = torch.cat([pos, pos.new_tensor([self.K])])
        if self.blocks is not None:
            for b in self.blocks:
                z = b(z, pos if self.use_rope else None)
        return z

    def cuts(self):
        return torch.cat([self.ta_c0, self.ta_c0 + torch.cumsum(F.softplus(self.ta_dc) + 1e-3, 1)], 1)   # ordered [C,K]

    def set_cuts(self, x, m, q=None):
        """initialise ordered cut points at tail quantiles of observed standardised values (training split only)."""
        q = q or self.ta_q
        assert len(q) == self.ta_cuts
        with torch.no_grad():
            for c in range(self.n_val):
                v = x[..., c][m[..., c] > 0]
                v = v[torch.randperm(len(v), device=v.device)[:200000]] if len(v) > 10 else x[..., c].flatten()[:1000]
                e = torch.quantile(v.float(), torch.tensor(q, device=v.device))
                e = torch.maximum(e, torch.cat([e[:1], e[:-1] + 0.05]).cummax(0).values)
                self.ta_c0[c, 0] = e[0]
                gaps = (e[1:] - e[:-1]).clamp_min(0.05) - 1e-3
                self.ta_dc[c] = torch.log(torch.expm1(gaps))     # inverse softplus

    def abstraction(self, x):
        """[B,T,88] -> soft clinical states per variable: below c1 | c1-c2 | ... | above cK | unavailable."""
        v, rest, av = x[..., :self.n_val], x[..., self.n_val:3 * self.n_val], x[..., 3 * self.n_val:4 * self.n_val]
        c, k = self.cuts(), self.ta_logk.exp()
        s = torch.sigmoid(k * (v[..., None] - c))             # [B,T,C,K] P(value above cut k)
        ones, zeros = torch.ones_like(s[..., :1]), torch.zeros_like(s[..., :1])
        st = torch.cat([ones, s], -1) - torch.cat([s, zeros], -1)   # K+1 ordered states
        st = torch.cat([st * av[..., None], (1 - av)[..., None]], -1)
        return torch.cat(([v] if self.ta_keep else []) + [st.flatten(-2), rest], -1)

    def ple(self, x):
        v = x[..., :self.n_val, None]                         # [B,T,C,1]
        lo, hi = self.edges[:, :-1], self.edges[:, 1:]        # [C,K]
        enc = ((v - lo) / (hi - lo)).clamp(0, 1)
        enc[..., 0] = torch.where(v[..., 0] < lo[:, 0], (v[..., 0] - lo[:, 0]) / (hi[:, 0] - lo[:, 0]), enc[..., 0])
        enc[..., -1] = torch.where(v[..., 0] > hi[:, -1], (v[..., 0] - lo[:, -1]) / (hi[:, -1] - lo[:, -1]), enc[..., -1])
        return torch.cat([enc.flatten(-2), x[..., self.n_val:]], -1)

    def patches(self, h):
        return torch.stack([h[:, s:s + self.patch] for s in self.starts], 1)  # [B,K,P,d]

    def forward(self, x):
        B = x.shape[0]
        c = None
        if self.ctx_dim:
            c, x = x[:, 0, -self.ctx_dim:], x[..., :-self.ctx_dim]
        x_orig = x
        if self.cfo_gaug and self.training:                   # control: same counterfactual rules on the whole sequence
            r = torch.rand(x.shape[0], device=x.device)
            if self.twoview:                                  # factual first half, every second-half sample counterfactual
                h2 = x.shape[0] // 2
                r[:h2] = 2.0
                r[h2:] = r[h2:] * self.cfo_gaug
            nr = len(self.cfo_gaug_reg)
            self._reg = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)   # -1 factual, j regime index
            for j, reg in enumerate(self.cfo_gaug_reg):
                sel = (r >= j * self.cfo_gaug / nr) & (r < (j + 1) * self.cfo_gaug / nr)
                self._reg[sel] = j
                if sel.any():
                    x = x.clone()
                    x[sel] = counterfactual_slice(x[sel], reg)
        cont = self.contamination(x) if self.cattn else None
        if self.obsb:
            x = self._obsb_input(x)
        if self.streams:
            return self._stream_forward(x, c)
        if self.dynpatch:
            return self._dyn_forward(x, x_orig, c, B)
        if self.ta_cuts:
            x = self.abstraction(x)
        elif self.ple_bins:
            x = self.ple(x)
        h = self.proj(x)
        if self.windows:
            hs = []
            for s, e in self.windows:                         # one reset BiLSTM pass per window, all ending at e
                _, (hn, _) = self.lstm(h[:, s:e])
                hs.append(hn.transpose(0, 1).reshape(B, -1))
            hn = torch.stack(hs, 1)
        elif self.state == "carry":
            out, _ = self.lstm(h)                             # [B,T,2H] states carried across patches
            H = out.shape[-1] // 2
            ends = [s + self.patch - 1 for s in self.starts]
            hn = torch.cat([out[:, ends, :H], out[:, self.starts, H:]], -1)
        elif self.cfo or self.cfo_aug:
            return self._cfo_forward(x, c)
        else:
            p = self.patches(h)                               # [B,K,P,d]
            _, (hn, _) = self.lstm(p.reshape(B * self.K, self.patch, self.d))   # states restart per patch
            hn = hn.transpose(0, 1).reshape(B, self.K, -1)    # concat terminal fwd/bwd states
        z = self.tok(hn)
        self.imag_loss = None
        if self.imag2:
            if not hasattr(self, "_teacher") and not self.imag_stay:
                import copy
                self._teacher = [copy.deepcopy(nn.ModuleList([self.proj, self.lstm, self.tok])).requires_grad_(False)]
            if self.imag_stay:
                return self._imag_stay_forward(x, x_orig, z, c, B)
            zc = z
            if self.imag_ctx:
                zc = self.imag_ctx_blk(z, torch.arange(self.K, device=x.device, dtype=torch.float32) if self.use_rope else None)
            mu, lv = self.imag_head(zc).chunk(2, -1)
            lv = lv.clamp(-6, 6)
            if getattr(self, "_subst", None) is not None:     # test-time patch substitution: [B,K] bool
                z = torch.where(self._subst[..., None], mu, z)
            if self.training:
                tp, tl, tt = self._teacher[0]
                with torch.no_grad():
                    po = self.patches(tp(x_orig))
                    _, (ht, _) = tl(po.reshape(B * self.K, self.patch, self.d))
                    zt = tt(ht.transpose(0, 1).reshape(B, self.K, -1))
                self.imag_loss = 0.5 * (lv + (zt - mu) ** 2 / lv.exp()).mean()      # Gaussian NLL
            z = torch.cat([z, self.imag_proj(torch.cat([mu, lv], -1)) + self.imag_type], 1)
        if self.imagine and self.training and x is not x_orig:
            with torch.no_grad():                             # target tokens under the factual observation process
                ho = self.proj(x_orig)
                if self.state == "carry":
                    out, _ = self.lstm(ho)
                    H = out.shape[-1] // 2
                    ends = [s + self.patch - 1 for s in self.starts]
                    ht = torch.cat([out[:, ends, :H], out[:, self.starts, H:]], -1)
                else:
                    po = self.patches(ho)
                    _, (ht, _) = self.lstm(po.reshape(B * self.K, self.patch, self.d))
                    ht = ht.transpose(0, 1).reshape(B, self.K, -1)
                zt = self.tok(ht)
            changed = (x != x_orig).flatten(1).any(1)          # samples whose observation process was thinned
            if changed.any():
                self.imag_loss = F.mse_loss(z[changed], zt[changed])
        bias = None
        if cont is not None:
            bias = -self.cattn_logb.exp() * cont                       # [B,K]
            self._pool_w = torch.softmax(bias, 1)
        if c is not None:
            z = torch.cat([z, self.ctx(c)[:, None]], 1)       # attends with the trajectory tokens
            if bias is not None:
                bias = torch.cat([bias, bias.new_zeros(B, 1)], 1)
                self._pool_w = torch.softmax(bias, 1)
        if self.blocks is not None:
            pos = torch.arange(z.shape[1], device=x.device, dtype=torch.float32) if self.use_rope else None
            if self.imag2:                                    # imagined token k shares position k
                pos = torch.arange(self.K, device=x.device, dtype=torch.float32).repeat(2)
                if c is not None:
                    pos = torch.cat([pos, pos.new_tensor([self.K])])
            for b in self.blocks:
                z = b(z, pos, bias)
        return z

    def _imag_stay_forward(self, x, x_orig, z, c, B):
        if not hasattr(self, "_teacher"):
            import copy
            self._teacher = [copy.deepcopy(nn.ModuleList(self._teacher_mods())).requires_grad_(False)]
        pos = torch.arange(self.K, device=x.device, dtype=torch.float32) if self.use_rope else None
        zc = self.imag_ctx_blk(z, pos)
        if self.imag_wpool:
            w = torch.softmax(self.imag_score(zc).squeeze(-1), 1)                        # [B,K] per-patch weights
            self._imag_w = w.detach()
            zc = (zc * w[..., None]).sum(1)
        else:
            zc = zc.mean(1)
        mu, lv = self.imag_head(zc).chunk(2, -1)                                         # [B,d] each
        lv = lv.clamp(-6, 6)
        if self.training:
            tm = self._teacher[0]
            tp, tl, tt, tb = tm[:4]
            with torch.no_grad():
                hx = tp(x_orig)
                if self.windows:                              # landmark-anchored windows of different lengths
                    hs = []
                    for s_, e_ in self.windows:
                        _, (hn_, _) = tl(hx[:, s_:e_])
                        hs.append(hn_.transpose(0, 1).reshape(B, -1))
                    ht = torch.stack(hs, 1)
                else:
                    _, (ht, _) = tl(self.patches(hx).reshape(B * self.K, self.patch, self.d))
                    ht = ht.transpose(0, 1).reshape(B, self.K, -1)
                zt0 = tt(ht)
                zt = zt0
                for b in tb:
                    zt = b(zt, pos, None)
                if self.imag_wshared:                         # teacher summary pooled with the teacher's own weights
                    wt = torch.softmax(tm[5](tm[4](zt0, pos)).squeeze(-1), 1)
                    zt = (zt * wt[..., None]).sum(1)
                else:
                    zt = zt.mean(1)
            self.imag_loss = 0.5 * (lv + (zt - mu) ** 2 / lv.exp()).mean()
        it = self.imag_proj(torch.cat([mu, lv], -1)) + self.imag_type                   # [B,d] imagined stay token
        if self.pgate:
            n = self.n_val
            obs = torch.stack([torch.cat([x[:, s_:s_ + self.patch, 2 * n:3 * n].mean(1),
                                          x[:, s_:s_ + self.patch, 3 * n:4 * n].mean(1)], -1) for s_ in self.starts], 1)
            g = torch.sigmoid(self.gate_net(obs))                                        # [B,K,1]
            self._gate = g.detach().squeeze(-1)
            u = self.gate_mix(torch.cat([z, it[:, None].expand_as(z)], -1))
            z = z + g * (u - z)
        z = torch.cat([z, it[:, None]], 1)
        if c is not None:
            z = torch.cat([z, self.ctx(c)[:, None]], 1)
        if self.use_rope:
            pos = torch.cat([pos, pos.new_tensor([self.K] * (z.shape[1] - self.K))])
        for b in self.blocks:
            z = b(z, pos, None)
        if self.imag_wshared:                                 # prediction pooling: real tokens by w, imagined token 1/(K+1)
            n = z.shape[1]
            self._pool_w = torch.cat([w * self.K / n, w.new_full((B, n - self.K), 1.0 / n)], 1)
        return z

    def _dyn_forward(self, x, x_orig, c, B):
        if not hasattr(self, "_teacher"):
            import copy
            self._teacher = [copy.deepcopy(nn.ModuleList(self._teacher_mods())).requires_grad_(False)]
        K = self.Kd
        if getattr(self, "pre_attn", False):                  # reversal: hourly attention -> reset patches -> mean pooling
            h = self.proj(x)
            ph = torch.arange(h.shape[1], device=x.device, dtype=torch.float32)
            for b in self.blocks:
                h = b(h, ph, None)
            return self.tok(self.dyn(h, x))
        z = self.tok(self.dyn(self.proj(x), x))                                         # [B,2K,d]
        if not self.imag2 and getattr(self, "clean", False):  # clean encoder: K reset-patch tokens -> attention
            pos = torch.arange(K, device=x.device, dtype=torch.float32)
            for b in self.blocks:
                z = b(z, pos, None)
            return z
        if not self.imag2:                                    # encoder without any restoration head
            pos = torch.arange(K, device=x.device, dtype=torch.float32).repeat(2)
            z = torch.cat([z, z[:, :1] * 0], 1)               # placeholder token keeps the head layout [2K + 1]
            pos = torch.cat([pos, pos.new_tensor([K])])
            for b in self.blocks:
                z = b(z, pos, None)
            return z
        if getattr(self, "_keep_tok", False):                 # diagnostics: recurrent-path tokens before attention
            self._tok = z[:, :self.Kd].detach()
        pos = torch.arange(K, device=x.device, dtype=torch.float32).repeat(2)
        mu, lv = self.imag_head(self.imag_ctx_blk(z, pos).mean(1)).chunk(2, -1)
        lv = lv.clamp(-6, 6)
        if self.pimag:
            mp, lp = self.pimag_head(z[:, :K].detach() if self.pimag_sg else z[:, :K]).chunk(2, -1)   # [B,K,od]
            lp = lp.clamp(-6, 6)
            if getattr(self, "_keep_diag", False):            # diagnostics: factual and restored patch tokens
                self._diag = (z[:, :K].detach(), mp.detach(), lp.detach())
        if self.training:
            tp, td, tt, tb = self._teacher[0]
            with torch.no_grad():
                zt0 = tt(td(tp(x_orig), x_orig))
                zt = zt0
                for b in tb:
                    zt = b(zt, pos, None)
                zt = zt.mean(1)
            self.imag_loss = 0.5 * (lv + (zt - mu) ** 2 / lv.exp()).mean()
            if self.pimag and self._reg is not None:
                if self.pimag == "lab":
                    msk = self._reg == self.cfo_gaug_reg.index("lab")
                else:
                    msk = self._reg >= 0
                if msk.any() and self.pimag_target == "values":
                    n = self.n_val
                    xv, xm = x_orig[msk][..., LAB_IDX], x_orig[msk][..., [2 * n + i for i in LAB_IDX]]
                    st, en = self.dyn.starts, self.dyn.ends
                    cnt = torch.stack([xm[:, a:b + 1].sum(1) for a, b in zip(st, en)], 1)              # [b,K,10]
                    tgt = torch.stack([(xv[:, a:b + 1] * xm[:, a:b + 1]).sum(1) for a, b in zip(st, en)], 1) / cnt.clamp(min=1)
                    w = (cnt > 0).float()                     # only labs actually measured in the factual block
                    nll = 0.5 * (lp[msk] + (tgt - mp[msk]) ** 2 / lp[msk].exp())
                    self.imag_loss = self.imag_loss + (nll * w).sum() / w.sum().clamp(min=1)
                elif msk.any():
                    tgt = zt0[msk][:, :K]
                    self.imag_loss = self.imag_loss + 0.5 * (lp[msk] + (tgt - mp[msk]) ** 2 / lp[msk].exp()).mean()
        it = self.imag_proj(torch.cat([mu, lv], -1)) + self.imag_type
        z = torch.cat([z, it[:, None]], 1)
        pos = torch.cat([pos, pos.new_tensor([K])])
        if self.pimag and not getattr(self, "_drop_pimag", False):   # restored patch token k shares position k
            z = torch.cat([z, self.pimag_proj(torch.cat([mp, lp], -1)) + self.pimag_type], 1)
            pos = torch.cat([pos, torch.arange(K, device=x.device, dtype=torch.float32)])
        if c is not None:
            z = torch.cat([z, self.ctx(c)[:, None]], 1)
            pos = torch.cat([pos, pos.new_tensor([K])])
        for b in self.blocks:
            z = b(z, pos, None)
        return z

    def _teacher_mods(self):
        if self.dynpatch and not self.imag2:
            return []
        if self.dynpatch:
            return [self.proj, self.dyn, self.tok, self.blocks]
        mods = [self.proj, self.lstm, self.tok]
        if self.imag_stay:
            mods.append(self.blocks)
        if self.imag_wshared:
            mods += [self.imag_ctx_blk, self.imag_score]
        return mods

    @torch.no_grad()
    def ema_update(self):
        if self.imag2 and hasattr(self, "_teacher"):
            for pt, ps in zip(self._teacher[0].parameters(), nn.ModuleList(self._teacher_mods()).parameters()):
                pt.mul_(self.imag_m).add_(ps.detach(), alpha=1 - self.imag_m)


class PatchFusion(nn.Module):
    """Manuscript model: tokens -> mean pool -> dropout -> Linear(256,1). One model per endpoint."""

    def __init__(self, **kw):
        super().__init__()
        self.enc = PatchFusionEncoder(**kw)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.enc.d, 1))

    def forward(self, x):
        return self.head(self.enc(x).mean(1)).squeeze(-1)


# ---------------------------------------------------------------- baselines
class LSTMBase(nn.Module):
    def __init__(self, in_dim=66, d=256, h=360):
        super().__init__()
        self.proj, self.lstm, self.head = nn.Linear(in_dim, d), nn.LSTM(d, h, batch_first=True), nn.Linear(h, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(self.proj(x))
        return self.head(hn[-1]).squeeze(-1)


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device)[:, None].float()
    i = torch.arange(0, d, 2, device=device).float()
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos / 10000 ** (i / d))
    pe[:, 1::2] = torch.cos(pos / 10000 ** (i / d))
    return pe


class TransformerBase(nn.Module):
    """12 hourly tokens, absolute sinusoidal positions, 2 layers d256 ff1024, mean pool."""

    def __init__(self, in_dim=66, d=256, layers=2, heads=8, ff=1024, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        z = self.proj(x) + sinusoid(x.shape[1], self.proj.out_features, x.device)
        for b in self.blocks:
            z = b(z)
        return self.head(z.mean(1)).squeeze(-1)


class PatchTST(nn.Module):
    """channel-independent patching (patch 6, stride 3), shared transformer, mean over patches+channels."""

    def __init__(self, in_dim=66, patch=6, stride=3, T=12, d=128, layers=3, heads=8, ff=256, dropout=0.2):
        super().__init__()
        self.patch, self.starts = patch, list(range(0, T - patch + 1, stride))
        self.emb = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(len(self.starts), d) * 0.02)
        self.ch = nn.Parameter(torch.randn(in_dim, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x):
        B, T, C = x.shape
        p = torch.stack([x[:, s:s + self.patch] for s in self.starts], 1)    # [B,K,P,C]
        p = p.permute(0, 3, 1, 2).reshape(B * C, len(self.starts), self.patch)
        z = self.emb(p) + self.pos + self.ch.repeat(B, 1)[:, None]
        for b in self.blocks:
            z = b(z)
        return self.head(z.mean(1).view(B, C, -1).mean(1)).squeeze(-1)


class GRUD(nn.Module):
    """GRU-D (Che et al. 2018) on the 22 value channels with masks and elapsed time since last observation."""

    def __init__(self, n=22, h=128, dropout=0.2):
        super().__init__()
        self.n, self.h = n, h
        self.gx = nn.Linear(n, n)          # input decay (diagonal used)
        self.gh = nn.Linear(n, h)
        self.cell = nn.GRUCell(3 * n, h)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, x):
        v, m = x[..., :self.n], x[..., 2 * self.n:]
        B, T, _ = v.shape
        delta = torch.zeros(B, self.n, device=x.device)
        last = torch.zeros(B, self.n, device=x.device)    # standardised mean = 0
        hs = torch.zeros(B, self.h, device=x.device)
        for t in range(T):
            mt, vt = m[:, t], v[:, t]
            gx = torch.exp(-F.relu(delta * torch.diagonal(self.gx.weight) + self.gx.bias))
            xin = mt * vt + (1 - mt) * gx * last
            hs = hs * torch.exp(-F.relu(self.gh(delta)))
            hs = self.cell(torch.cat([xin, mt, delta], -1), hs)
            last = torch.where(mt > 0, vt, last)
            delta = torch.where(mt > 0, torch.ones_like(delta), delta + 1)
        return self.head(hs).squeeze(-1)


class BaseTokens(nn.Module):
    """reference encoders for the dynamic competing-risk setting: input [B,T,88] -> tokens [B,K,d] (no patches)."""

    def __init__(self, kind, in_dim=88, T=48, n_val=22, dropout=0.2, **_):
        super().__init__()
        self.kind, self.n = kind, n_val
        if kind == "lstm":                                    # one LSTM over the whole look-back, last state
            self.proj, self.rnn, self.d = nn.Linear(in_dim, 256), nn.LSTM(256, 360, batch_first=True), 360
        elif kind == "transformer":                           # hourly tokens, sinusoidal positions, mean pooling
            self.proj, self.d = nn.Linear(in_dim, 256), 256
            self.blocks = nn.ModuleList([Block(256, 8, 1024, dropout) for _ in range(2)])
        elif kind == "patchtst":                              # channel-independent patches (6 h, stride 3)
            self.m = PatchTST(in_dim=in_dim, T=T, dropout=dropout)
            self.d = 128
        elif kind == "grud":                                  # GRU-D on values, masks and elapsed time
            self.m, self.d = GRUD(n=n_val, dropout=dropout), 128
        elif kind in ("tbal", "lgtrl", "hipatch", "raindrop"):  # common-input adaptations (hier_baselines.py)
            import hier_baselines as HB
            self.m = {"tbal": HB.TBAL, "lgtrl": HB.LGTRL, "hipatch": HB.HiPatch, "raindrop": HB.Raindrop}[kind](
                in_dim=in_dim, T=T, dropout=dropout)
            self.d = self.m.d
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind in ("tbal", "lgtrl", "hipatch", "raindrop"):
            return self.m(x)
        if self.kind == "lstm":
            _, (hn, _) = self.rnn(self.proj(x))
            return hn[-1][:, None]
        if self.kind == "transformer":
            z = self.proj(x) + sinusoid(x.shape[1], 256, x.device)
            for b in self.blocks:
                z = b(z)
            return z
        if self.kind == "patchtst":
            m = self.m
            B, T, C = x.shape
            p = torch.stack([x[:, s:s + m.patch] for s in m.starts], 1)
            p = p.permute(0, 3, 1, 2).reshape(B * C, len(m.starts), m.patch)
            z = m.emb(p) + m.pos + m.ch.repeat(B, 1)[:, None]
            zs = []
            for c in z.split(32768):                          # attention kernel batch limit (65535 sequences)
                for b in m.blocks:
                    c = b(c)
                zs.append(c)
            return torch.cat(zs).view(B, C, len(m.starts), -1).mean(1)
        m, n = self.m, self.n                                 # grud: values 0:n, masks 2n:3n
        v, mk = x[..., :n], x[..., 2 * n:3 * n]
        B, T, _ = v.shape
        delta = v.new_zeros(B, n)
        last = v.new_zeros(B, n)
        hs = v.new_zeros(B, m.h)
        for t in range(T):
            mt, vt = mk[:, t], v[:, t]
            gx = torch.exp(-F.relu(delta * torch.diagonal(m.gx.weight) + m.gx.bias))
            xin = mt * vt + (1 - mt) * gx * last
            hs = hs * torch.exp(-F.relu(m.gh(delta)))
            hs = m.cell(torch.cat([xin, mt, delta], -1), hs)
            last = torch.where(mt > 0, vt, last)
            delta = torch.where(mt > 0, torch.ones_like(delta), delta + 1)
        return hs[:, None]


class HorizonBase(nn.Module):
    """reference encoder + the same competing-risk head (mean-pooled tokens) as PatchFusion."""

    def __init__(self, kind, **kw):
        super().__init__()
        self.enc = BaseTokens(kind, **kw)
        self.head = CompetingRiskHead(self.enc.d, queries=False)

    def forward(self, x):
        return self.head(self.enc(x))


# ---------------------------------------------------------------- multi-horizon extensions
class MultiTaskPF(nn.Module):
    """shared PatchFusion encoder + 4 independent sigmoid heads (ablation: label sharing without coherence)."""

    def __init__(self, **kw):
        super().__init__()
        self.enc = PatchFusionEncoder(**kw)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.enc.d, N_HORIZON))

    def forward(self, x):
        return self.head(self.enc(x).mean(1))           # logits [B,4] (independent)


class CompetingRiskHead(nn.Module):
    """per-interval {death, discharge-alive, remain} conditional probabilities -> cumulative incidence of death.

    queries=True : each interval owns a learnable query that cross-attends over the trajectory tokens
                   (horizon-specific evidence selection); queries=False: shared mean-pooled vector.
    """

    def __init__(self, d=256, heads=8, dropout=0.2, queries=True, n_int=N_HORIZON, concat_k=0):
        super().__init__()
        self.queries, self.n_int, self.concat_k = queries, n_int, concat_k
        if queries:
            self.q = nn.Parameter(torch.randn(n_int, d) * 0.02)
            self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 3))
        elif concat_k:
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d * concat_k, n_int * 3))
        else:
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_int * 3))
        self.last_attn = None

    def forward(self, z):
        B = z.shape[0]
        if self.queries:
            q = self.q[None].expand(B, -1, -1)
            h, w = self.att(q, z, z, need_weights=True)
            self.last_attn = w.detach()
            logits = self.out(self.norm(h + q))          # [B,n_int,3]
        elif self.concat_k:
            logits = self.out(z.reshape(B, -1)).view(B, self.n_int, 3)   # each interval weights each token
        else:
            logits = self.out(z.mean(1)).view(B, self.n_int, 3)
        # final interval runs to discharge: no 'remain' state
        neg = torch.zeros_like(logits)
        neg[:, -1, 2] = -1e4
        return logits + neg


def cr_probs(logits):
    """logits [B,J,3] -> log conditional probs, cumulative incidence of death [B,J]."""
    lp = F.log_softmax(logits, -1)
    p = lp.exp()
    surv = torch.cumprod(torch.cat([torch.ones_like(p[:, :1, 2]), p[:, :-1, 2]], 1), 1)   # S_{j-1}
    cif = torch.cumsum(surv * p[:, :, 0], 1)
    return lp, cif


def cr_nll(logits, interval, etype, death_weight=1.0, sw=None, reduce=True):
    """discrete-time competing-risk likelihood. interval j in 0..J-1, etype 1=death 2=discharge.
    sw: optional per-sample weights (importance reweighting); reduce=False returns per-sample losses."""
    lp, _ = cr_probs(logits)
    J = logits.shape[1]
    ar = torch.arange(J, device=logits.device)[None]
    before = (ar < interval[:, None]).float()
    remain = (lp[:, :, 2] * before).sum(1)
    idx = (etype - 1).long()                              # 0 death, 1 discharge
    ev = lp[torch.arange(lp.shape[0], device=lp.device), interval.long()].gather(1, idx[:, None]).squeeze(1)
    w = torch.where(etype == 1, torch.full_like(ev, death_weight), torch.ones_like(ev))
    per = -(remain + w * ev)
    if not reduce:
        return per
    if sw is not None:
        return (sw * per).sum() / sw.sum()
    return per.mean()


class HorizonPF(nn.Module):
    """PatchFusion encoder + competing-risk head (queries on/off)."""

    def __init__(self, queries=True, concat=False, **kw):
        super().__init__()
        self.enc = PatchFusionEncoder(**kw)
        self.head = CompetingRiskHead(self.enc.d, queries=queries, concat_k=self.enc.K if concat else 0)
        if self.enc.dynpatch and not getattr(self.enc, "clean", False):   # interval weight on the two branches
            self.dyn_a = nn.Parameter(torch.zeros(N_HORIZON))
        if self.enc.obsb and not self.enc.obsb_ind_in_enc:
            self.ind_w = nn.Parameter(torch.zeros(N_HORIZON, self.enc.n_val))
            self.register_buffer("ind_b", torch.zeros(self.enc.n_val))      # per-variable evidence bound (logit units)

    def forward(self, x):
        z = self.enc(x)
        if self.enc.dynpatch and not getattr(self.enc, "clean", False):
            K, B, J = self.enc.Kd, z.shape[0], self.head.n_int
            a = torch.sigmoid(self.dyn_a)[None, :, None]                                   # [1,J,1]
            v = (1 - 1 / (K + 1)) * (a * z[:, None, :K].mean(2) + (1 - a) * z[:, None, K:2 * K].mean(2)) \
                + z[:, None, 2 * K] / (K + 1)                                               # [B,J,d]
            lg = self.head.out(v).view(B, J, J, 3)
            lg = lg[:, torch.arange(J), torch.arange(J)]                                    # interval j from its own mix
            neg = torch.zeros_like(lg)
            neg[:, -1, 2] = -1e4
            return lg + neg
        if self.enc._pool_w is not None and not self.head.queries and not self.head.concat_k:   # contamination-weighted pooling
            B = z.shape[0]
            lg = self.head.out((z * self.enc._pool_w[..., None]).sum(1)).view(B, self.head.n_int, 3)
            neg = torch.zeros_like(lg)
            neg[:, -1, 2] = -1e4
            out = lg + neg
        else:
            out = self.head(z)
        if hasattr(self, "ind_w"):                            # |contribution of indicator k| <= ind_b[k]
            ev = (torch.tanh(self.ind_w)[None] * self.ind_b * self.enc._ind[:, None, :]).sum(-1)   # [B, J]
            out = torch.cat([out[..., :1] + ev[..., None], out[..., 1:]], -1)
        return out


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, m in [("PatchFusion", PatchFusion()), ("LSTM", LSTMBase()), ("Transformer", TransformerBase()),
                    ("PatchTST", PatchTST()), ("GRU-D", GRUD()), ("HorizonPF", HorizonPF()),
                    ("HorizonPF-noQ", HorizonPF(queries=False)), ("MultiTaskPF", MultiTaskPF())]:
        y = m(torch.randn(4, 12, 66))
        print(f"{name:14s} params={count(m):>9,d} out={tuple(y.shape)}")


def prism_view(X, rng_q=(0.3, 1.0), n=22):
    """PRISM-style sampling-pattern variation: every non-vital channel keeps each observed hour with a per-sample,
    per-channel probability q ~ U(rng_q); dropped hours get value/delta/mask/availability set to their absent state."""
    B, T, _ = X.shape
    q = torch.empty(B, 1, n - 6, device=X.device).uniform_(*rng_q)
    keep = (torch.rand(B, T, n - 6, device=X.device) < q).float()
    k = torch.ones(B, T, n, device=X.device)
    k[:, :, 6:] = keep
    obs = X[..., 2 * n:3 * n] > 0                             # only observed hours are dropped
    k = torch.where(obs, k, torch.ones_like(k))
    return X * k.repeat(1, 1, X.shape[-1] // n)
