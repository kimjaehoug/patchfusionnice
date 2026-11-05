"""Common-input adaptations of published irregular / hierarchical ICU encoders for the dynamic 48-h setting.

Input x [B,T,88] = standardised carried values (0:22), hourly change (22:44), measured mask (44:66), availability (66:88).
Each encoder returns tokens [B,K,d] that feed the same competing-risk head as PatchFusion. Demographic embeddings and
dataset-specific preprocessing of the original papers are omitted (common inputs only).

TBAL      time-aware bidirectional LSTM with attention pooling (Zheng et al., 2025): the cell memory is decayed by the
          elapsed time since the last measurement before each step (T-LSTM decay 1/log(e + dt)).
LGTRL     local recurrent + global transformer (Zou et al., 2023): one BiLSTM over the whole look-back, its hourly states
          related by transformer self-attention (recurrent state is never confined to a patch).
HiPatch   hierarchical patch graph (Luo et al., 2025): observation nodes (variable, hour) are attention-pooled into
          (variable, 3-h patch) nodes, a graph attention layer mixes variables inside each patch, and adjacent patches
          are merged level by level (3 -> 6 -> 12 -> 24 -> 48 h) with the same intra-patch graph at every level.
Raindrop  sensor graph network (Zhang et al., 2022): per-sensor observation embeddings, learned inter-sensor message
          passing at each hour restricted to measured sensors, per-sensor temporal self-attention, sensor tokens.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_VAL = 22


def split(x, n=N_VAL):
    return x[..., :n], x[..., 2 * n:3 * n]                   # values, measured mask


def elapsed(m):
    """hours since the last measurement of each variable before each hour (0 at a measurement) [B,T,n]."""
    B, T, n = m.shape
    out, cur = [], m.new_zeros(B, n)
    for t in range(T):
        cur = torch.where(m[:, t] > 0, torch.zeros_like(cur), cur + 1)
        out.append(cur)
    return torch.stack(out, 1)


class TBAL(nn.Module):
    def __init__(self, in_dim=88, h=128, dropout=0.2, **_):
        super().__init__()
        self.proj = nn.Linear(in_dim, 128)
        self.cf, self.cb = nn.LSTMCell(128, h), nn.LSTMCell(128, h)
        self.dec_f, self.dec_b = nn.Linear(h, h), nn.Linear(h, h)    # short-term memory component (T-LSTM)
        self.att = nn.Sequential(nn.Linear(2 * h, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop, self.h, self.d = nn.Dropout(dropout), h, 2 * h

    def _run(self, z, dt, cell, dec, rev):
        B, T, _ = z.shape
        hs, cs = z.new_zeros(B, self.h), z.new_zeros(B, self.h)
        out = [None] * T
        for t in (range(T - 1, -1, -1) if rev else range(T)):
            cst = torch.tanh(dec(cs))                         # decompose memory, decay the short-term part by elapsed time
            cs = cs - cst + cst / torch.log(math.e + dt[:, t, None])
            hs, cs = cell(z[:, t], (hs, cs))
            out[t] = hs
        return torch.stack(out, 1)

    def forward(self, x):
        _, m = split(x)
        dt = elapsed(m).mean(-1)                              # [B,T] mean elapsed time over variables
        z = self.proj(x)
        H = torch.cat([self._run(z, dt, self.cf, self.dec_f, False), self._run(z, dt, self.cb, self.dec_b, True)], -1)
        a = torch.softmax(self.att(H), 1)                     # attention pooling over hours
        return self.drop((a * H).sum(1))[:, None]


class LGTRL(nn.Module):
    def __init__(self, in_dim=88, d=256, h=128, layers=2, heads=8, ff=512, dropout=0.2, **_):
        super().__init__()
        from models import Block
        self.proj = nn.Linear(in_dim, d)
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.tok = nn.Linear(2 * h, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.d = d

    def forward(self, x):
        z = self.tok(self.lstm(self.proj(x))[0])              # local (recurrent, whole look-back) hourly states
        pos = torch.arange(z.shape[1], device=x.device, dtype=torch.float32)
        for b in self.blocks:                                 # global self-attention over hours
            z = b(z, pos, None)
        return z


class GraphAtt(nn.Module):
    """dense graph attention over the variable nodes of one patch (mask: node present)."""

    def __init__(self, d, heads=4, dropout=0.2):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, z, present):                            # z [N,V,d], present [N,V] bool
        kpm = ~present
        kpm = kpm & ~(~present).all(1, keepdim=True)          # empty patch: attend to all (output masked later)
        z = z + self.att(self.n1(z), self.n1(z), self.n1(z), key_padding_mask=kpm, need_weights=False)[0]
        return z + self.ff(self.n2(z))


class HiPatch(nn.Module):
    def __init__(self, T=48, d=64, patch=3, dropout=0.2, **_):
        super().__init__()
        self.patch, self.d = patch, d
        self.var = nn.Parameter(torch.randn(N_VAL, d) * 0.02)
        self.val = nn.Linear(1, d)
        self.time = nn.Linear(1, d)
        self.pool = nn.Linear(d, 1)                           # attention pooling of member nodes into a parent node
        n_lv = int(math.log2(T // patch)) + 1
        self.gat = nn.ModuleList([GraphAtt(d, dropout=dropout) for _ in range(n_lv)])
        self.lvl = nn.Parameter(torch.zeros(n_lv, d))

    def _merge(self, z, pres, g):                             # z [B,K,g,V,d] members -> [B,K,V,d]
        s = self.pool(z).squeeze(-1).masked_fill(~pres, -1e4)
        w = torch.softmax(s, 2) * pres.float()
        return (w[..., None] * z).sum(2), pres.any(2)

    def forward(self, x):
        v, m = split(x)
        B, T, V = v.shape
        tt = (torch.arange(T, device=x.device, dtype=torch.float32) / T)[None, :, None, None]
        e = self.var + self.val(v[..., None]) + self.time(tt.expand(B, T, V, 1))          # observation nodes [B,T,V,d]
        pres = m > 0
        K = T // self.patch
        z, p = self._merge(e.view(B, K, self.patch, V, self.d), pres.view(B, K, self.patch, V), self.patch)
        outs = []
        for lv, g in enumerate(self.gat):
            Kc = z.shape[1]
            zz = g(z.reshape(B * Kc, V, self.d) + self.lvl[lv], p.reshape(B * Kc, V)).view(B, Kc, V, self.d)
            z = zz * p[..., None]
            outs.append(z.sum(2) / p.sum(2, keepdim=True).clamp(min=1))                     # patch tokens of this level
            if Kc == 1:
                break
            if Kc % 2:                                        # odd count: pad one empty patch
                z = torch.cat([z, torch.zeros_like(z[:, :1])], 1)
                p = torch.cat([p, torch.zeros_like(p[:, :1])], 1)
                Kc += 1
            z, p = self._merge(z.view(B, Kc // 2, 2, V, self.d), p.view(B, Kc // 2, 2, V), 2)
        return torch.cat(outs, 1)                             # multi-scale patch tokens


class Raindrop(nn.Module):
    def __init__(self, T=48, d=32, layers=2, heads=4, dropout=0.2, **_):
        super().__init__()
        self.d = d
        self.obs = nn.Parameter(torch.randn(N_VAL, d) * 0.1)     # per-sensor observation embedding (value * weight)
        self.obs_b = nn.Parameter(torch.zeros(N_VAL, d))
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.adj = nn.Parameter(torch.zeros(N_VAL, N_VAL))      # learned inter-sensor graph prior
        self.msg = nn.Linear(d, d)
        self.sens = nn.Parameter(torch.randn(N_VAL, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(el, layers)
        self.register_buffer("pe", self._pe(T, d))

    @staticmethod
    def _pe(T, d):
        pos = torch.arange(T)[:, None].float()
        i = torch.arange(0, d, 2).float()
        pe = torch.zeros(T, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / 10000 ** (i / d)), torch.cos(pos / 10000 ** (i / d))
        return pe

    def forward(self, x):
        v, m = split(x)
        B, T, V = v.shape
        h = F.relu(v[..., None] * self.obs + self.obs_b) * m[..., None]                   # [B,T,V,d], 0 if unobserved
        a = (self.q(h) @ self.k(h).transpose(-1, -2)) / math.sqrt(self.d) + self.adj       # [B,T,V,V]
        a = a.masked_fill(m[:, :, None, :] == 0, -1e4)
        w = torch.softmax(a, -1) * (m[:, :, None, :] > 0)
        h = h + F.relu(self.msg(w @ h))                       # observed sensors pass messages to every sensor
        z = (h + self.sens + self.pe[None, :, None]).permute(0, 2, 1, 3).reshape(B * V, T, self.d)
        kpm = (m.permute(0, 2, 1).reshape(B * V, T) == 0)
        none = kpm.all(1)
        kpm[none] = False                                     # never-measured sensor: attend over its message states
        zs = []
        for zc, kc in zip(z.split(32768), kpm.split(32768)):
            zs.append(self.temporal(zc, src_key_padding_mask=kc))
        z = torch.cat(zs)
        keep = (~kpm).float()[..., None]
        z = (z * keep).sum(1) / keep.sum(1).clamp(min=1)      # per-sensor temporal aggregation
        return z.view(B, V, self.d)                           # sensor tokens
