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

    def forward(self, x, pos=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if pos is not None:
            q, k = rope(q, pos), rope(k, pos)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    """pre-norm transformer layer; same parameter count as nn.TransformerEncoderLayer(norm_first=True)."""

    def __init__(self, d, h, ff, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = RoPEAttention(d, h, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x, pos=None):
        x = x + self.d1(self.att(self.n1(x), pos))
        return x + self.d2(self.ff(self.n2(x)))


# ---------------------------------------------------------------- PatchFusion encoder
class PatchFusionEncoder(nn.Module):
    """[B,12,66] -> trajectory tokens [B,K,256] after RoPE global attention (manuscript Table S4b)."""

    def __init__(self, in_dim=66, d=256, lstm_h=189, patch=6, stride=3, T=12, layers=2, heads=8, ff=512,
                 dropout=0.2, bidir=True, use_rope=True, use_global=True, state="reset"):
        """state='reset': recurrent state restarts in every patch (manuscript).
        state='carry': one BiLSTM over the whole 12 h; token k = forward state at the end of patch k and
                       backward state at its start (same parameters and token count, state not confined)."""
        super().__init__()
        self.state = state
        self.patch, self.stride = patch, stride
        self.starts = list(range(0, T - patch + 1, stride))
        self.K = len(self.starts)
        self.proj = nn.Linear(in_dim, d)
        self.lstm = nn.LSTM(d, lstm_h, batch_first=True, bidirectional=bidir)
        self.tok = nn.Linear(lstm_h * (2 if bidir else 1), d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)]) if use_global else None
        self.use_rope = use_rope
        self.d = d

    def patches(self, h):
        return torch.stack([h[:, s:s + self.patch] for s in self.starts], 1)  # [B,K,P,d]

    def forward(self, x):
        B = x.shape[0]
        h = self.proj(x)
        if self.state == "carry":
            out, _ = self.lstm(h)                             # [B,T,2H] states carried across patches
            H = out.shape[-1] // 2
            ends = [s + self.patch - 1 for s in self.starts]
            hn = torch.cat([out[:, ends, :H], out[:, self.starts, H:]], -1)
        else:
            p = self.patches(h)                               # [B,K,P,d]
            _, (hn, _) = self.lstm(p.reshape(B * self.K, self.patch, self.d))   # states restart per patch
            hn = hn.transpose(0, 1).reshape(B, self.K, -1)    # concat terminal fwd/bwd states
        z = self.tok(hn)
        if self.blocks is not None:
            pos = torch.arange(self.K, device=x.device, dtype=torch.float32) if self.use_rope else None
            for b in self.blocks:
                z = b(z, pos)
        return z


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

    def __init__(self, d=256, heads=8, dropout=0.2, queries=True, n_int=N_HORIZON):
        super().__init__()
        self.queries, self.n_int = queries, n_int
        if queries:
            self.q = nn.Parameter(torch.randn(n_int, d) * 0.02)
            self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 3))
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


def cr_nll(logits, interval, etype, death_weight=1.0):
    """discrete-time competing-risk likelihood. interval j in 0..J-1, etype 1=death 2=discharge."""
    lp, _ = cr_probs(logits)
    J = logits.shape[1]
    ar = torch.arange(J, device=logits.device)[None]
    before = (ar < interval[:, None]).float()
    remain = (lp[:, :, 2] * before).sum(1)
    idx = (etype - 1).long()                              # 0 death, 1 discharge
    ev = lp[torch.arange(lp.shape[0], device=lp.device), interval.long()].gather(1, idx[:, None]).squeeze(1)
    w = torch.where(etype == 1, torch.full_like(ev, death_weight), torch.ones_like(ev))
    return -(remain + w * ev).mean()


class HorizonPF(nn.Module):
    """PatchFusion encoder + competing-risk head (queries on/off)."""

    def __init__(self, queries=True, **kw):
        super().__init__()
        self.enc = PatchFusionEncoder(**kw)
        self.head = CompetingRiskHead(self.enc.d, queries=queries)

    def forward(self, x):
        return self.head(self.enc(x))


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, m in [("PatchFusion", PatchFusion()), ("LSTM", LSTMBase()), ("Transformer", TransformerBase()),
                    ("PatchTST", PatchTST()), ("GRU-D", GRUD()), ("HorizonPF", HorizonPF()),
                    ("HorizonPF-noQ", HorizonPF(queries=False)), ("MultiTaskPF", MultiTaskPF())]:
        y = m(torch.randn(4, 12, 66))
        print(f"{name:14s} params={count(m):>9,d} out={tuple(y.shape)}")
