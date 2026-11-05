"""Feasibility of footprint-certified PatchFusion on the hour-12 data (fold 0 of the main protocol, frozen eICU).

Threat set: within one 3-hour window (0-3, 3-6, 6-9 or 9-12 h) any subset of sparse-variable measurements may be missing,
with persistence re-imputation. Training adds kappa * (certified CIF width for y24 and yhosp) under a randomly drawn
window per sample (kappa ramped over the first epochs). Certification on eICU is done in float64.

Usage: python cert_train.py --tag NAME --kappa 1.0 [--state reset|carry] [--seed 2]
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

import train as T
import models as M
import certify as CF

WINDOWS = [(0, 3), (3, 6), (6, 9), (9, 12)]


def capture_box(X, win, wlen, cvars, n_val=22):
    """GPU threat box for capture/documentation gaps of carry-0 variables: any subset of their observations with hour
    in [win*wlen, (win+1)*wlen) may be missing. A deleted observation at hour t moves value/mask at t to (0, 0) and the
    first differences at t and t+1 to 0, so each affected cell's interval is the hull of its value and 0.
    X: [B,T,66]; win: [B] window index."""
    B, Tn, _ = X.shape
    hours = torch.arange(Tn, device=X.device)[None]
    a = (win * wlen)[:, None]
    inw = (hours >= a) & (hours < a + wlen)                               # [B,T]
    lo, hi = X.clone(), X.clone()
    for c in cvars:
        obs = (X[:, :, 2 * n_val + c] > 0) & inw                            # deletable observations
        nxt = torch.zeros_like(obs)
        nxt[:, 1:] = obs[:, :-1]
        for ch, cells in ((c, obs), (n_val + c, obs | nxt), (2 * n_val + c, obs)):
            v = X[:, :, ch]
            lo[:, :, ch] = torch.where(cells, torch.minimum(v, torch.zeros_like(v)), v)
            hi[:, :, ch] = torch.where(cells, torch.maximum(v, torch.zeros_like(v)), v)
    return CF.Box(lo, hi)


def boxes(d, idx, stats, dev):
    out_lo, out_hi = [], []
    for a, b in WINDOWS:
        bx = CF.input_box(*CF.window_threat(d["x"][idx], d["m"][idx], a, b), stats, avail=False)
        out_lo.append(bx.lo)
        out_hi.append(bx.hi)
    return torch.stack(out_lo).to(dev), torch.stack(out_hi).to(dev)          # [W,N,T,C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--state", default="reset")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max_epochs", type=int, default=40)
    ap.add_argument("--ramp", type=int, default=5)
    ap.add_argument("--n_cert", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=300)
    ap.add_argument("--cert_bs", type=int, default=0)       # bound the certified-width term on a random sub-batch (memory)
    ap.add_argument("--data", default="")                  # "" hour-12, "_dyn" dynamic landmarks (T=48)
    ap.add_argument("--threat", default="labs")            # labs: 3-h windows, all sparse vars; capture: 6-h windows, GCS/UO/NE
    a = ap.parse_args()
    T.LOG = f"{T.ROOT}/logs/train/{a.tag}.log"
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    d, e = T.load("mimic" + a.data), T.load("eicu" + a.data)
    Tn = d["x"].shape[1]
    cap = [CF.RG.VARS.index(v) for v in ("gcs", "uo", "ne")]
    wlen = 3 if a.threat == "labs" else 6
    nwin = Tn // wlen
    N = len(d["x"])
    order = np.random.RandomState(0).permutation(N)
    tr_o, te_o = next(iter(GroupKFold(5).split(order, groups=d["subject_id"][order])))
    tr_idx = order[tr_o]
    r = np.random.RandomState(100)
    subj = np.unique(d["subject_id"][tr_idx])
    is_va = np.isin(d["subject_id"][tr_idx], r.choice(subj, int(0.2 * len(subj)), replace=False))
    in_tr, in_va = tr_idx[~is_va], tr_idx[is_va]
    stats = T.fit_stats(d["x"][in_tr])
    tr, va = T.tensors(d, in_tr, stats, dev), T.tensors(d, in_va, stats, dev)
    t0 = time.time()
    if a.threat == "labs":
        blo, bhi = boxes(d, in_tr, stats, dev)
    T.log(f"=== {a.tag} kappa={a.kappa} state={a.state} seed={a.seed}; threat boxes built in {time.time() - t0:.0f}s")
    model = M.HorizonPF(queries=False, cert=True, state=a.state, T=Tn).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    n, bs = len(in_tr), a.bs
    best, best_state, bad = -1, None, 0
    Yva = va[1].cpu().numpy()
    for ep in range(1, a.max_epochs + 1):
        te = time.time()
        model.train()
        kap = a.kappa * min(1.0, ep / a.ramp)
        perm = torch.randperm(n, generator=g).to(dev)
        tot_l, tot_w = 0.0, 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = M.cr_nll(model(tr[0][b]), tr[2][b], tr[3][b])
            wv = torch.tensor(0.0, device=dev)
            if kap > 0:
                if a.cert_bs and len(b) > a.cert_bs:
                    b = b[torch.randperm(len(b), generator=g)[:a.cert_bs].to(dev)]
                w = torch.randint(0, nwin, (len(b),), generator=g).to(dev)
                xb = CF.Box(blo[w, b], bhi[w, b]) if a.threat == "labs" else capture_box(tr[0][b], w, wlen, cap)
                lo, hi, zl = CF.cert_bounds(model, xb, return_logits=True)
                wv = torch.log1p(zl.hi - zl.lo).mean()                 # non-saturating certified-width surrogate
                loss = loss + kap * wv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_l += loss.item()
            tot_w += float(wv.detach())
        P = T.predict("hpf_noq", model, va[0])
        s = float(np.mean([average_precision_score(Yva[:, j], P[:, j]) for j in range(4)]))
        mark = ""
        if s > best and ep >= a.ramp:                        # select only once the certification term is at full weight
            best, bad, mark = s, 0, " *best"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif ep >= a.ramp:
            bad += 1
        T.log(f"[{a.tag}] ep{ep:03d} loss={tot_l / (n / bs):.4f} width={tot_w / (n / bs):.4f} val_sel={s:.4f} "
              f"y24 AUROC={roc_auc_score(Yva[:, 0], P[:, 0]):.4f} yhosp AUROC={roc_auc_score(Yva[:, 3], P[:, 3]):.4f} ({time.time() - te:.0f}s){mark}")
        if bad >= 8:
            break
    model.load_state_dict(best_state)
    model.eval()
    # ---------------- evaluation: clean eICU metrics + certification (float64) on a random eICU subset
    Pe = np.concatenate([T.predict("hpf_noq", model, T.tensors(e, np.arange(k, min(k + 20000, len(e["x"]))), stats, dev)[0])
                         for k in range(0, len(e["x"]), 20000)])          # chunked: keeps GPU memory flat
    res = {"tag": a.tag, "kappa": a.kappa, "state": a.state, "seed": a.seed, "data": a.data, "threat": a.threat}
    for j, ep_ in enumerate(T.ENDPOINTS):
        res[f"ext_{ep_}_auroc"] = float(roc_auc_score(e[ep_], Pe[:, j]))
        res[f"ext_{ep_}_auprc"] = float(average_precision_score(e[ep_], Pe[:, j]))
    Pv = T.predict("hpf_noq", model, va[0])
    thr = float(np.quantile(Pv[:, 3], 0.9))                     # alert = top-10% development risk (yhosp)
    sub = np.random.RandomState(7).choice(len(e["x"]), a.n_cert, replace=False)
    m64 = model.double()
    lo_all, hi_all = [], []
    Xs = T.tensors(e, sub, stats, dev)[0]
    with torch.no_grad():
        for wi in range(nwin):
            los, his = [], []
            for k in range(0, len(sub), 500):
                s_ = sub[k:k + 500]
                if a.threat == "labs":
                    wa, wb = wi * wlen, (wi + 1) * wlen
                    bx = CF.input_box(*CF.window_threat(e["x"][s_], e["m"][s_], wa, wb), stats, avail=False)
                    bx = CF.Box(bx.lo.to(dev), bx.hi.to(dev))
                else:
                    xs = Xs[k:k + 500]
                    bx = capture_box(xs, torch.full((len(xs),), wi, device=dev), wlen, cap)
                lo, hi = CF.cert_bounds(m64, CF.Box(bx.lo.double(), bx.hi.double()))
                los.append(lo[:, 3].cpu().numpy())
                his.append(hi[:, 3].cpu().numpy())
            lo_all.append(np.concatenate(los))
            hi_all.append(np.concatenate(his))
    lo_all, hi_all = np.stack(lo_all), np.stack(hi_all)                  # [W, n]
    p = Pe[sub, 3]
    stable = ((hi_all < thr) | (lo_all >= thr)).all(0)
    res.update(alert_thr=thr, cert_width_median=float(np.median(hi_all - lo_all)),
               cert_width_p90=float(np.quantile(hi_all - lo_all, 0.9)),
               certified_alert_stable=float(stable.mean()),
               certified_among_alerted=float(stable[p >= thr].mean()) if (p >= thr).any() else float("nan"),
               certified_among_not_alerted=float(stable[p < thr].mean()))
    T.log(json.dumps(res))
    with open(f"{T.ROOT}/results/cert_runs.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
