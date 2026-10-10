"""Patient-level GroupKFold OOF evaluation on MIMIC-IV + frozen full-cohort model applied to eICU-CRD.

python train.py --model pf --endpoint y24 --seed 2
python train.py --model hpf --seed 2              (multi-horizon models ignore --endpoint)
Outputs results/preds/{tag}.npz with OOF (MIMIC) and external (eICU) probabilities for all endpoints it predicts.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

import models as M

ROOT = os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENDPOINTS = ["y24", "y48", "y168", "yhosp"]
CUTS = (24.0, 48.0, 168.0)   # interval edges after the hour-12 landmark


LOG = None


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")


def load(src):
    d = np.load(f"{ROOT}/data/{src}_tensor.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def make_input(x, stats):
    mu, sd = stats
    v = (x - mu) / sd
    v = np.clip(v, -10, 10)
    dv = np.zeros_like(v)
    dv[:, 1:] = v[:, 1:] - v[:, :-1]
    dv = np.nan_to_num(dv, nan=0.0)
    return np.nan_to_num(v, nan=0.0), dv


def fit_stats(x):
    mu = np.nanmean(x, axis=(0, 1))
    sd = np.nanstd(x, axis=(0, 1))
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def tensors(d, idx, stats, dev):
    v, dv = make_input(d["x"][idx], stats)
    X = np.concatenate([v, dv, d["m"][idx]], -1).astype(np.float32)
    Y = np.stack([d[e][idx] for e in ENDPOINTS], 1).astype(np.float32)
    t = d["t_event"][idx]
    interval = np.digitize(t, CUTS, right=True)          # 0..3
    return (torch.tensor(X, device=dev), torch.tensor(Y, device=dev),
            torch.tensor(interval, device=dev), torch.tensor(d["event"][idx].astype(np.int64), device=dev))


def focal(logit, y, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logit)
    pt = y * p + (1 - y) * (1 - p)
    at = alpha * y + (1 - alpha) * (1 - y)
    return (at * (1 - pt) ** gamma * F.binary_cross_entropy_with_logits(logit, y, reduction="none")).mean()


MULTI = {"mtpf", "hpf", "hpf_noq", "hpf_w"}


def build(name, cfg):
    kw = cfg.get("enc", {})
    if name == "pf":
        return M.PatchFusion(**kw)
    if name == "lstm":
        return M.LSTMBase()
    if name == "transformer":
        return M.TransformerBase()
    if name == "patchtst":
        return M.PatchTST()
    if name == "grud":
        return M.GRUD()
    if name == "mtpf":
        return M.MultiTaskPF(**kw)
    if name in ("hpf", "hpf_w"):
        return M.HorizonPF(queries=True, **kw)
    if name == "hpf_noq":
        return M.HorizonPF(queries=False, **kw)
    raise ValueError(name)


def loss_fn(name, out, Y, interval, etype, ep_i, cfg):
    if name in ("hpf", "hpf_noq", "hpf_w"):
        return M.cr_nll(out, interval, etype, death_weight=cfg.get("death_weight", 1.0))
    if name == "mtpf":
        return sum(focal(out[:, j], Y[:, j]) for j in range(4)) / 4
    return focal(out, Y[:, ep_i])


@torch.no_grad()
def predict(name, model, X, bs=1024):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        o = model(X[i:i + bs])
        if name in ("hpf", "hpf_noq", "hpf_w"):
            o = M.cr_probs(o)[1]
        else:
            o = torch.sigmoid(o)
        outs.append(o.float().cpu())
    torch.cuda.empty_cache()
    return torch.cat(outs).numpy()


def score(name, P, Y, ep_i):
    if name in MULTI:
        return float(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in range(4)]))
    return float(average_precision_score(Y[:, ep_i], P))


def val_report(name, P, Y, ep_i):
    if name in MULTI:
        cols = range(4)
        P2 = P
    else:
        cols = [ep_i]
        P2 = P.reshape(-1, 1)
    parts = []
    for k, j in enumerate(cols):
        parts.append(f"{ENDPOINTS[j]} AUROC={roc_auc_score(Y[:, j], P2[:, k]):.4f} AUPRC={average_precision_score(Y[:, j], P2[:, k]):.4f}")
    return " | ".join(parts)


def train_one(name, cfg, tr, va, ep_i, seed, max_epochs=100, patience=10, fixed_epochs=None, prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(name, cfg).to(tr[0].device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-3))
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(tr[0])
    bs = cfg.get("bs", 300)
    best, best_ep, best_state, bad = -1, 0, None, 0
    epochs = fixed_epochs or max_epochs
    Yva = va[1].cpu().numpy() if va is not None else None
    log(f"{prefix} start: n_train={n} n_val={0 if va is None else len(va[0])} params={M.count(model):,} "
        f"bs={bs} lr={cfg.get('lr', 1e-3)} wd={cfg.get('wd', 1e-3)} max_epochs={epochs} patience={patience}")
    for ep in range(1, epochs + 1):
        te = time.time()
        model.train()
        perm = torch.randperm(n, generator=g).to(tr[0].device)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            out = model(tr[0][b])
            loss = loss_fn(name, out, tr[1][b], tr[2][b], tr[3][b], ep_i, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        if fixed_epochs:
            log(f"{prefix} ep{ep:03d} train_loss={tot / nb:.5f} ({time.time() - te:.1f}s)")
            continue
        P = predict(name, model, va[0])
        s = score(name, P, Yva, ep_i)
        mark = ""
        if s > best:
            best, best_ep, bad = s, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            mark = " *best"
        else:
            bad += 1
            mark = f" (no improve {bad}/{patience})"
        log(f"{prefix} ep{ep:03d} train_loss={tot / nb:.5f} val_sel={s:.4f} | {val_report(name, P, Yva, ep_i)} "
            f"({time.time() - te:.1f}s){mark}")
        if bad >= patience:
            log(f"{prefix} early stop at ep{ep}; best ep{best_ep} val_sel={best:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default="y24")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no_external", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--n_folds_run", type=int, default=5, help="run only the first k outer folds (quick check)")
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = a.model
    ep_i = ENDPOINTS.index(a.endpoint)
    tag = a.tag or (f"{name}_s{a.seed}" if name in MULTI else f"{name}_{a.endpoint}_s{a.seed}")
    os.makedirs(f"{ROOT}/results/preds", exist_ok=True)
    os.makedirs(f"{ROOT}/logs/train", exist_ok=True)
    global LOG
    LOG = f"{ROOT}/logs/train/{tag}.log"
    t0 = time.time()
    log(f"=== {tag} model={name} endpoint={'all' if name in MULTI else a.endpoint} seed={a.seed} cfg={cfg} "
        f"device={dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")

    d = load("mimic")
    N = len(d["stay_id"])
    groups = d["subject_id"]
    rng = np.random.RandomState(0)                       # identical folds for every model / seed
    order = rng.permutation(N)
    folds = list(GroupKFold(5).split(order, groups=groups[order]))
    ncol = 4 if name in MULTI else 1
    oof = np.full((N, ncol), np.nan, dtype=np.float32)
    best_eps = []
    for f, (tr_o, te_o) in enumerate(folds[:a.n_folds_run]):
        tr_idx, te_idx = order[tr_o], order[te_o]
        r = np.random.RandomState(100 + f)
        subj = np.unique(groups[tr_idx])
        va_subj = r.choice(subj, int(0.2 * len(subj)), replace=False)
        is_va = np.isin(groups[tr_idx], va_subj)
        in_tr, in_va = tr_idx[~is_va], tr_idx[is_va]
        stats = fit_stats(d["x"][in_tr])
        log(f"fold{f}: train={len(in_tr)} val={len(in_va)} test={len(te_idx)} "
            f"test_events=" + ",".join(f"{e}:{int(d[e][te_idx].sum())}" for e in ENDPOINTS))
        model, be, bs_ = train_one(name, cfg, tensors(d, in_tr, stats, dev), tensors(d, in_va, stats, dev),
                                   ep_i, a.seed, max_epochs=a.max_epochs, prefix=f"[{tag} f{f}]")
        te = tensors(d, te_idx, stats, dev)
        p = predict(name, model, te[0])
        oof[te_idx] = p.reshape(len(te_idx), ncol)
        best_eps.append(be)
        msg = []
        for j in range(ncol):
            e = ENDPOINTS[j] if name in MULTI else a.endpoint
            yt = d[e][te_idx]
            msg.append(f"{e} AUROC={roc_auc_score(yt, p.reshape(-1, ncol)[:, j]):.4f} AUPRC={average_precision_score(yt, p.reshape(-1, ncol)[:, j]):.4f}")
        log(f"fold{f} done ({(time.time() - t0) / 60:.1f} min) best_ep={be} TEST: " + " | ".join(msg))

    res = {"tag": tag, "model": name, "seed": a.seed, "cfg": cfg, "best_eps": best_eps}
    eps = ENDPOINTS if name in MULTI else [a.endpoint]
    done = ~np.isnan(oof[:, 0])
    for j, e in enumerate(eps):
        res[f"oof_{e}_auroc"] = float(roc_auc_score(d[e][done], oof[done, j]))
        res[f"oof_{e}_auprc"] = float(average_precision_score(d[e][done], oof[done, j]))
        log(f"OOF {e}: AUROC={res[f'oof_{e}_auroc']:.4f} AUPRC={res[f'oof_{e}_auprc']:.4f} (n={done.sum()})")
    save = {"oof": oof, "endpoints": np.array(eps)}

    if not a.no_external:
        ext = load("eicu")
        stats = fit_stats(d["x"])
        full = tensors(d, np.arange(N), stats, dev)
        E = int(np.median(best_eps))
        log(f"final full-cohort retrain for {E} epochs (median best epoch {best_eps}) -> frozen eICU")
        model, _, _ = train_one(name, cfg, full, None, ep_i, a.seed, fixed_epochs=E, prefix=f"[{tag} final]")
        pe = predict(name, model, tensors(ext, np.arange(len(ext["stay_id"])), stats, dev)[0])
        pe = pe.reshape(-1, ncol)
        for j, e in enumerate(eps):
            res[f"ext_{e}_auroc"] = float(roc_auc_score(ext[e], pe[:, j]))
            res[f"ext_{e}_auprc"] = float(average_precision_score(ext[e], pe[:, j]))
            log(f"eICU {e}: AUROC={res[f'ext_{e}_auroc']:.4f} AUPRC={res[f'ext_{e}_auprc']:.4f}")
        save["ext"] = pe
        torch.save(model.state_dict(), f"{ROOT}/results/preds/{tag}.pt")
    np.savez_compressed(f"{ROOT}/results/preds/{tag}.npz", **save)
    res["minutes"] = (time.time() - t0) / 60
    if a.n_folds_run == 5:
        with open(f"{ROOT}/results/runs.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")
    log(f"=== finished {tag} in {res['minutes']:.1f} min")


if __name__ == "__main__":
    main()
