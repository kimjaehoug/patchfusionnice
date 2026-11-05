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
import regime as RG

ROOT = os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENDPOINTS = ["y24", "y48", "y168", "yhosp"]
CUTS = (24.0, 48.0, 168.0)   # interval edges after the hour-12 landmark


LOG = None
ANCHOR_Z = None    # (lo, hi) clinical critical values on the current standardised scale (obsb encoder)
IND_B = None       # per-variable evidence bounds for the bounded indicator pathway
AVAIL = False   # append value-availability flags (observed or carried, vs never available) as 22 extra channels


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


def tensors(d, idx, stats, dev, xm=None, desc=None):
    """xm: optional (x, m) replacing d's values/masks for idx (simulated regimes); desc: per-sample site descriptor."""
    x, m = xm if xm is not None else (d["x"][idx], d["m"][idx])
    v, dv = make_input(x, stats)
    parts = [v, dv, m]
    if AVAIL:
        parts.append((~np.isnan(x)).astype(np.float32))
    if desc is not None:
        parts.append(np.repeat(desc[:, None, :], v.shape[1], 1))
    X = np.concatenate(parts, -1).astype(np.float32)
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


MULTI = {"mtpf", "hpf", "hpf_noq", "hpf_w", "hcr"}
CR = ("hpf", "hpf_noq", "hpf_w", "hcr")


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
    if name == "hcr":                                     # reference encoder + competing-risk head
        return M.HorizonBase(cfg["base"], **kw)
    if name == "hpf_noq":
        return M.HorizonPF(queries=False, concat=cfg.get("concat", False), **kw)
    raise ValueError(name)


def loss_fn(name, out, Y, interval, etype, ep_i, cfg):
    if name in CR:
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
        if name in CR:
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


def train_one(name, cfg, tr, va, ep_i, seed, max_epochs=100, patience=10, fixed_epochs=None, prefix="", tr_fn=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    DEVM = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # training tensors may live on the CPU
    model = build(name, cfg).to(DEVM)
    enc = getattr(model, "enc", None)
    if enc is not None and getattr(enc, "ple_bins", 0):     # bin edges from this training split only
        enc.set_edges(tr[0][..., :22], tr[0][..., 44:66])
    if enc is not None and getattr(enc, "obsb", False):
        enc.set_anchors(*ANCHOR_Z)
        if hasattr(model, "ind_b"):
            model.ind_b.copy_(torch.as_tensor(IND_B))
    if enc is not None and getattr(enc, "ta_cuts", 0):                # learned-abstraction cut init, training split only
        enc.set_cuts(tr[0][..., :22], tr[0][..., 44:66])
    lr, wd = cfg.get("lr", 1e-3), cfg.get("wd", 1e-3)
    ta = [p for n, p in model.named_parameters() if ".ta_" in n]
    if ta and cfg.get("ta_lr_mult"):                          # abstraction cut points: no decay, faster lr
        rest = [p for n, p in model.named_parameters() if ".ta_" not in n]
        opt = torch.optim.Adam([{"params": rest, "lr": lr, "weight_decay": wd},
                                {"params": ta, "lr": lr * cfg["ta_lr_mult"], "weight_decay": 0.0}])
    elif cfg.get("enc", {}).get("dyn_mode") == "boost":      # dynamic-patch detector: 10x learning rate
        det = [p for n, p in model.named_parameters() if ".dyn.det." in n]
        rest = [p for n, p in model.named_parameters() if ".dyn.det." not in n]
        opt = torch.optim.Adam([{"params": rest, "lr": lr, "weight_decay": wd},
                                {"params": det, "lr": lr * 10, "weight_decay": 0.0}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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
        if tr_fn is not None:                             # fresh simulated observation regimes every epoch
            tr = tr_fn()
        model.train()
        perm = torch.randperm(n, generator=g).to(tr[0].device)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if cfg.get("enc", {}).get("twoview"):               # factual + counterfactual view of the same patients
                b = torch.cat([b, b])
            tb = [t[b].to(DEVM, non_blocking=True) for t in tr]  # batch to the GPU
            out = model(tb[0])
            if cfg.get("prism"):                          # worst-of-K sampling-pattern variations (PRISM-style)
                losses = [M.cr_nll(out, tb[2], tb[3], reduce=True)]
                for _ in range(int(cfg["prism"]) - 1):
                    losses.append(M.cr_nll(model(M.prism_view(tb[0])), tb[2], tb[3]))
                loss = torch.stack(losses).max()
            elif len(tr) == 5:                            # importance weights (reweighting baseline)
                loss = M.cr_nll(out, tb[2], tb[3], sw=tb[4])
            else:
                loss = loss_fn(name, out, tb[1], tb[2], tb[3], ep_i, cfg)
            il = getattr(getattr(model, "enc", None), "imag_loss", None)
            if il is not None:
                w = model.enc.imag2 * min(1.0, ep / 3) if model.enc.imag2 else model.enc.imagine
                loss = loss + w * il
            dyn = getattr(getattr(model, "enc", None), "dyn", None)
            if dyn is not None and dyn.inv_loss is not None and cfg["enc"].get("inv_w", 0):
                loss = loss + cfg["enc"]["inv_w"] * dyn.inv_loss    # observation-invariant handoff
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if hasattr(getattr(model, "enc", None), "ema_update"):
                model.enc.ema_update()
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




def reweight_tensors(d, idx, RW, stats, dev):
    """Subbaswamy-style reweighting towards a target observation process: each development window gets
    w = prod_labs [p_t/p_d]^tested [(1-p_t)/(1-p_d)]^(1-tested) * prod_docs [rho_t/rho_d]^doc [(1-rho_t)/(1-rho_d)]^(1-doc),
    with window-level selection probabilities from the label-free profiles (pooled over target hospitals), clipped and
    normalised to mean 1."""
    import obsprofile as PR
    acu = PR.BlockAcuity().fit(d["x"][idx], d["yhosp"][idx])
    devp = PR.estimate(d["x"][idx], d["m"][idx], acu)
    if RW["target"].endswith(":sites"):
        tx = np.concatenate([sx for sx, _ in PR.target_sites(RW["target"])]); tm = np.concatenate([sm for _, sm in PR.target_sites(RW["target"])])
    else:
        tx, tm = PR.target_samples(RW["target"])
    tgtp = PR.fit_window(PR.estimate(tx, tm, acu, beta=devp.beta, n=len(tx)), devp)
    x, m = d["x"][idx], d["m"][idx]
    with np.errstate(all="ignore"):
        aw = np.nan_to_num(np.nanmean(acu(x), 1), nan=0.0)
    logw = np.zeros(len(idx))
    bm = PR._block_measured(m)
    for c in PR.LAB:
        pd_ = 1 / (1 + np.exp(-(devp.beta_w[c] * aw + devp.alpha_w[c])))
        pt_ = 1 / (1 + np.exp(-(devp.beta_w[c] * aw + tgtp.alpha_w[c])))
        t = bm[..., c].any(1)
        logw += np.where(t, np.log(pt_ + 1e-6) - np.log(pd_ + 1e-6), np.log(1 - pt_ + 1e-6) - np.log(1 - pd_ + 1e-6))
    for c in PR.DOC:                                       # absent streams cannot be reweighted (no support): skipped
        if devp.rho[c] > 0 and tgtp.rho[c] >= 0.05:
            t = (m[:, :, c] > 0).any(1)
            rt, rd = np.clip(tgtp.rho[c], 1e-3, 1 - 1e-3), np.clip(devp.rho[c], 1e-3, 1 - 1e-3)
            logw += np.where(t, np.log(rt / rd), np.log((1 - rt) / (1 - rd)))
    w = np.exp(np.clip(logw, np.log(0.1), np.log(10.0)))
    w = w / w.mean()
    log(f"reweighting to {RW['target']}: weight quantiles " + " ".join(f"{q:.2f}" for q in np.quantile(w, [0.01, 0.25, 0.5, 0.75, 0.99])))
    return tensors(d, idx, stats, torch.device("cpu")) + (torch.tensor(w, dtype=torch.float32),)

def profile_trfn(d, idx, PROF, rs_p, stats, dev):
    """profile-matched observation imagination: development acuity and profile from d[idx]; target profiles from
    unlabeled target data; returns a function producing factual + imagined training tensors (fresh each call)."""
    import obsprofile as PR
    if PROF.get("simple"):                                   # comparator: rate-matched thinning per target hospital
        r_dev = PR.obs_rate(d["m"][idx])
        r_sites = [PR.obs_rate(sm) for sx, sm in PR.target_sites(PROF["target"])]
        log(f"simple thinning: {len(r_sites)} site rate profiles from {PROF['target']}")

        def tr_fn_simple():
            perm = rs_p.permutation(len(idx))
            x, m = d["x"][idx][perm], d["m"][idx][perm]
            xs, ms = np.empty_like(x), np.empty_like(m)
            for i in range(0, len(x), 2000):
                r_t = r_sites[rs_p.randint(len(r_sites))].copy()
                if PROF.get("keep_vitals"):                  # smoke: vital-sign rates left at development level
                    r_t[PR.VIT] = r_dev[PR.VIT]
                fn_ = PR.imagine_naive if PROF.get("naive") else PR.imagine_rate     # naive: re-masking without re-imputation
                xs[i:i + 2000], ms[i:i + 2000] = fn_(x[i:i + 2000], m[i:i + 2000], r_dev, r_t, rs_p)
            inv = np.argsort(perm)
            cpu = torch.device("cpu")
            f, c = tensors(d, idx, stats, cpu), tensors(d, idx, stats, cpu, xm=(xs[inv], ms[inv]))
            return tuple(torch.cat([u, v]) for u, v in zip(f, c))
        return tr_fn_simple
    acu = PR.BlockAcuity().fit(d["x"][idx], d["yhosp"][idx])
    devp = PR.estimate(d["x"][idx], d["m"][idx], acu)
    PR.RAND_VITALS_FIXED = bool(PROF.get("fix_rand_vitals"))
    PR.FREE_SLOPE = bool(PROF.get("free_slope"))
    PR.CAD_HOURLY = bool(PROF.get("cad_hourly"))
    PR.KEEP_ONE = not PROF.get("no_keep_one")
    if PROF["target"].endswith(":sites"):                  # one profile per hospital of the target source
        tgtp = [PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp) for sx, sm in PR.target_sites(PROF["target"])]
        log(f"{len(tgtp)} site profiles from {PROF['target']}")
    else:
        tx, tm = PR.target_samples(PROF["target"])
        tgtp = PR.estimate(tx, tm, acu, beta=devp.beta)
        del tx, tm
        if PROF.get("hour_only"):                         # isolate the hour scale: no documentation / site changes
            tgtp.rho, tgtp.absent = devp.rho.copy(), []
        PR.fit_window(tgtp, devp)
    log(f"profile dev: {devp.summary()}")
    mode, scheme, parts = PROF.get("mode", "patch"), PROF.get("scheme", ""), PROF.get("parts")

    def tr_fn():
        perm = rs_p.permutation(len(idx))                 # chunks of random patients get fresh profiles
        if parts and scheme:
            xs, ms = PR.imagine_mixed(d["x"][idx][perm], d["m"][idx][perm], acu, devp, tgtp, scheme=scheme, rng=rs_p,
                                      parts=parts, tfrac=PROF.get("tfrac", 0.5))
            inv = np.argsort(perm)
            xs, ms = xs[inv], ms[inv]
        elif parts:
            xs, ms = PR.imagine_two(d["x"][idx], d["m"][idx], acu, devp, tgtp, parts=parts, rng=rs_p)
        elif scheme:
            xs, ms = PR.imagine_mixed(d["x"][idx][perm], d["m"][idx][perm], acu, devp, tgtp, mode=mode, scheme=scheme, rng=rs_p)
            inv = np.argsort(perm)
            xs, ms = xs[inv], ms[inv]
        else:
            xs, ms = PR.imagine(d["x"][idx], d["m"][idx], acu, devp, tgtp, mode=mode, rng=rs_p)
        cpu = torch.device("cpu")                         # full doubled training set stays in host memory
        f, c = tensors(d, idx, stats, cpu), tensors(d, idx, stats, cpu, xm=(xs, ms))
        return tuple(torch.cat([u, v]) for u, v in zip(f, c))   # factual + imagined views
    return tr_fn

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
    ap.add_argument("--final_epochs", type=int, default=0, help="quick check: skip CV, train the frozen model for this many epochs")
    ap.add_argument("--sub_frac", type=float, default=1.0, help="quick check: train the frozen model on this fraction of patients")
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    global AVAIL
    AVAIL = bool(cfg.get("avail", False)) or bool(cfg.get("regime"))
    if AVAIL:
        cfg.setdefault("enc", {})["in_dim"] = 88
    DATA = cfg.get("data", "")                           # "" = hour-12 landmark data, "_dyn" = dynamic landmarks, 48-h look-back
    if DATA.startswith("_dyn"):                           # look-back length of the dynamic data
        cfg.setdefault("enc", {}).setdefault("T", {"_dyn": 48, "_dyn7d": 168}[DATA])
    OBSB = bool(cfg.get("enc", {}).get("obsb"))
    if OBSB:
        AVAIL = True
        cfg["enc"]["in_dim"] = 88
    global ANCHOR_Z, IND_B
    REG = cfg.get("regime")                              # None | "sim" | "sim+desc"
    if REG == "sim+desc":
        cfg.setdefault("enc", {})["ctx_dim"] = RG.DESC_DIM
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

    d = load("mimic" + DATA)
    N = len(d["stay_id"])
    groups = d["subject_id"]
    rng = np.random.RandomState(0)                       # identical folds for every model / seed
    order = rng.permutation(N)
    folds = list(GroupKFold(5).split(order, groups=groups[order]))
    ncol = 4 if name in MULTI else 1
    oof = np.full((N, ncol), np.nan, dtype=np.float32)
    best_eps = []
    if a.final_epochs:
        best_eps = [a.final_epochs]
    for f, (tr_o, te_o) in enumerate([] if a.final_epochs else folds[:a.n_folds_run]):
        tr_idx, te_idx = order[tr_o], order[te_o]
        r = np.random.RandomState(100 + f)
        subj = np.unique(groups[tr_idx])
        va_subj = r.choice(subj, int(0.2 * len(subj)), replace=False)
        is_va = np.isin(groups[tr_idx], va_subj)
        in_tr, in_va = tr_idx[~is_va], tr_idx[is_va]
        stats = fit_stats(d["x"][in_tr])
        if OBSB:
            ANCHOR_Z = RG.anchors_z(*stats)
            IND_B = RG.evidence_bounds(d["x"][in_tr], d["m"][in_tr], d["yhosp"][in_tr])
            log(f"fold{f} evidence bounds: " + " ".join(f"{v}={b:.2f}" for v, b in zip(RG.VARS, IND_B) if b > 0))
        log(f"fold{f}: train={len(in_tr)} val={len(in_va)} test={len(te_idx)} "
            f"test_events=" + ",".join(f"{e}:{int(d[e][te_idx].sum())}" for e in ENDPOINTS))
        tr_fn, dsc = None, (lambda idx: None)
        if REG:
            acu = RG.Acuity().fit(d["x"][in_tr], d["yhosp"][in_tr])        # vitals-only, training split labels
            ref = RG.measured(d["m"][in_tr]).mean(0)
            acu_tr, rs = acu(d["x"][in_tr]), np.random.RandomState(1000 * a.seed + f)
            use_desc = REG == "sim+desc"

            def tr_fn(in_tr=in_tr, acu_tr=acu_tr, ref=ref, rs=rs, stats=stats, use_desc=use_desc):
                xs, ms, ds = RG.simulate(d["x"][in_tr], d["m"][in_tr], acu_tr, ref, rs)
                return tensors(d, in_tr, stats, dev, xm=(xs, ms), desc=ds if use_desc else None)

            def dsc(idx, acu=acu, ref=ref, use_desc=use_desc):              # native MIMIC regime, label-free
                return np.tile(RG.descriptor(RG.measured(d["m"][idx]), acu(d["x"][idx]), ref), (len(idx), 1)) if use_desc else None
        if cfg.get("profile"):
            tr_fn = profile_trfn(d, in_tr, cfg["profile"], np.random.RandomState(1000 * a.seed + f), stats, dev)
        if cfg.get("reweight"):
            _rw = reweight_tensors(d, in_tr, cfg["reweight"], stats, dev)
            tr_fn = lambda _rw=_rw: _rw
        model, be, bs_ = train_one(name, cfg, tr_fn() if tr_fn else tensors(d, in_tr, stats, dev),
                                   tensors(d, in_va, stats, dev, desc=dsc(in_va)),
                                   ep_i, a.seed, max_epochs=a.max_epochs, patience=cfg.get("patience", 10), prefix=f"[{tag} f{f}]", tr_fn=tr_fn)
        te = tensors(d, te_idx, stats, dev, desc=dsc(te_idx))
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
    for j, e in enumerate(eps if done.any() else []):
        res[f"oof_{e}_auroc"] = float(roc_auc_score(d[e][done], oof[done, j]))
        res[f"oof_{e}_auprc"] = float(average_precision_score(d[e][done], oof[done, j]))
        log(f"OOF {e}: AUROC={res[f'oof_{e}_auroc']:.4f} AUPRC={res[f'oof_{e}_auprc']:.4f} (n={done.sum()})")
    save = {"oof": oof, "endpoints": np.array(eps)}

    if not a.no_external:
        ext = load("eicu" + DATA)
        stats = fit_stats(d["x"])
        if OBSB:
            ANCHOR_Z = RG.anchors_z(*stats)
            IND_B = RG.evidence_bounds(d["x"], d["m"], d["yhosp"])
        E = int(np.median(best_eps))
        log(f"final full-cohort retrain for {E} epochs (median best epoch {best_eps}) -> frozen eICU")
        tr_fn, ext_desc = None, None
        if REG:
            acu = RG.Acuity().fit(d["x"], d["yhosp"])
            ref = RG.measured(d["m"]).mean(0)
            acu_all, rs = acu(d["x"]), np.random.RandomState(1000 * a.seed + 99)
            use_desc = REG == "sim+desc"

            def tr_fn():
                xs, ms, ds = RG.simulate(d["x"], d["m"], acu_all, ref, rs)
                return tensors(d, np.arange(N), stats, dev, xm=(xs, ms), desc=ds if use_desc else None)
            if use_desc:                                   # per-hospital descriptor from unlabeled eICU data
                import pandas as pd
                site = ext["site"] if "site" in ext else \
                    pd.read_parquet(f"{ROOT}/data/eicu_cohort.parquet").sort_values("stay_id")["hospitalid"].values
                ext_desc = RG.site_descriptors(ext["m"], acu(ext["x"]), site, ref)
        all_idx = np.arange(N)
        if a.sub_frac < 1.0:                               # quick check: random subset of patients
            us = np.unique(groups)
            keep = np.random.RandomState(7).choice(us, int(a.sub_frac * len(us)), replace=False)
            all_idx = np.where(np.isin(groups, keep))[0]
            log(f"sub_frac {a.sub_frac}: training on {len(all_idx)} of {N}")
        if cfg.get("profile"):                             # profile-matched observation imagination (label-free target)
            tr_fn = profile_trfn(d, all_idx, cfg["profile"], np.random.RandomState(1000 * a.seed + 7), stats, dev)
        if cfg.get("reweight"):
            _rw = reweight_tensors(d, all_idx, cfg["reweight"], stats, dev)
            tr_fn = lambda _rw=_rw: _rw
        full = tr_fn() if tr_fn else tensors(d, all_idx, stats, dev)
        model, _, _ = train_one(name, cfg, full, None, ep_i, a.seed, fixed_epochs=E, prefix=f"[{tag} final]", tr_fn=tr_fn)
        pe = predict(name, model, tensors(ext, np.arange(len(ext["stay_id"])), stats, dev, desc=ext_desc)[0])
        pe = pe.reshape(-1, ncol)
        for j, e in enumerate(eps):
            res[f"ext_{e}_auroc"] = float(roc_auc_score(ext[e], pe[:, j]))
            res[f"ext_{e}_auprc"] = float(average_precision_score(ext[e], pe[:, j]))
            log(f"eICU {e}: AUROC={res[f'ext_{e}_auroc']:.4f} AUPRC={res[f'ext_{e}_auprc']:.4f}")
        save["ext"] = pe
        torch.save(model.state_dict(), f"{ROOT}/results/preds/{tag}.pt")
    np.savez_compressed(f"{ROOT}/results/preds/{tag}.npz", **save)
    res["minutes"] = (time.time() - t0) / 60
    if a.n_folds_run == 5 and not a.final_epochs:
        with open(f"{ROOT}/results/runs.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")
    log(f"=== finished {tag} in {res['minutes']:.1f} min")


if __name__ == "__main__":
    main()
