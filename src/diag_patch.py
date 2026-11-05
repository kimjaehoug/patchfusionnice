"""Why does patching hurt on eICU? Diagnostics on frozen full-cohort models (no retraining).

D1 recency: occlude the last 3 h (hours 9-11 replaced by hour-8 values, raw masks 0) -> drop in eICU AUPRC.
D2 pooling: re-pool the encoder's final tokens (all / real only / last real token / imagined only) with the frozen head.
Usage: python diag_patch.py [--data eicu] [--ep y24] group=tags ...
"""
import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
import models as M
from eval_hallym import frozen_preds

EPS = ["y24", "y48", "y168", "yhosp"]


def load_model(tag, dev):
    import json
    runs = {json.loads(l)["tag"]: json.loads(l) for fn in ("runs.jsonl", "runs_quick.jsonl")
            for l in open(f"{T.ROOT}/results/{fn}")}
    r = runs[tag]
    cfg = json.loads(json.dumps(r["cfg"]))
    if str(cfg.get("data", "")).startswith("_dyn"):
        cfg.setdefault("enc", {}).setdefault("T", {"_dyn": 48, "_dyn7d": 168}[cfg["data"]])
    T.AVAIL = bool(cfg.get("avail")) or bool(cfg.get("regime"))
    if T.AVAIL:
        cfg.setdefault("enc", {})["in_dim"] = 88
    m = T.build(r["model"], cfg).to(dev)
    m.load_state_dict(torch.load(f"{T.ROOT}/results/preds/{tag}.pt", map_location=dev), strict=False)
    return m.eval()


@torch.no_grad()
def pooled(model, X, bs=2048):
    """returns dict pooling -> CIF [N,4]"""
    K = model.enc.K
    res = {}
    for i in range(0, len(X), bs):
        z = model.enc(X[i:i + bs])
        pools = {"all": z.mean(1), "last_tok": z[:, K - 1], "first_tok": z[:, 0]}
        if z.shape[1] >= 2 * K:
            pools["real_only"] = z[:, :K].mean(1)
            pools["imag_only"] = z[:, K:2 * K].mean(1)
        for k, v in pools.items():
            lg = model.head.out(v).view(len(v), model.head.n_int, 3)
            lg[:, -1, 2] = -1e4
            res.setdefault(k, []).append(M.cr_probs(lg)[1].cpu())
    return {k: torch.cat(v).numpy() for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eicu")
    ap.add_argument("--ep", default="y24")
    ap.add_argument("groups", nargs="+")
    a = ap.parse_args()
    dev = torch.device("cuda")
    d = T.load("mimic")
    stats = T.fit_stats(d["x"])
    ext = T.load(a.data)
    j = EPS.index(a.ep)
    y = ext[a.ep]
    xo, mo = ext["x"].copy(), ext["m"].copy()
    xo[:, 9:12] = xo[:, 8:9]
    mo[:, 9:12] = 0
    occ = dict(ext, x=xo, m=mo)
    print(f"{a.data} {a.ep}: N={len(y)} events={int(y.sum())}")
    for g in a.groups:
        name, tags = g.split("=")
        tags = tags.split(",")
        full = np.mean([frozen_preds(t, ext, stats, dev)[:, j] for t in tags], 0)
        o = np.mean([frozen_preds(t, occ, stats, dev)[:, j] for t in tags], 0)
        pools = {}
        for t in tags:
            m = load_model(t, dev)
            X = T.tensors(ext, np.arange(len(y)), stats, dev)[0]
            for k, v in pooled(m, X).items():
                pools.setdefault(k, []).append(v[:, j])
            del X
            torch.cuda.empty_cache()
        print(f"\n{name}: full AUROC/AUPRC {A(y, full):.3f}/{AP(y, full):.3f} | last-3h occluded {A(y, o):.3f}/{AP(y, o):.3f}"
              f"  (dAUPRC {AP(y, o) - AP(y, full):+.3f})")
        print("   re-pool: " + "  ".join(f"{k} {A(y, np.mean(v, 0)):.3f}/{AP(y, np.mean(v, 0)):.3f}" for k, v in pools.items()))


if __name__ == "__main__":
    main()
