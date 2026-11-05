"""Frozen-model evaluation on the Hallym cohort (no retraining, no recalibration).

Usage: python eval_hallym.py [--data hallym] group=tag1,tag2,... [group=...]
Each tag's frozen full-cohort model (results/preds/{tag}.pt) is rebuilt from its cfg in results/runs.jsonl,
standardised with full-MIMIC statistics exactly as for eICU, and applied to data/{data}_tensor.npz.
Tags in a group are averaged (ensemble). Reports AUROC / AUPRC / O:E per endpoint with patient-cluster bootstrap
95% CIs, per-hospital AUROC, and paired bootstrap differences against the first group.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

ROOT = T.ROOT
EPS = ["y24", "y48", "y168", "yhosp"]


def frozen_preds(tag, ext, stats, dev):
    runs = {}
    for fn in ("runs.jsonl", "runs_quick.jsonl"):               # runs_quick: CV-free quick checks (--final_epochs)
        if os.path.exists(f"{ROOT}/results/{fn}"):
            for l in open(f"{ROOT}/results/{fn}"):
                r = json.loads(l)
                runs[r["tag"]] = r
    r = runs[tag]
    cfg = json.loads(json.dumps(r["cfg"]))
    if str(cfg.get("data", "")).startswith("_dyn"):
        cfg.setdefault("enc", {}).setdefault("T", {"_dyn": 48, "_dyn7d": 168}[cfg["data"]])
    T.AVAIL = bool(cfg.get("avail", False)) or bool(cfg.get("regime"))
    if T.AVAIL:
        cfg.setdefault("enc", {})["in_dim"] = 88
    model = T.build(r["model"], cfg).to(dev)
    sd = torch.load(f"{ROOT}/results/preds/{tag}.pt", map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [{tag}] missing={missing[:4]} unexpected={unexpected[:4]}")
    X = T.tensors(ext, np.arange(len(ext["stay_id"])), stats, dev)[0]
    return T.predict(r["model"], model, X).reshape(len(X), -1)


def boot_idx(groups, B, rs):
    u, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(u))]
    for _ in range(B):
        pick = rs.randint(0, len(u), len(u))
        yield np.concatenate([members[k] for k in pick])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hallym")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("groups", nargs="+")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = T.load("mimic" + a.data[len("hallym"):] if a.data.startswith("hallym") else "mimic")   # matching MIMIC data
    stats = T.fit_stats(d["x"])
    ext = T.load(a.data)
    if "site" in ext:
        site = np.asarray(ext["site"]).astype(str)
    else:
        site = pd.read_parquet(f"{ROOT}/data/{a.data}_cohort.parquet").sort_values("stay_id")["hospital"].values
    subj = ext["subject_id"]
    print(f"{a.data}: N={len(subj)} " + " ".join(f"{e}={int(ext[e].sum())}" for e in EPS))
    P = {}
    for g in a.groups:
        name, tags = g.split("=")
        tags = [t for t in tags.split(",") if os.path.exists(f"{ROOT}/results/preds/{t}.pt")]
        P[name] = np.mean([frozen_preds(t, ext, stats, dev) for t in tags], 0)
        print(f"  {name}: {len(tags)} models")
    np.savez_compressed(f"{ROOT}/results/{a.data}_frozen_preds.npz", **P)
    rs = np.random.RandomState(0)
    idxs = list(boot_idx(subj, a.B, rs))
    names = list(P)
    for j, e in enumerate(EPS):
        y = ext[e]
        if y.sum() < 10:
            print(f"\n{e}: {int(y.sum())} events - not evaluated")
            continue
        print(f"\n{e} ({int(y.sum())} events)")
        bs = {n: [] for n in names}
        for ii in idxs:
            if y[ii].sum() == 0:
                continue
            for n in names:
                p = P[n][ii, j]
                bs[n].append((A(y[ii], p), AP(y[ii], p), y[ii].sum() / p.sum()))
        for n in names:
            p = P[n][:, j]
            b = np.array(bs[n])
            lo, hi = np.percentile(b, [2.5, 97.5], 0)
            ph = " ".join(f"{h[-3:]}={A(y[site == h], p[site == h]):.3f}" for h in np.unique(site) if 0 < y[site == h].sum() < (site == h).sum())
            line = (f"  {n:22s} AUROC {A(y, p):.3f} [{lo[0]:.3f},{hi[0]:.3f}]  AUPRC {AP(y, p):.3f} [{lo[1]:.3f},{hi[1]:.3f}]"
                    f"  O/E {y.sum() / p.sum():.2f} [{lo[2]:.2f},{hi[2]:.2f}] | {ph}")
            if n != names[0]:
                dd = b[:, :2] - np.array(bs[names[0]])[:, :2]
                l2, h2 = np.percentile(dd, [2.5, 97.5], 0)
                line += (f"\n  {'':22s} vs {names[0]}: dAUROC {A(y, p) - A(y, P[names[0]][:, j]):+.3f} [{l2[0]:+.3f},{h2[0]:+.3f}]"
                         f"  dAUPRC {AP(y, p) - AP(y, P[names[0]][:, j]):+.3f} [{l2[1]:+.3f},{h2[1]:+.3f}]")
            print(line)


if __name__ == "__main__":
    main()
