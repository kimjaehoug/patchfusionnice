"""Path-specific sensitivity to observation-process perturbations on a fixed eICU subsample (frozen models).

Paths (each perturbs only the observation process, never the underlying physiology):
  temporal  - early observations (hours 0-8) of sparse variables removed and re-imputed: stale/absent evidence that a
              recurrent state could carry forward to the landmark;
  selective - testing threshold raised for labs: tests kept only in the most acute 50% of tested patients;
  capture   - documentation gaps: GCS / urine output / norepinephrine observations removed at random for 50% of patients.
Reports mean |Δlogit| (yhosp, y24) against the unperturbed input and AUROC under each perturbation.

Usage: python path_sensitivity.py tag1 tag2 ...
"""
import os
import sys
import json
import numpy as np
import torch

sys.argv, TAGS = sys.argv[:1], sys.argv[1:]
import train as T
import regime as RG
from sklearn.metrics import roc_auc_score

ROOT = T.ROOT
LABS = [RG.VARS.index(v) for v in ["bun", "lactate", "wbc", "hb", "na", "k", "cl", "ph", "be", "ag"]]
CAPT = [RG.VARS.index(v) for v in ["gcs", "uo", "ne"]]


def perturb(x, m, acu, kind, rng):
    raw = np.where(m > 0, x, np.nan)
    if kind == "temporal":
        sp = [c for c in range(22) if c not in RG.VIT]
        raw[:, :9, sp] = np.nan
    elif kind == "selective":
        for c in LABS:
            t = m[:, :, c].sum(1) > 0
            if t.sum() < 10:
                continue
            cut = np.quantile(acu[t], 0.5)
            drop = t & (acu < cut)
            raw[drop, :, c] = np.nan
    elif kind == "capture":
        for c in CAPT:
            drop = rng.rand(len(x)) < 0.5
            raw[drop, :, c] = np.nan
    return RG.refill(raw)


def main():
    d, e = T.load("mimic"), T.load("eicu")
    stats = T.fit_stats(d["x"])
    acu_m = RG.Acuity().fit(d["x"], d["yhosp"])
    sub = np.random.RandomState(0).choice(len(e["x"]), 30000, replace=False)
    ed = {k: e[k][sub] for k in T.ENDPOINTS + ["t_event", "event", "x", "m"]}
    acu = acu_m(ed["x"])
    views = {"native": (ed["x"], ed["m"])}
    for k in ["temporal", "selective", "capture"]:
        views[k] = perturb(ed["x"], ed["m"], acu, k, np.random.RandomState(1))
    runs = {json.loads(l)["tag"]: json.loads(l) for l in open(f"{ROOT}/results/runs.jsonl")}
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    for tag in TAGS:
        cfg = json.loads(json.dumps(runs[tag]["cfg"]))
        enc = cfg.setdefault("enc", {})
        T.AVAIL = bool(cfg.get("avail")) or bool(cfg.get("regime")) or bool(enc.get("obsb"))
        if T.AVAIL:
            enc["in_dim"] = 88
        if cfg.get("regime") == "sim+desc":
            enc["ctx_dim"] = RG.DESC_DIM
        model = T.build(runs[tag]["model"], cfg)
        model.load_state_dict(torch.load(f"{ROOT}/results/preds/{tag}.pt", map_location="cpu"))
        model.eval()
        desc = None
        if cfg.get("regime") == "sim+desc":
            desc = np.tile(RG.descriptor(RG.measured(d["m"]), acu_m(d["x"]), RG.measured(d["m"]).mean(0)), (len(sub), 1))
        P = {k: T.predict(runs[tag]["model"], model, T.tensors(ed, np.arange(len(sub)), stats, torch.device("cpu"), xm=v, desc=desc)[0])
             for k, v in views.items()}
        out = []
        for k in ["temporal", "selective", "capture"]:
            dh = np.abs(lg(P[k][:, 3]) - lg(P["native"][:, 3])).mean()
            d24 = np.abs(lg(P[k][:, 0]) - lg(P["native"][:, 0])).mean()
            out.append(f"{k} {dh:.3f}/{d24:.3f} (AUROC {roc_auc_score(ed['yhosp'], P[k][:, 3]):.3f})")
        print(f"{tag:16s} native AUROC {roc_auc_score(ed['yhosp'], P['native'][:, 3]):.3f} | mean|Δlogit| yhosp/y24: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
