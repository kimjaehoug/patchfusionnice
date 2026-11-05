"""Summarise the formal runs (three-seed ensembles) for the manuscript: eICU-CRD all hospitals and low-testing tertile,
Hallym four hospitals, development fold-0 OOF. Writes results/formal_summary.json.

Usage: python formal_summary.py            (transport study)
       python formal_summary.py --enc      (encoder design study -> results/encoder_summary.json)
"""
from bootstrap_par import boot_metric
import os
import re
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from eval_hallym import frozen_preds, boot_idx

ROOT = T.ROOT
SEEDS = [2, 3, 4, 5, 6]
GROUPS = {"base": "F_base", "generic": "F_gen", "rw_H": "F_rwH", "rw_E": "F_rwE",
          "prop_H": "F_propH", "prop_E": "F_propE", "sel_H": "F_selH", "sel_E": "F_selE"}
ENC = {"patchfusion": "F_base", "no_reset": "E_noreset", "reversal": "E_rev", "p_cnn": "E_pcnn", "p_mlp": "E_pmlp",
       "p_linear": "E_plinear", "p_mean": "E_pmean", "p_shuffle": "E_pshuffle", "lstm": "E_lstm",
       "transformer": "E_transformer", "patchtst": "E_patchtst", "grud": "E_grud", "lgtrl": "E_lgtrl",
       "hipatch": "E_hipatch", "tbal": "E_tbal", "raindrop": "E_raindrop", "lightgbm": "F_lgbm"}
EPS = ["y24", "y48", "y168", "yhosp"]
# --wide: final model with the wider recurrent patch encoder (LSTM 256); comparators keep their own sizes
if "--wide" in sys.argv:
    GROUPS = {g: p.replace("F_", "W_") for g, p in GROUPS.items()}
    GROUPS["generic"] = "W_gen"
    ENC = {**ENC, "patchfusion": "W_base", "no_reset": "W_noreset", "reversal": "WE_rev", "p_cnn": "WE_pcnn", "p_mlp": "WE_pmlp",
           "p_linear": "WE_plinear", "p_mean": "WE_pmean", "p_shuffle": "WE_pshuffle"}


ENC_SEEDS = [2, 3, 4]                                   # encoder study: three seeds for every encoder, incl. PatchFusion


def tags(pfx, seeds=None):
    return [f"{pfx}_s{s}" for s in (seeds or SEEDS) if os.path.exists(f"{ROOT}/results/preds/{pfx}_s{s}.npz")]


def main():
    enc = "--enc" in sys.argv
    groups = ENC if enc else GROUPS
    e = T.load("eicu_dyn")
    site = e["site"]
    crit = [7, 13, 14, 16, 20, 21]                     # lactate ph be ne uo gcs
    hr = pd.Series(e["m"][:, :, crit].mean((1, 2))).groupby(site).transform("mean").values
    big = pd.Series(1, index=site).groupby(level=0).transform("size").values >= 200
    tert = np.full(len(hr), None, object)
    tert[big] = pd.qcut(pd.Series(hr[big]), 3, labels=["low", "mid", "high"]).values
    low = tert == "low"
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    h = T.load("hallym_dyn")
    out, keep = {}, {}
    for g, pfx in groups.items():
        tg = tags(pfx, ENC_SEEDS if enc else None)
        if not tg:
            continue
        r = {"n_seeds": len(tg)}
        pe = np.mean([np.load(f"{ROOT}/results/preds/{t}.npz")["ext"] for t in tg], 0)
        for j, ep in enumerate(EPS):
            y = e[ep]
            r[f"eicu_{ep}"] = [A(y, pe[:, j]), AP(y, pe[:, j]), y.sum() / pe[:, j].sum()]
            r[f"eicu_low_{ep}"] = [A(y[low], pe[low, j]), AP(y[low], pe[low, j]), y[low].sum() / pe[low, j].sum()]
        ph = np.mean([np.load(f"{ROOT}/results/preds/{t}.npz")["hal"] if pfx == "F_lgbm" else frozen_preds(t, h, stats, "cuda")
                      for t in tg], 0)
        for j, ep in enumerate(EPS):
            y = h[ep]
            r[f"hallym_{ep}"] = [A(y, ph[:, j]), AP(y, ph[:, j]), y.sum() / ph[:, j].sum()]
        oof = {ep: [] for ep in EPS}
        for t in tg:
            if pfx == "F_lgbm":
                r_ = [json.loads(l) for l in open(f"{ROOT}/results/runs.jsonl") if json.loads(l)["tag"] == t][-1]
                for ep in EPS:
                    oof[ep].append((r_[f"oof_{ep}_auroc"], r_[f"oof_{ep}_auprc"]))
                continue
            txt = open(f"{ROOT}/logs/train/{t}.log").read()
            for ep, a_, b_ in re.findall(r"OOF (y\w+): AUROC=([\d.]+) AUPRC=([\d.]+)", txt):
                oof[ep].append((float(a_), float(b_)))
        po_ = [np.load(f"{ROOT}/results/preds/{t}.npz") for t in tg]
        if all("oof" in z.files for z in po_):          # seed ensemble on the held-out fold (as for eICU-CRD and Hallym)
            po_ = np.nanmean([z["oof"] for z in po_], 0)
            ok_ = ~np.isnan(po_[:, 0])
            for j, ep in enumerate(EPS):
                yo = d[ep][ok_]
                r[f"oof_{ep}"] = [A(yo, po_[ok_, j]), AP(yo, po_[ok_, j])]
                r[f"oof_{ep}_seedmean"] = list(np.mean(oof[ep], 0)) if oof[ep] else None
        else:
            for ep in EPS:
                if oof[ep]:
                    r[f"oof_{ep}"] = list(np.mean(oof[ep], 0))
        out[g] = r
        if enc:
            po = np.nanmean([np.load(f"{ROOT}/results/preds/{t}.npz")["oof"] for t in tg], 0)
            keep[g] = {"mimic": po, "eicu": pe, "hallym": ph}
        print(g, len(tg), " ".join(f"{k}={v[0]:.3f}/{v[1]:.3f}" for k, v in r.items() if k.endswith("y24")))
        torch.cuda.empty_cache()
    if enc and "patchfusion" in keep:                  # paired cluster bootstrap of differences to PatchFusion
        te = ~np.isnan(keep["patchfusion"]["mimic"][:, 0])
        sets = {"mimic": (d, te, d["subject_id"][te]), "eicu": (e, np.ones(len(site), bool), site),
                "hallym": (h, np.ones(len(h["y24"]), bool), h["subject_id"])}
        for dn, (dd, msk, cl) in sets.items():
            idxs = list(boot_idx(cl, 500, np.random.RandomState(0)))
            for ep in ("y24", "yhosp"):
                j = EPS.index(ep)
                y = dd[ep][msk]
                idx_ok = [i for i in idxs if 0 < y[i].sum() < len(i)]
                ref = keep["patchfusion"][dn][msk, j]
                for k, fnm in (("auroc", A), ("auprc", AP)):
                    rb = boot_metric(fnm, y, ref, idx_ok)
                    for g in keep:
                        if g == "patchfusion":
                            continue
                        p = keep[g][dn][msk, j]
                        bs = boot_metric(fnm, y, p, idx_ok) - rb
                        out[g][f"d_{dn}_{ep}_{k}"] = [fnm(y, p) - fnm(y, ref), *np.percentile(bs, [2.5, 97.5])]
            print("bootstrap", dn, "done")
    fn = ("encoder_summary" if enc else "formal_summary") + ("_W" if "--wide" in sys.argv else "") + ".json"
    json.dump(out, open(f"{ROOT}/results/{fn}", "w"), indent=1)
    print(f"written results/{fn}")


if __name__ == "__main__":
    main()
