"""LightGBM on the dynamic 48-h landmark data (flattened 48x88 input), one binary model per endpoint.
Same fold-0 split / inner validation / full-cohort retrain as train.py --n_folds_run 1; predictions on eICU-CRD and Hallym.

Usage: python train_lgbm_dyn.py --seed 2
"""
import json
import time
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

import train as T

T.AVAIL = True


def flat(d, idx, stats):
    v, dv = T.make_input(d["x"][idx], stats)
    return np.concatenate([v, dv, d["m"][idx], (~np.isnan(d["x"][idx])).astype(np.float32)], -1).reshape(len(idx), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    tag = f"F_lgbm_s{a.seed}"
    T.LOG = f"{T.ROOT}/logs/train/{tag}.log"
    P = dict(objective="binary", learning_rate=0.05, num_leaves=63, max_depth=6, min_data_in_leaf=50,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=16, seed=a.seed)
    d = T.load("mimic_dyn")
    N = len(d["stay_id"])
    groups = d["subject_id"]
    order = np.random.RandomState(0).permutation(N)
    tr_o, te_o = list(GroupKFold(5).split(order, groups=groups[order]))[0]
    tr_idx, te_idx = order[tr_o], order[te_o]
    r = np.random.RandomState(100)
    subj = np.unique(groups[tr_idx])
    is_va = np.isin(groups[tr_idx], r.choice(subj, int(0.2 * len(subj)), replace=False))
    in_tr, in_va = tr_idx[~is_va], tr_idx[is_va]
    stats = T.fit_stats(d["x"][in_tr])
    Xtr, Xva, Xte = flat(d, in_tr, stats), flat(d, in_va, stats), flat(d, te_idx, stats)
    stats_all = T.fit_stats(d["x"])
    Xall = flat(d, np.arange(N), stats_all)
    ext, hal = T.load("eicu_dyn"), T.load("hallym_dyn")
    Xe, Xh = flat(ext, np.arange(len(ext["stay_id"])), stats_all), flat(hal, np.arange(len(hal["stay_id"])), stats_all)
    oof = np.full((N, 4), np.nan, np.float32)
    pe, ph, its = np.zeros((len(Xe), 4), np.float32), np.zeros((len(Xh), 4), np.float32), []
    res = {"tag": tag, "model": "lgbm_dyn", "seed": a.seed, "cfg": {"data": "_dyn"}}
    for j, e in enumerate(T.ENDPOINTS):
        t0 = time.time()
        bst = lgb.train(P, lgb.Dataset(Xtr, d[e][in_tr]), 2000, valid_sets=[lgb.Dataset(Xva, d[e][in_va])],
                        feval=lambda p, ds: ("auprc", average_precision_score(ds.get_label(), p), True),
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[te_idx, j] = bst.predict(Xte, num_iteration=bst.best_iteration)
        yt = d[e][te_idx]
        res[f"oof_{e}_auroc"] = float(roc_auc_score(yt, oof[te_idx, j]))
        res[f"oof_{e}_auprc"] = float(average_precision_score(yt, oof[te_idx, j]))
        T.log(f"OOF {e}: AUROC={res[f'oof_{e}_auroc']:.4f} AUPRC={res[f'oof_{e}_auprc']:.4f} (n={len(te_idx)}) "
              f"best_iter={bst.best_iteration}")
        its.append(bst.best_iteration)
        fb = lgb.train(P, lgb.Dataset(Xall, d[e]), max(bst.best_iteration, 1))
        pe[:, j], ph[:, j] = fb.predict(Xe), fb.predict(Xh)
        T.log(f"[{e}] eICU AUROC={roc_auc_score(ext[e], pe[:, j]):.4f} AUPRC={average_precision_score(ext[e], pe[:, j]):.4f} | "
              f"Hallym AUROC={roc_auc_score(hal[e], ph[:, j]):.4f} AUPRC={average_precision_score(hal[e], ph[:, j]):.4f} "
              f"({(time.time() - t0) / 60:.1f} min)")
    res["best_eps"] = its
    np.savez_compressed(f"{T.ROOT}/results/preds/{tag}.npz", oof=oof, ext=pe, hal=ph, endpoints=np.array(T.ENDPOINTS))
    with open(f"{T.ROOT}/results/runs.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
