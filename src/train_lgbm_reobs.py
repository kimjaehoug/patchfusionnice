"""LightGBM trained on development data re-observed for a target (no target outcome labels); same flattened 48x88
input, split, hyper-parameters and refit protocol as train_lgbm_dyn.py. Plan: paper/notes/lgbm_reobs_prereg.md.
Training data = factual + one imagined copy; early stopping on factual + imagined development validation windows.

Usage: python train_lgbm_reobs.py --seed 2 --target E --arm three      (arm: three | rate; target: E | H)
"""
import time
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
import torch

import train as T

T.AVAIL = True
TARGETS = {"E": "eicu_dyn:low:sites", "H": "hallym_dyn:sites"}
ARMS = {"three": {"scheme": "mix", "fix_rand_vitals": True, "parts": ["sel", "cad", "vitcad", "doccad"], "tfrac": 1.0},
        "rate": {"simple": True}}


def flat(d, idx, stats):
    v, dv = T.make_input(d["x"][idx], stats)
    return np.concatenate([v, dv, d["m"][idx], (~np.isnan(d["x"][idx])).astype(np.float32)], -1).reshape(len(idx), -1)


def reobs(d, idx, prof, seed, stats):
    """factual + imagined copy of d[idx] as flat features and labels (one draw)."""
    fn = T.profile_trfn(d, idx, prof, np.random.RandomState(1000 * seed + 7), stats, torch.device("cpu"))
    X, Y = fn()[:2]
    return X.numpy().reshape(len(X), -1), Y.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--target", choices=list(TARGETS), required=True)
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    a = ap.parse_args()
    tag = f"L_{a.arm}{a.target}_s{a.seed}"
    T.LOG = f"{T.ROOT}/logs/train/{tag}.log"
    prof = dict(ARMS[a.arm], target=TARGETS[a.target])
    P = dict(objective="binary", learning_rate=0.05, num_leaves=63, max_depth=6, min_data_in_leaf=50,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=16, seed=a.seed)
    d = T.load("mimic_dyn")
    N = len(d["stay_id"]); groups = d["subject_id"]
    order = np.random.RandomState(0).permutation(N)
    tr_o, _ = list(GroupKFold(5).split(order, groups=groups[order]))[0]
    tr_idx = order[tr_o]
    r = np.random.RandomState(100)
    subj = np.unique(groups[tr_idx])
    is_va = np.isin(groups[tr_idx], r.choice(subj, int(0.2 * len(subj)), replace=False))
    in_tr, in_va = tr_idx[~is_va], tr_idx[is_va]
    stats = T.fit_stats(d["x"][in_tr])
    Xtr, Ytr = reobs(d, in_tr, prof, a.seed, stats)
    Xva, Yva = reobs(d, in_va, prof, a.seed + 100, stats)
    stats_all = T.fit_stats(d["x"])
    Xall, Yall = reobs(d, np.arange(N), prof, a.seed, stats_all)
    ext, hal = T.load("eicu_dyn"), T.load("hallym_dyn")
    Xe, Xh = flat(ext, np.arange(len(ext["stay_id"])), stats_all), flat(hal, np.arange(len(hal["stay_id"])), stats_all)
    pe, ph, its = np.zeros((len(Xe), 4), np.float32), np.zeros((len(Xh), 4), np.float32), []
    for j, e in enumerate(T.ENDPOINTS):
        t0 = time.time()
        bst = lgb.train(P, lgb.Dataset(Xtr, Ytr[:, j]), 2000, valid_sets=[lgb.Dataset(Xva, Yva[:, j])],
                        feval=lambda p, ds: ("auprc", average_precision_score(ds.get_label(), p), True),
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        its.append(bst.best_iteration)
        fb = lgb.train(P, lgb.Dataset(Xall, Yall[:, j]), max(bst.best_iteration, 1))
        pe[:, j], ph[:, j] = fb.predict(Xe), fb.predict(Xh)
        T.log(f"[{tag} {e}] best_iter={bst.best_iteration} eICU AUROC={roc_auc_score(ext[e], pe[:, j]):.4f} | "
              f"Hallym AUROC={roc_auc_score(hal[e], ph[:, j]):.4f} ({(time.time() - t0) / 60:.1f} min)")
    np.savez_compressed(f"{T.ROOT}/results/preds/{tag}.npz", ext=pe, hal=ph, best_iter=np.array(its), endpoints=np.array(T.ENDPOINTS))
    T.log(f"=== finished {tag}")


if __name__ == "__main__":
    main()
