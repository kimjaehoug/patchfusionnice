"""LightGBM on the flattened 12x66 tensor; same folds / inner split / external protocol as train.py."""
import json
import time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

import train as T

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, max_depth=6, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8, seed=2)


def flat(d, idx, stats):
    v, dv = T.make_input(d["x"][idx], stats)
    return np.concatenate([v, dv, d["m"][idx]], -1).reshape(len(idx), -1)


def main():
    T.LOG = f"{T.ROOT}/logs/train/lgbm.log"
    d = T.load("mimic")
    ext = T.load("eicu")
    N = len(d["stay_id"])
    groups = d["subject_id"]
    order = np.random.RandomState(0).permutation(N)
    folds = list(GroupKFold(5).split(order, groups=groups[order]))
    for e in T.ENDPOINTS:
        t0 = time.time()
        oof = np.zeros(N, dtype=np.float32)
        iters = []
        for f, (tr_o, te_o) in enumerate(folds):
            tr_idx, te_idx = order[tr_o], order[te_o]
            r = np.random.RandomState(100 + f)
            subj = np.unique(groups[tr_idx])
            is_va = np.isin(groups[tr_idx], r.choice(subj, int(0.2 * len(subj)), replace=False))
            in_tr, in_va = tr_idx[~is_va], tr_idx[is_va]
            stats = T.fit_stats(d["x"][in_tr])
            dtr = lgb.Dataset(flat(d, in_tr, stats), d[e][in_tr])
            dva = lgb.Dataset(flat(d, in_va, stats), d[e][in_va])
            bst = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva], feval=lambda p, ds: ("auprc", average_precision_score(ds.get_label(), p), True),
                            callbacks=[lgb.early_stopping(50, verbose=False)])
            iters.append(bst.best_iteration)
            oof[te_idx] = bst.predict(flat(d, te_idx, stats), num_iteration=bst.best_iteration)
            T.log(f"[lgbm {e} f{f}] best_iter={bst.best_iteration} TEST AUROC={roc_auc_score(d[e][te_idx], oof[te_idx]):.4f} "
                  f"AUPRC={average_precision_score(d[e][te_idx], oof[te_idx]):.4f}")
        stats = T.fit_stats(d["x"])
        bst = lgb.train(PARAMS, lgb.Dataset(flat(d, np.arange(N), stats), d[e]), int(np.median(iters)))
        pe = bst.predict(flat(ext, np.arange(len(ext["stay_id"])), stats))
        tag = f"lgbm_{e}_s2"
        res = {"tag": tag, "model": "lgbm", "seed": 2, "best_eps": iters,
               f"oof_{e}_auroc": float(roc_auc_score(d[e], oof)), f"oof_{e}_auprc": float(average_precision_score(d[e], oof)),
               f"ext_{e}_auroc": float(roc_auc_score(ext[e], pe)), f"ext_{e}_auprc": float(average_precision_score(ext[e], pe)),
               "minutes": (time.time() - t0) / 60}
        np.savez_compressed(f"{T.ROOT}/results/preds/{tag}.npz", oof=oof[:, None], ext=pe[:, None], endpoints=np.array([e]))
        with open(f"{T.ROOT}/results/runs.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")
        T.log(f"[lgbm {e}] OOF AUROC={res[f'oof_{e}_auroc']:.4f} AUPRC={res[f'oof_{e}_auprc']:.4f} | "
              f"eICU AUROC={res[f'ext_{e}_auroc']:.4f} AUPRC={res[f'ext_{e}_auprc']:.4f}")


if __name__ == "__main__":
    main()
