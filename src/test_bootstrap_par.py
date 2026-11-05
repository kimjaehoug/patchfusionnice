"""Check bootstrap_par.boot_diff against the serial computation: identical values, and speed-up."""
import time, numpy as np, train as T
from eval_hallym import boot_idx
from bootstrap_par import boot_diff
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
e = T.load("eicu_dyn"); y = e["y24"]
ld = lambda p: np.mean([np.load(f"{T.ROOT}/results/preds/{p}_s{s}.npz")["ext"] for s in (2, 3, 4)], 0)[:, 0]
a, b = ld("W_base"), ld("W_noreset")
idx = [i for i in boot_idx(e["site"], 200, np.random.RandomState(0)) if 0 < y[i].sum() < len(i)]
for nm, f in (("AUROC", A), ("AUPRC", AP)):
    t0 = time.time(); ser = np.array([f(y[i], a[i]) - f(y[i], b[i]) for i in idx]); ts = time.time() - t0
    t0 = time.time(); par = boot_diff(f, y, a, b, idx); tp = time.time() - t0
    print(f"{nm}: n={len(idx)} identical={np.array_equal(ser, par)} max|diff|={np.abs(ser - par).max():.2e} | serial {ts:.1f}s parallel {tp:.1f}s speed-up x{ts / tp:.1f} | CI serial {np.percentile(ser, [2.5, 97.5]).round(4)} parallel {np.percentile(par, [2.5, 97.5]).round(4)}", flush=True)
