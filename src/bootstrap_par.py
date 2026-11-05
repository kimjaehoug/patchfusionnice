"""Parallel cluster-bootstrap of metric differences. Same resamples and same arithmetic as the serial list
comprehension np.array([f(y[i], a[i]) - f(y[i], b[i]) for i in idx]); only the loop is split over processes.
Arrays are passed once per worker chunk (fork, copy-on-write), not once per resample."""
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

N_JOBS = int(os.environ.get("BOOT_JOBS", 24))
_G = {}


def _chunk(args):
    lo, hi = args
    y, a, b, idx, f = _G["y"], _G["a"], _G["b"], _G["idx"], _G["f"]
    return [f(y[i], a[i]) - f(y[i], b[i]) for i in idx[lo:hi]]


def boot_diff(f, y, a, b, idx, n_jobs=None):
    """f: metric(y_true, score); y, a, b: full arrays; idx: list of index arrays (one per resample).
    Returns np.array of f(y[i], a[i]) - f(y[i], b[i]) in the order of idx."""
    n_jobs = n_jobs or N_JOBS
    if n_jobs <= 1 or len(idx) < 2 * n_jobs:
        return np.array([f(y[i], a[i]) - f(y[i], b[i]) for i in idx])
    _G.update(y=y, a=a, b=b, idx=idx, f=f)
    edges = np.linspace(0, len(idx), n_jobs + 1).astype(int)
    with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("fork")) as ex:
        parts = list(ex.map(_chunk, list(zip(edges[:-1], edges[1:]))))
    _G.clear()
    return np.array([v for p in parts for v in p])


def _chunk1(args):
    lo, hi = args
    y, a, idx, f = _G["y"], _G["a"], _G["idx"], _G["f"]
    return [f(y[i], a[i]) for i in idx[lo:hi]]


def boot_metric(f, y, a, idx, n_jobs=None):
    """np.array([f(y[i], a[i]) for i in idx]), computed in parallel."""
    n_jobs = n_jobs or N_JOBS
    if n_jobs <= 1 or len(idx) < 2 * n_jobs:
        return np.array([f(y[i], a[i]) for i in idx])
    _G.update(y=y, a=a, idx=idx, f=f)
    edges = np.linspace(0, len(idx), n_jobs + 1).astype(int)
    with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("fork")) as ex:
        parts = list(ex.map(_chunk1, list(zip(edges[:-1], edges[1:]))))
    _G.clear()
    return np.array([v for p in parts for v in p])
