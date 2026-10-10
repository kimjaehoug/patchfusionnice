"""long table -> hourly tensors (N,12,22) of persistence-filled raw values + raw-observation masks + labels.

Usage: python build_tensor.py mimic|eicu
Persistence (Supplementary Methods): vitals and FiO2 carried forward within the 12-h window; ABG/lactate <=4 h,
chemistries <=8 h, CBC <=12 h; treatment indicators, urine output, GCS and norepinephrine not carried.
Standardisation is done later with training-fold statistics.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
VARS = ["rr", "temp", "hr", "spo2", "sbp", "dbp", "bun", "lactate", "wbc", "hb", "na", "k", "cl", "ph", "be", "ag",
        "ne", "vaso", "fio2", "vent", "uo", "gcs"]
CARRY = {"rr": 12, "temp": 12, "hr": 12, "spo2": 12, "sbp": 12, "dbp": 12, "fio2": 12,
         "ph": 4, "be": 4, "lactate": 4, "bun": 8, "na": 8, "k": 8, "cl": 8, "ag": 8, "wbc": 12, "hb": 12,
         "ne": 0, "vaso": 0, "vent": 0, "uo": 0, "gcs": 0}
RANGE = {"rr": (0, 80), "temp": (25, 45), "hr": (0, 300), "spo2": (0, 100), "sbp": (0, 300), "dbp": (0, 200),
         "bun": (0, 300), "lactate": (0, 30), "wbc": (0, 500), "hb": (0, 25), "na": (90, 200), "k": (1, 12),
         "cl": (50, 180), "ph": (6.5, 8.0), "be": (-40, 40), "ag": (0, 60), "ne": (0, 5), "vaso": (0, 1),
         "fio2": (21, 100), "vent": (0, 1), "uo": (0, 3000), "gcs": (3, 15)}


def main(src):
    coh = pd.read_parquet(f"{ROOT}/{src}_cohort.parquet").sort_values("stay_id").reset_index(drop=True)
    lg = pd.read_parquet(f"{ROOT}/{src}_long.parquet")
    lg = lg[lg["var"].isin(VARS)]
    idx = pd.Series(np.arange(len(coh)), index=coh["stay_id"].values)
    lg = lg[lg["stay_id"].isin(idx.index)]
    N, T, C = len(coh), 12, len(VARS)
    raw = np.full((N, T, C), np.nan, dtype=np.float32)
    vi = {v: i for i, v in enumerate(VARS)}
    lo = lg["var"].map(lambda v: RANGE[v][0]).values
    hi = lg["var"].map(lambda v: RANGE[v][1]).values
    val = lg["value"].values.astype(np.float64)
    ok = (val >= lo) & (val <= hi)
    lg = lg[ok]
    raw[idx[lg["stay_id"].values].values, lg["hour"].values.astype(int), lg["var"].map(vi).values] = lg["value"].values
    mask = (~np.isnan(raw)).astype(np.float32)
    filled = raw.copy()
    for v, c in vi.items():
        w = CARRY[v]
        if w == 0:
            continue
        last = np.full(N, np.nan, dtype=np.float32)
        age = np.full(N, 10 ** 6)
        for t in range(T):
            obs = mask[:, t, c] > 0
            last[obs] = raw[obs, t, c]
            age[obs] = 0
            fill = (~obs) & (age <= w) & ~np.isnan(last)
            filled[fill, t, c] = last[fill]
            age = age + 1
    t_death = coh["t_death_h"].values.astype(float)
    dead = ~np.isnan(t_death)
    labels = {
        "y24": (dead & (t_death <= 24)).astype(np.int8),
        "y48": (dead & (t_death <= 48)).astype(np.int8),
        "y168": (dead & (t_death <= 168)).astype(np.int8),
        "yhosp": dead.astype(np.int8),
    }
    t_event = np.where(dead, t_death, coh["t_disch_h"].values.astype(float))
    np.savez_compressed(f"{ROOT}/{src}_tensor.npz", x=filled, m=mask, stay_id=coh["stay_id"].values,
                        subject_id=coh["subject_id"].astype(str).values, t_event=t_event,
                        event=np.where(dead, 1, 2).astype(np.int8), **labels)
    print(src, "N", N)
    for k, y in labels.items():
        print(f"  {k}: {y.sum()} ({y.mean() * 100:.2f}%)")
    print("  observed fraction per var (raw mask):")
    print("  " + " ".join(f"{v}={mask[:, :, c].mean():.2f}" for v, c in vi.items()))
    print("  stays with any obs per var:")
    print("  " + " ".join(f"{v}={(mask[:, :, c].sum(1) > 0).mean():.2f}" for v, c in vi.items()))


if __name__ == "__main__":
    main(sys.argv[1])
