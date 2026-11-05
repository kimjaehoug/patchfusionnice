"""72-hour long table -> dynamic-landmark samples with a 48-hour look-back.

For every stay and landmark L in LANDMARKS (hours from ICU admission) at which the patient is alive and still in the ICU,
one sample is formed from hours [L-48, L) of the persistence-filled timeline (hours before admission are unavailable).
Event times are re-expressed relative to L, so the competing-risk intervals and the endpoints (L,L+24], (L,L+48],
(L,L+168] and in-hospital death keep their hour-12 meaning. Persistence rules are those of build_tensor.py, applied on
the whole 72-hour timeline so that values carried across the window start remain available.

Usage: python build_dynamic.py mimic|eicu   (reads {src}_cohort72 / {src}_long72, writes {src}_dyn_tensor.npz)
"""
import os
import sys
import numpy as np
import pandas as pd
from build_tensor import VARS, CARRY, RANGE

ROOT = os.path.join(os.environ.get("PF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
H = int(os.environ.get("PF_H", "72"))                 # extraction window of the long table ({src}_long{H})
W = int(os.environ.get("PF_W", "48"))                 # look-back window
LANDMARKS = tuple(int(v) for v in os.environ.get("PF_LM", "12,24,36,48,60,72").split(","))
OUT_SUF = os.environ.get("PF_OUT", "_dyn")            # output {src}{OUT_SUF}_tensor.npz


def icu_out_hours(src, coh):
    if "t_icu_h" in coh:                                      # Hallym: end of continuous hourly monitoring
        return coh["t_icu_h"].values.astype(float)
    if src == "mimic":
        return ((pd.to_datetime(coh["outtime"]) - pd.to_datetime(coh["intime"])).dt.total_seconds() / 3600).values
    import duckdb
    e = os.environ.get("EICU_DIR", os.path.join(ROOT, "raw", "eicu-collaborative-research-database-2.0"))
    p = duckdb.sql(f"SELECT patientunitstayid AS stay_id, unitdischargeoffset/60.0 AS h "
                   f"FROM read_csv('{e}/patient.csv.gz', header=true, quote='\"', strict_mode=false)").df()
    return coh[["stay_id"]].merge(p, on="stay_id", how="left")["h"].values


def main(src):
    coh = pd.read_parquet(f"{ROOT}/{src}_cohort{H}.parquet").sort_values("stay_id").reset_index(drop=True)
    lg = pd.read_parquet(f"{ROOT}/{src}_long{H}.parquet")
    lg = lg[lg["var"].isin(VARS) & lg["stay_id"].isin(coh["stay_id"])]
    lo = lg["var"].map(lambda v: RANGE[v][0]).values
    hi = lg["var"].map(lambda v: RANGE[v][1]).values
    val = lg["value"].values.astype(np.float64)
    lg = lg[(val >= lo) & (val <= hi)]
    idx = pd.Series(np.arange(len(coh)), index=coh["stay_id"].values)
    vi = {v: i for i, v in enumerate(VARS)}
    N, C = len(coh), len(VARS)
    raw = np.full((N, H, C), np.nan, dtype=np.float32)
    raw[idx[lg["stay_id"].values].values, lg["hour"].values.astype(int), lg["var"].map(vi).values] = lg["value"].values
    mask = (~np.isnan(raw)).astype(np.float32)
    filled = raw.copy()
    for v, c in vi.items():
        w = CARRY[v]
        if w == 0:
            continue
        last = np.full(N, np.nan, dtype=np.float32)
        age = np.full(N, 10 ** 6)
        for t in range(H):
            obs = mask[:, t, c] > 0
            last[obs] = raw[obs, t, c]
            age[obs] = 0
            fill = (~obs) & (age <= w) & ~np.isnan(last)
            filled[fill, t, c] = last[fill]
            age = age + 1
    t_death = coh["t_death_h"].values.astype(float)
    t_disch = coh["t_disch_h"].values.astype(float)
    t_icu = icu_out_hours(src, coh)
    dead = ~np.isnan(t_death)
    xs, ms, keys = [], [], []
    for L in LANDMARKS:
        ok = (t_icu >= L) & (~dead | (t_death > L)) & (t_disch > L)
        i = np.where(ok)[0]
        x = np.full((len(i), W, C), np.nan, dtype=np.float32)
        m = np.zeros((len(i), W, C), dtype=np.float32)
        s = L - W
        a, b = max(s, 0), L
        x[:, a - s:, :] = filled[i, a:b]
        m[:, a - s:, :] = mask[i, a:b]
        xs.append(x)
        ms.append(m)
        keys.append(np.stack([i, np.full(len(i), L)], 1))
        print(f"{src} L={L}: {len(i)} samples, deaths in hospital {int(dead[i].sum())}", flush=True)
    k = np.concatenate(keys)
    i, L = k[:, 0], k[:, 1].astype(float)
    t_ev = np.where(dead[i], t_death[i], t_disch[i]) - L
    out = dict(x=np.concatenate(xs), m=np.concatenate(ms), stay_id=coh["stay_id"].values[i],
               subject_id=coh["subject_id"].values[i], landmark=L, t_event=t_ev, event=np.where(dead[i], 1, 2).astype(np.int8))
    rel = t_death[i] - L
    out["y24"] = (dead[i] & (rel <= 24)).astype(np.int8)
    out["y48"] = (dead[i] & (rel <= 48)).astype(np.int8)
    out["y168"] = (dead[i] & (rel <= 168)).astype(np.int8)
    out["yhosp"] = dead[i].astype(np.int8)
    if "hospitalid" in coh:
        out["site"] = coh["hospitalid"].values[i]
    np.savez_compressed(f"{ROOT}/{src}{OUT_SUF}_tensor.npz", **out)
    print(f"{src}: {len(i)} samples from {len(np.unique(i))} stays; prevalence "
          + " ".join(f"{e}={out[e].mean():.4f}" for e in ["y24", "y48", "y168", "yhosp"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
