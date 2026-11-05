"""Aggregates for Table 1 of the Journal of Medical Systems manuscript: age, sex and study period per data source, restricted
to the stays of the dynamic-landmark cohorts. Prints aggregates only (no patient-level values); categories with fewer than 10
stays are suppressed. Run from src/: python table1_aggregates.py  -> paper/JOMS/table1_aggregates.txt"""
import numpy as np
import pandas as pd
from train import ROOT

L = []
for name, coh, key in (("MIMIC-IV", "mimic_cohort72", "mimic_dyn"), ("eICU-CRD", "eicu_cohort72", "eicu_dyn"),
                       ("Hallym", "hallym_cohort72", "hallym_dyn")):
    c = pd.read_parquet(f"{ROOT}/data/{coh}.parquet")
    ids = np.unique(np.load(f"{ROOT}/data/{key}_tensor.npz", allow_pickle=True)["stay_id"])
    c = c[c["stay_id"].isin(ids)]
    a = pd.to_numeric(c["age"], errors="coerce")
    g = c["gender"].astype(str).str.strip().str.lower()
    female = g.str.startswith(("f", "w", "여"))
    counts = g.value_counts()
    L += [name, f"  stays in cohort: {len(c)} of {len(ids)} in tensor",
          f"  age median (IQR): {a.median():.0f} ({a.quantile(.25):.0f}-{a.quantile(.75):.0f}), missing {int(a.isna().sum())}",
          f"  female: {int(female.sum())} ({100 * female.mean():.1f}%)",
          "  sex categories: " + ", ".join(f"{k}={v if v >= 10 else '<10'}" for k, v in counts.items())]
    if "t0" in c:
        y = pd.to_datetime(c["t0"], errors="coerce").dt.year
        L.append(f"  year of start of ICU-level care: {int(y.min())}-{int(y.max())}")
open(f"{ROOT}/paper/JOMS/table1_aggregates.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
