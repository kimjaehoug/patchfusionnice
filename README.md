# PatchFusion ESWA extension

Hour-12 landmark ICU mortality prediction on MIMIC-IV 3.1 (development, patient-level GroupKFold) with frozen external evaluation on eICU-CRD 2.0.

- `src/extract_mimic.py`, `src/extract_eicu.py` — cohort + first-12-hour hourly long table (DuckDB over the raw PhysioNet CSVs)
- `src/build_tensor.py mimic|eicu` — persistence-filled 22-channel tensors, raw-observation masks and labels
- `src/models.py` — PatchFusion (manuscript spec, 1,844,505 parameters), baselines (LSTM, Transformer, PatchTST, GRU-D), multi-horizon variants (`HorizonPF`: horizon queries + competing-risk discrete-time head; `state="carry"` encoder control)
- `src/train.py` — OOF evaluation, full-cohort retrain, frozen eICU scoring; per-epoch logs in `logs/train/{tag}.log`
- `src/run_queue.py jobs.txt --parallel 1 --gpu 0` — sequential job queue; `src/train_lgbm.py` — LightGBM baseline

## Data

No data is included. MIMIC-IV and eICU-CRD require PhysioNet credentialing and a data use agreement; raw files, derived
tables and predictions must stay on credentialed, institution-controlled machines (`data/`, `results/`, `logs/` are git-ignored).

```bash
export MIMIC_DIR=/path/to/mimic-iv-3.1 EICU_DIR=/path/to/eicu-collaborative-research-database-2.0
python3 src/extract_mimic.py && python3 src/extract_eicu.py
python3 src/build_tensor.py mimic && python3 src/build_tensor.py eicu
cd src && python3 run_queue.py jobs_phase1.txt --parallel 1 --gpu 0
```

Requirements: Python 3.12, torch, duckdb, numpy, pandas, pyarrow, scikit-learn, lightgbm.
