"""Damage-ranked re-observation, step 1 (paper/notes/damage_guided_prereg.md): damage of all six profile components at
both targets (backbone WXRbase seeds 2-5, 30,000 held-out development samples, as Fig. 4), applicability, ranking, and
the training job file for the ranked and all-k subsets. No target outcome label is used.
Usage: python damage_rank.py  -> results/damage_rank_W.json, src/jobs_dmgsel.txt"""
import copy
import itertools
import json
import numpy as np
import train as T
import obsprofile as PR
from regime import VARS, VIT
from eval_hallym import frozen_preds
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

SEEDS = (2, 3, 4, 5)
COMPS = ["absent", "streams", "sel", "cad", "doccad", "vitcad"]
CODE = {"absent": "a", "streams": "s", "sel": "l", "cad": "c", "doccad": "d", "vitcad": "v"}
AFFECT = {"sel": PR.LAB, "cad": PR.LAB, "streams": PR.DOC, "doccad": PR.DOC, "vitcad": VIT}
TGT = {"H": "hallym_dyn:sites", "E": "eicu_dyn:low:sites"}
d = T.load("mimic_dyn")
st = T.fit_stats(d["x"])
groups = d["subject_id"]
us = np.unique(groups)
keep = np.random.RandomState(7).choice(us, int(0.3 * len(us)), replace=False)
train_mask = np.isin(groups, keep)
tr = np.where(train_mask)[0]
ho = np.where(~train_mask)[0]
idx = np.sort(np.random.RandomState(0).choice(ho, 30000, replace=False))
acu = PR.BlockAcuity().fit(d["x"][tr], d["yhosp"][tr])
devp = PR.estimate(d["x"][tr], d["m"][tr], acu)
TG = {k: [PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp) for sx, sm in PR.target_sites(v)] for k, v in TGT.items()}
x0, m0 = d["x"][idx], d["m"][idx]
base = {k: d[k][idx] for k in ("stay_id", "t_event", "event", "y24", "y48", "y168", "yhosp")}


def view(kind, tg):
    xs, ms = np.empty_like(x0), np.empty_like(m0)
    r = np.random.RandomState(1)
    site = r.randint(0, len(TG[tg]), len(idx))
    for k in range(len(TG[tg])):
        part = np.where(site == k)[0]
        if len(part):
            xs[part], ms[part] = PR.imagine_two(x0[part], m0[part], acu, devp, copy.deepcopy(TG[tg][k]),
                                                parts=("explicit", kind), rng=r)
    return xs, ms


y24, yh = base["y24"], base["yhosp"]
met = lambda p: (A(y24, p[:, 0]), AP(y24, p[:, 0]), A(yh, p[:, 3]))
f0 = {s: met(frozen_preds(f"WXRbase_s{s}", {**base, "x": x0, "m": m0}, st, "cuda")) for s in SEEDS}
out = {"seeds": SEEDS, "targets": {}}
for tg in TGT:
    res = {}
    for kind in COMPS:
        xs, ms = view(kind, tg)
        cols = AFFECT.get(kind) or sorted({c for p in TG[tg] for c in p.absent})
        obs = (m0[:, :, cols] > 0) if cols else np.zeros(1, bool)
        removed = float(((m0[:, :, cols] > 0) & (ms[:, :, cols] == 0)).sum() / max(obs.sum(), 1)) if cols else 0.0
        rows = np.array([[b - a for a, b in zip(f0[s], met(frozen_preds(f"WXRbase_s{s}", {**base, "x": xs, "m": ms}, st, "cuda")))]
                         for s in SEEDS])
        res[kind] = {"removed_share": removed, "applicable": removed >= 0.005, "per_seed": rows.tolist(),
                     "mean": rows.mean(0).tolist(), "min": rows.min(0).tolist(), "max": rows.max(0).tolist()}
        print(f"{tg} {kind:8s} removed {removed:6.3f}  dAUROC24 {rows[:, 0].mean():+.4f} [{rows[:, 0].min():+.3f},{rows[:, 0].max():+.3f}]"
              f"  dAUPRC24 {rows[:, 1].mean():+.4f}  dAUROChosp {rows[:, 2].mean():+.4f}", flush=True)
    app = [k for k in COMPS if res[k]["applicable"]]
    rank = sorted(app, key=lambda k: (res[k]["mean"][0], res[k]["mean"][1]))
    out["targets"][tg] = {"components": res, "applicable": app, "ranking": rank,
                          "top2": rank[:2], "top3": rank[:3], "all2": [list(c) for c in itertools.combinations(app, 2)]}
    print(tg, "ranking:", rank, flush=True)
json.dump(out, open(f"{T.ROOT}/results/damage_rank_W.json", "w"), indent=1, default=float)

CFG = {"data": "_dyn", "bs": 1024, "avail": True, "enc": {"dynpatch": True, "dyn_mode": "hard", "clean": True, "lstm_h": 256}}
lines, seen = [], set()
for tg, o in out["targets"].items():
    sets = [(o["top2"], SEEDS), (o["top3"], SEEDS)] + [(c, (2, 3)) for c in o["all2"]]
    for comps, seeds in sets:
        comps = sorted(comps, key=COMPS.index)
        code = "".join(CODE[c] for c in comps)
        for s in seeds:
            tag = f"WD{tg}_{code}_s{s}"
            if tag in seen:
                continue
            seen.add(tag)
            cfg = copy.deepcopy(CFG)
            cfg["profile"] = {"target": TGT[tg], "scheme": "mix", "parts": ["explicit"] + comps, "fix_rand_vitals": True}
            lines.append(f"{tag} --model hpf_noq --seed {s} --final_epochs 10 --sub_frac 0.3 --cfg '{json.dumps(cfg)}'")
open(f"{T.ROOT}/src/jobs_dmgsel.txt", "w").write("\n".join(lines) + "\n")
print(len(lines), "jobs written to src/jobs_dmgsel.txt")
