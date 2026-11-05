"""Damage-guided selective re-observation v2 (paper/notes/damage_guided_prereg_v2.md): greedy conditional damage on
held-out development patients (backbone WXRbase seeds 2-5; no target outcome label), then the training jobs for the
greedy prefixes and for all single components and pairs (seeds 2-5), skipping models that already exist.
Usage: python damage_greedy.py  -> results/damage_greedy_W.json, src/jobs_dmgsel2.txt"""
import copy
import itertools
import json
import os
import numpy as np
import train as T
import obsprofile as PR
from eval_hallym import frozen_preds
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP

SEEDS = (2, 3, 4, 5)
COMPS = ["absent", "streams", "sel", "cad", "doccad", "vitcad"]
CODE = {"absent": "a", "streams": "s", "sel": "l", "cad": "c", "doccad": "d", "vitcad": "v"}
TGT = {"H": "hallym_dyn:sites", "E": "eicu_dyn:low:sites"}
code = lambda comps: "".join(CODE[c] for c in sorted(comps, key=COMPS.index))
d = T.load("mimic_dyn")
st = T.fit_stats(d["x"])
groups = d["subject_id"]
us = np.unique(groups)
keep = np.random.RandomState(7).choice(us, int(0.3 * len(us)), replace=False)
train_mask = np.isin(groups, keep)
tr = np.where(train_mask)[0]
idx = np.sort(np.random.RandomState(0).choice(np.where(~train_mask)[0], 30000, replace=False))
acu = PR.BlockAcuity().fit(d["x"][tr], d["yhosp"][tr])
devp = PR.estimate(d["x"][tr], d["m"][tr], acu)
TG = {k: [PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp) for sx, sm in PR.target_sites(v)] for k, v in TGT.items()}
x0, m0 = d["x"][idx], d["m"][idx]
base = {k: d[k][idx] for k in ("stay_id", "t_event", "event", "y24", "y48", "y168", "yhosp")}
y24 = base["y24"]


def view(comps, tg):
    xs, ms = np.empty_like(x0), np.empty_like(m0)
    r = np.random.RandomState(1)
    site = r.randint(0, len(TG[tg]), len(idx))
    for k in range(len(TG[tg])):
        part = np.where(site == k)[0]
        if len(part):
            xs[part], ms[part] = PR.imagine_two(x0[part], m0[part], acu, devp, copy.deepcopy(TG[tg][k]),
                                                parts=("explicit",) + tuple(comps), rng=r)
    return xs, ms


def metrics(comps, tg):
    xm = (x0, m0) if not comps else view(comps, tg)
    out = []
    for s in SEEDS:
        p = frozen_preds(f"WXRbase_s{s}", {**base, "x": xm[0], "m": xm[1]}, st, "cuda")
        out.append((A(y24, p[:, 0]), AP(y24, p[:, 0])))
    return np.array(out)                                   # [seeds, 2]


res = {"seeds": SEEDS, "targets": {}}
for tg in TGT:
    S, steps = [], []
    cur = metrics(S, tg)
    while len(S) < len(COMPS):
        cand = {}
        for c in [c for c in COMPS if c not in S]:
            dlt = metrics(S + [c], tg) - cur              # damage of c given S, per seed
            cand[c] = {"mean": dlt.mean(0).tolist(), "per_seed": dlt.tolist()}
        best = min(cand, key=lambda c: (cand[c]["mean"][0], cand[c]["mean"][1]))
        steps.append({"given": list(S), "candidates": cand, "chosen": best})
        print(f"{tg} step {len(S) + 1} given {S}: " + "  ".join(f"{c} {v['mean'][0]:+.4f}/{v['mean'][1]:+.4f}" for c, v in cand.items())
              + f"  -> {best}", flush=True)
        S.append(best)
        cur = metrics(S, tg)
    res["targets"][tg] = {"order": S, "steps": steps}
json.dump(res, open(f"{T.ROOT}/results/damage_greedy_W.json", "w"), indent=1, default=float)

CFG = {"data": "_dyn", "bs": 1024, "avail": True, "enc": {"dynpatch": True, "dyn_mode": "hard", "clean": True, "lstm_h": 256}}
first, rest, seen = [], [], set()
for tg, o in res["targets"].items():
    greedy = [o["order"][:k] for k in range(1, 6)]
    others = [[c] for c in COMPS] + [list(p) for p in itertools.combinations(COMPS, 2)]
    for comps, bucket in [(g, first) for g in greedy] + [(c, rest) for c in others]:
        for s in SEEDS:
            tag = f"WD{tg}_{code(comps)}_s{s}"
            if tag in seen or os.path.exists(f"{T.ROOT}/results/preds/{tag}.npz"):
                continue
            seen.add(tag)
            cfg = copy.deepcopy(CFG)
            cfg["profile"] = {"target": TGT[tg], "scheme": "mix", "parts": ["explicit"] + sorted(comps, key=COMPS.index),
                              "fix_rand_vitals": True}
            bucket.append(f"{tag} --model hpf_noq --seed {s} --final_epochs 10 --sub_frac 0.3 --cfg '{json.dumps(cfg)}'")
open(f"{T.ROOT}/src/jobs_dmgsel2.txt", "w").write("\n".join(first + rest) + "\n")
print(len(first), "greedy +", len(rest), "comparator jobs written to src/jobs_dmgsel2.txt")
