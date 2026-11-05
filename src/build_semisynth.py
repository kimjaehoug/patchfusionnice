"""Semi-synthetic targets for paper/notes/structure_semisynth_prereg.md: held-out 30% of MIMIC-IV patients re-observed by a
known observation process. Writes data/synA_dyn_tensor.npz (selection) and data/synB_dyn_tensor.npz (stay-level streams)."""
import copy
import numpy as np
import train as T
import obsprofile as PR
d = T.load("mimic_dyn"); groups = d["subject_id"]; us = np.unique(groups)
keep = np.random.RandomState(7).choice(us, int(0.7 * len(us)), replace=False)
tr = np.where(np.isin(groups, keep))[0]; ho = np.where(~np.isin(groups, keep))[0]
print("development windows", len(tr), "held-out target windows", len(ho), flush=True)
acu = PR.BlockAcuity().fit(d["x"][tr], d["yhosp"][tr]); devp = PR.estimate(d["x"][tr], d["m"][tr], acu)
x, m = d["x"][ho], d["m"][ho]
tA = copy.deepcopy(devp)
for c in PR.LAB:
    tA.alpha_w[c] = devp.alpha_w[c] - 2.0
PR.fit_window(tA, devp) if False else None
xa, ma = PR.imagine_two(x, m, acu, devp, tA, parts=["sel"], rng=np.random.RandomState(11))
tB = copy.deepcopy(devp); tB.absent = []
for v in ("gcs", "uo"):
    c = PR.VARS.index(v); tB.rho[c] = 0.4 * devp.rho[c]
xb, mb = PR.imagine_two(x, m, acu, devp, tB, parts=[], rng=np.random.RandomState(12))
base = {k: v[ho] for k, v in d.items() if hasattr(v, "__len__") and len(v) == len(groups) and k not in ("x", "m")}
for name, (xx, mm) in (("synA", (xa, ma)), ("synB", (xb, mb))):
    out = {**base, "x": xx.astype(np.float32), "m": mm.astype(np.float32), "site": np.array([name] * len(ho), dtype=object)}
    np.savez_compressed(f"{T.ROOT}/data/{name}_dyn_tensor.npz", **out)
    r0, r1 = PR.obs_rate(m), PR.obs_rate(mm)
    print(name, "lab rate", round(float(r0[PR.LAB].mean()), 4), "->", round(float(r1[PR.LAB].mean()), 4),
          "| gcs", round(float(r0[PR.VARS.index('gcs')]), 4), "->", round(float(r1[PR.VARS.index('gcs')]), 4),
          "| uo", round(float(r0[PR.VARS.index('uo')]), 4), "->", round(float(r1[PR.VARS.index('uo')]), 4), flush=True)
