"""S15 revised: per-window statistics -> mean over windows within a hospital -> unweighted mean over hospitals.
Acuity-thirds use the target hospital's boundaries for all records of that hospital. Mean interval uses windows with
>= 2 test hours (share of such windows reported); longest gap runs from the first record of the window. Imagination is
repeated with three random draws (SD over draws). SpO2: number of hospitals recording SpO2 more often than the
development data (imagination cannot add observations). Also unique fatal stays per hospital for S17.
Writes results/v12_fid_W.json."""
import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
import train as T
import obsprofile as PR
from regime import VIT
from link_eval import low_tertile

rng0 = np.random.RandomState(0)
d = T.load("mimic_dyn"); i = rng0.permutation(len(d["x"]))[:40000]
acu = PR.BlockAcuity().fit(d["x"][i], d["yhosp"][i]); dv = PR.estimate(d["x"][i], d["m"][i], acu)
r_dev = PR.obs_rate(d["m"][i]); pool = rng0.permutation(np.setdiff1d(np.arange(len(d["x"])), i))
LAB, SP = PR.LAB, VIT[3]


def stats(x, m, q):
    o = m > 0; lab = o[:, :, LAB]; act = np.cumsum(o.any(2), 1) > 0
    tested = lab.any(1); th = lab.any(2)
    r = {"lab_window_presence": tested.mean(), "lab_hourly_rate": (lab & act[..., None]).sum() / max(act.sum() * len(LAB), 1),
         "spo2_hourly_rate": (o[:, :, SP] & act).sum() / max(act.sum(), 1)}
    gaps, runs, n2 = [], [], 0
    for n in range(len(o)):
        t = np.where(th[n])[0]
        if len(t) > 1:
            gaps.append(np.diff(t).mean()); n2 += 1
        a = np.where(act[n])[0]
        if len(a):
            best = cur = 0
            for v in ~th[n, a[0]:]:
                cur = cur + 1 if v else 0; best = max(best, cur)
            runs.append(best)
    r.update({"lab_mean_interval_h": np.mean(gaps) if gaps else np.nan, "share_windows_2plus": n2 / len(o), "lab_longest_gap_h": np.mean(runs),
              "lab_co_measurement": ((lab.sum(2) >= 2) & th).sum() / max(th.sum(), 1)})
    with np.errstate(all="ignore"):
        aw = np.nanmean(acu(x), 1)
    ok = ~np.isnan(aw)
    r["lab_presence_low_acuity"] = tested[ok & (aw <= q[0])].mean(); r["lab_presence_high_acuity"] = tested[ok & (aw > q[1])].mean()
    return {k: float(v) for k, v in r.items()}


out = {}
for tname, label in (("hallym_dyn:sites", "Hallym"), ("eicu_dyn:low:sites", "eICU-CRD low testing")):
    sites = PR.target_sites(tname); per = {g: [] for g in ("target", "development", "rate_level", "three_scale")}
    draws = {g: [[] for _ in range(3)] for g in ("rate_level", "three_scale")}
    sp_above, ptr = 0, 0
    for sx, sm in sites:
        k = min(len(sx), 3000); ri = rng0.permutation(len(sx))[:k]; di = pool[ptr:ptr + k]; ptr += k
        with np.errstate(all="ignore"):
            awt = np.nanmean(acu(sx[ri]), 1)
        q = np.nanquantile(awt, [1 / 3, 2 / 3])
        xd, md = d["x"][di], d["m"][di]
        tp = PR.fit_window(PR.estimate(sx, sm, acu, beta=dv.beta), dv)
        per["target"].append(stats(sx[ri], sm[ri], q)); per["development"].append(stats(xd, md, q))
        sp_above += int(PR.obs_rate(sm)[SP] > r_dev[SP])
        for rep_ in range(3):
            rng = np.random.RandomState(100 + rep_)
            draws["rate_level"][rep_].append(stats(*PR.imagine_rate(xd, md, r_dev, PR.obs_rate(sm), rng), q))
            draws["three_scale"][rep_].append(stats(*PR.imagine_two(xd, md, acu, dv, tp, parts=["sel", "cad", "vitcad", "doccad"], rng=rng), q))
    res = {"hospitals": len(sites), "spo2_hospitals_above_development": sp_above}
    keys = per["target"][0].keys()
    for g in ("target", "development"):
        res[g] = {k: float(np.nanmean([r[k] for r in per[g]])) for k in keys}
    for g in ("rate_level", "three_scale"):
        m_ = np.array([[np.nanmean([r[k] for r in draws[g][rep_]]) for k in keys] for rep_ in range(3)])
        res[g] = dict(zip(keys, m_.mean(0).tolist())); res[g + "_sd"] = dict(zip(keys, m_.std(0, ddof=1).tolist()))
    out[label] = res
    print(f"\n[{label}] hospitals {len(sites)}; SpO2 above development at {sp_above}")
    for k in keys:
        print(f"  {k:26s} target {res['target'][k]:.3f} dev {res['development'][k]:.3f} rate {res['rate_level'][k]:.3f}±{res['rate_level_sd'][k]:.3f} three {res['three_scale'][k]:.3f}±{res['three_scale_sd'][k]:.3f}", flush=True)
# unique fatal stays per hospital (S17)
e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
fs = {}
for name, ds, mask in (("hallym", h, np.ones(len(h["y24"]), bool)), ("eicu_low", e, low)):
    for s_ in np.unique(ds["site"][mask]):
        k = mask & (ds["site"] == s_)
        fs[f"{name}:{s_}"] = [int(ds["y24"][k].sum()), int(len(np.unique(ds["stay_id"][k & (ds["y24"] == 1)])))]
out["fatal_stays"] = fs
print({k: v for k, v in fs.items() if k.startswith("hallym")})
json.dump(out, open(f"{T.ROOT}/results/v12_fid_W.json", "w"), indent=1)
print("written results/v12_fid_W.json")
