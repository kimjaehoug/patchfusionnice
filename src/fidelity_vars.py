"""Per-variable fidelity of imagined observation rates vs the actual Hallym hospital (per site), proposed (target profile,
tfrac=1) vs rate-matched thinning. Rates over active hours (from the first record of the window)."""
import numpy as np
import train as T
import obsprofile as PR
from regime import VARS, VIT

rng = np.random.RandomState(0)
d = T.load("mimic_dyn"); idx = rng.permutation(len(d["x"]))[:40000]
x, m = d["x"][idx], d["m"][idx]
acu = PR.BlockAcuity().fit(x, d["yhosp"][idx]); devp = PR.estimate(x, m, acu)
sites = PR.target_sites("hallym_dyn:sites")
act = lambda mm: np.cumsum((mm > 0).any(2), 1) > 0


def rate(mm):
    a = act(mm); return ((mm > 0) & a[..., None]).sum((0, 1)) / a.sum()


def vmon(mm):   # share of active hours with any vital, and vitals per monitored hour
    a = act(mm); mon = (mm[:, :, VIT] > 0).any(2)
    return (mon & a).sum() / a.sum()


sub = rng.permutation(len(x))[:8000]
xs0, ms0 = x[sub], m[sub]
r_dev, r_dev_all = rate(ms0), PR.obs_rate(m)
E = {"prop": [], "thin": []}
print(f"dev: monitored-hour share {vmon(ms0):.3f}; vcad dev {np.round(devp.vcad[VIT], 2)}")
for k, (sx, sm) in enumerate(sites):
    tp = PR.fit_window(PR.estimate(sx, sm, acu, beta=devp.beta), devp)
    rt = rate(sm)
    xp, mp = PR.imagine_two(xs0, ms0, acu, devp, tp, parts=["sel", "cad", "vitcad", "doccad"], rng=rng)
    xt, mt = PR.imagine_rate(xs0, ms0, r_dev_all, PR.obs_rate(sm), rng)
    rp, rth = rate(mp), rate(mt)
    print(f"\n== site {k} (n={len(sx)}): monitored-hour share tgt {vmon(sm):.3f} prop {vmon(mp):.3f} thin {vmon(mt):.3f} dev {vmon(ms0):.3f}; vcad tgt {np.round(tp.vcad[VIT], 2)}")
    print(f"{'var':9s} {'dev':>6s} {'target':>6s} {'prop':>6s} {'thin':>6s}")
    for c in range(len(VARS)):
        print(f"{VARS[c]:9s} {r_dev[c]:6.3f} {rt[c]:6.3f} {rp[c]:6.3f} {rth[c]:6.3f}{'  <vital' if c in VIT else ''}")
    for g, rr in (("prop", rp), ("thin", rth)):
        E[g].append(np.abs(np.log((rr + 1e-3) / (rt + 1e-3))))
grp = {"vitals": VIT, "labs": PR.LAB, "doc": PR.DOC}
print("\nmean |log rate ratio| imagined vs target, averaged over sites")
for g in E:
    e = np.mean(E[g], 0)
    print(f"  {g}: " + "  ".join(f"{n} {e[ii].mean():.3f}" for n, ii in grp.items()) + f"  all {e.mean():.3f}")
