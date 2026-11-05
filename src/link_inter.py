"""Reset x patch-scale imagination: interaction = (full - sel | reset) - (full - sel | no reset), all horizons,
paired cluster bootstrap; also the reset effect under full imagination. Writes into results/link_summary.json['interaction']."""
from bootstrap_par import boot_diff
import sys, json, numpy as np, train as T
from eval_hallym import frozen_preds, boot_idx
from link_eval import low_tertile
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
S = (2, 3, 4, 5)
PFX = sys.argv[1] if len(sys.argv) > 1 else "X"; OUT = sys.argv[2] if len(sys.argv) > 2 else "link_summary.json"
d = T.load("mimic_dyn"); st = T.fit_stats(d["x"]); h = T.load("hallym_dyn"); e = T.load("eicu_dyn"); low = low_tertile(e)
L = json.load(open(f"{T.ROOT}/results/{OUT}")); out = {}; reff = {}
for tg, ds, cl, msk in (("H", h, h["subject_id"], np.ones(len(h["y24"]), bool)), ("E", e, e["site"][low], low)):
    get = (lambda t: frozen_preds(t, h, st, "cuda")) if tg == "H" else (lambda t: np.load(f"{T.ROOT}/results/preds/{t}.npz")["ext"][low])
    P = {k: np.mean([get(f"{PFX}{k}{tg}_s{s}") for s in S], 0) for k in ["Rfull", "Rsel", "Nfull", "Nsel"]}
    P.update({k: np.mean([get(f"{PFX}{k}_s{s}") for s in S], 0) for k in ["Rbase", "Nbase"]})
    idx = list(boot_idx(cl, 1000, np.random.RandomState(0)))
    for j, ep in enumerate(["y24", "y48", "y168", "yhosp"]):
        y = ds[ep][msk]; ok = [i for i in idx if 0 < y[i].sum() < len(i)]
        q2 = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
        rd = {}
        for k in ("base", "sel", "full"):
            b = boot_diff(A, y, P["R" + k][:, j], P["N" + k][:, j], ok)
            rd[k] = (A(y, P["R" + k][:, j]) - A(y, P["N" + k][:, j]), b)
        reff[f"{tg}_{ep}"] = {k: [v[0], *q2(v[1])] for k, v in rd.items()}
        reff[f"{tg}_{ep}"]["inter"] = [rd["full"][0] - rd["base"][0], *q2(rd["full"][1] - rd["base"][1])]
        for nm, f in (("auroc", A), ("auprc", AP)):
            g = lambda k, i=None: f(y if i is None else y[i], P[k][:, j] if i is None else P[k][i, j])
            pt = {"dR": g("Rfull") - g("Rsel"), "dN": g("Nfull") - g("Nsel")}
            bs = np.stack([boot_diff(f, y, P["Rfull"][:, j], P["Rsel"][:, j], ok), boot_diff(f, y, P["Nfull"][:, j], P["Nsel"][:, j], ok)], 1)
            it = bs[:, 0] - bs[:, 1]; q = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
            r = {"dR": [pt["dR"], *q(bs[:, 0])], "dN": [pt["dN"], *q(bs[:, 1])], "inter": [pt["dR"] - pt["dN"], *q(it), float((it > 0).mean())]}
            out[f"{tg}_{ep}_{nm}"] = r
            print(f"{tg} {ep} {nm}: patch gain R {r['dR'][0]:+.3f} [{r['dR'][1]:+.3f},{r['dR'][2]:+.3f}]  N {r['dN'][0]:+.3f} [{r['dN'][1]:+.3f},{r['dN'][2]:+.3f}]  interaction {r['inter'][0]:+.3f} [{r['inter'][1]:+.3f},{r['inter'][2]:+.3f}] P>0={r['inter'][3]:.2f}", flush=True)
L["interaction"] = out; L["reset_effect"] = reff
json.dump(L, open(f"{T.ROOT}/results/{OUT}", "w"), indent=1, default=float)
print("written", OUT)
