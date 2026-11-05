"""Damage-ranked re-observation, step 2 (paper/notes/damage_guided_prereg.md): evaluate top-2 / top-3 / all 2-subsets
against the backbone (WXRbase) and full re-observation (WXRfull{H,E}). Seed ensembles; Hallym patient-cluster and
low-testing eICU-CRD hospital-cluster bootstrap (B = 1000).
Usage: python dmgsel_eval.py  -> results/dmgsel_W.json, paper/notes/damage_guided_results.md"""
import json
import os
import shlex
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
import train as T
from eval_hallym import frozen_preds, boot_idx
from bootstrap_par import boot_metric
from link_eval import low_tertile

R = T.ROOT
CODE = {"absent": "a", "streams": "s", "sel": "l", "cad": "c", "doccad": "d", "vitcad": "v"}
NAME = {"absent": "absent variables (site)", "streams": "charting streams (stay)", "sel": "laboratory selection (stay)",
        "cad": "laboratory cadence (patch)", "doccad": "charting cadence (patch)", "vitcad": "vital-sign cadence (patch)"}
TNAME = {"H": "Hallym", "E": "low-testing eICU-CRD"}
S4, S2 = (2, 3, 4, 5), (2, 3)
B = 1000
code = lambda comps: "".join(CODE[c] for c in sorted(comps, key=list(CODE).index))


def register():
    """append quick-run records of finished WD jobs so that frozen_preds can rebuild the models"""
    p = f"{R}/results/runs_quick.jsonl"
    have = {json.loads(l)["tag"] for l in open(p) if l.strip()}
    with open(p, "a") as fh:
        for l in open(f"{R}/src/jobs_dmgsel.txt"):
            if not l.strip():
                continue
            tag, rest = l.split(None, 1)
            a = shlex.split(rest)
            if tag in have or not os.path.exists(f"{R}/results/preds/{tag}.pt"):
                continue
            fh.write(json.dumps({"tag": tag, "model": a[a.index("--model") + 1], "seed": int(a[a.index("--seed") + 1]),
                                 "cfg": json.loads(a[a.index("--cfg") + 1]), "best_eps": [10]}) + "\n")


def main():
    register()
    rk = json.load(open(f"{R}/results/damage_rank_W.json"))["targets"]
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    out, md = {}, []
    for tg in ("H", "E"):
        if tg == "H":
            h = T.load("hallym_dyn")
            msk = np.ones(len(h["y24"]), bool); ys, cl = h, h["subject_id"]
            get = lambda t: frozen_preds(t, h, stats, "cuda")
        else:
            e = T.load("eicu_dyn")
            msk = low_tertile(e); ys, cl = e, e["site"][msk]
            get = lambda t: np.load(f"{R}/results/preds/{t}.npz")["ext"]
        cache = {}

        def ens(pfx, seeds):
            tags = [f"{pfx}_s{s}" for s in seeds]
            miss = [t for t in tags if not os.path.exists(f"{R}/results/preds/{t}.npz")]
            if miss:
                return None
            for t in tags:
                if t not in cache:
                    cache[t] = get(t)[msk]
            return np.mean([cache[t] for t in tags], 0)

        o = rk[tg]
        G = {"backbone": ens("WXRbase", S4), "full": ens(f"WXRfull{tg}", S4),
             "top2": ens(f"WD{tg}_{code(o['top2'])}", S4), "top3": ens(f"WD{tg}_{code(o['top3'])}", S4)}
        G2 = {code(c): ens(f"WD{tg}_{code(c)}", S2) for c in o["all2"]}
        G2 = {k: v for k, v in G2.items() if v is not None}
        G2["backbone"], G2["full"] = ens("WXRbase", S2), ens(f"WXRfull{tg}", S2)
        y = ys["y24"][msk]
        idx = [i for i in boot_idx(cl, B, np.random.RandomState(0)) if 0 < y[i].sum() < len(i)]
        res = {"n_subsets_done": len(G2) - 2, "ranking": o["ranking"], "top2": o["top2"], "top3": o["top3"]}
        BM = {}
        for f, fn in (("auroc", A), ("auprc", AP)):
            for k, p in list(G.items()) + [("s2:" + k, p) for k, p in G2.items()]:
                if p is not None:
                    BM[(f, k)] = boot_metric(fn, y, p[:, 0], idx)
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for f, fn in (("auroc", A), ("auprc", AP)):
            pt = {k: float(fn(y, p[:, 0])) for k, p in G.items() if p is not None}
            r = {"point": pt}
            if all(k in pt for k in ("backbone", "full", "top2")):
                r["top2_minus_backbone"] = [pt["top2"] - pt["backbone"], *ci(BM[(f, "top2")] - BM[(f, "backbone")])]
                r["full_minus_backbone"] = [pt["full"] - pt["backbone"], *ci(BM[(f, "full")] - BM[(f, "backbone")])]
                r["top2_minus_full"] = [pt["top2"] - pt["full"], *ci(BM[(f, "top2")] - BM[(f, "full")])]
                r["recovered"] = (pt["top2"] - pt["backbone"]) / (pt["full"] - pt["backbone"]) if pt["full"] != pt["backbone"] else None
            if "top3" in pt and "full" in pt:
                r["top3_minus_full"] = [pt["top3"] - pt["full"], *ci(BM[(f, "top3")] - BM[(f, "full")])]
                r["recovered3"] = (pt["top3"] - pt["backbone"]) / (pt["full"] - pt["backbone"])
            subs = [k for k in G2 if k not in ("backbone", "full")]
            c2 = code(o["top2"])
            if c2 in subs and len(subs) >= 2:
                s_pt = {k: float(fn(y, G2[k][:, 0])) for k in subs}
                others = [k for k in subs if k != c2]
                r["subsets_s23"] = s_pt
                r["top2_rank"] = 1 + sum(v > s_pt[c2] for v in s_pt.values())
                mean_all = np.mean([BM[(f, "s2:" + k)] for k in subs], 0)
                mean_oth = np.mean([BM[(f, "s2:" + k)] for k in others], 0)
                r["top2_minus_random_mean"] = [s_pt[c2] - np.mean(list(s_pt.values())), *ci(BM[(f, "s2:" + c2)] - mean_all)]
                r["top2_minus_other_mean"] = [s_pt[c2] - np.mean([s_pt[k] for k in others]), *ci(BM[(f, "s2:" + c2)] - mean_oth)]
                dmg = {k: sum(rk[tg]["components"][c]["mean"][0] for c in [n for n in CODE if CODE[n] in k]) for k in subs}
                rho = spearmanr([-dmg[k] for k in subs], [s_pt[k] for k in subs])
                r["spearman_damage_vs_metric"] = [float(rho.correlation), float(rho.pvalue)]
                if "backbone" in G2 and G2["backbone"] is not None:
                    r["backbone_s23"] = float(fn(y, G2["backbone"][:, 0])); r["full_s23"] = float(fn(y, G2["full"][:, 0]))
            res[f] = r
        res["oe24"] = {k: float(y.sum() / p[:, 0].sum()) for k, p in G.items() if p is not None}
        out[tg] = res
        print(tg, json.dumps({f: {k: v for k, v in res[f].items() if k != "subsets_s23"} for f in ("auroc", "auprc")}, default=float)[:1500], flush=True)
    json.dump(out, open(f"{R}/results/dmgsel_W.json", "w"), indent=1, default=float)
    write_md(out, rk)


def fmt(v, nd=3):
    return f"{v[0]:+.{nd}f} ({v[1]:+.{nd}f} to {v[2]:+.{nd}f})"


def write_md(out, rk):
    L = ["# Damage-ranked re-observation: results", "",
         "Pre-registration: `paper/notes/damage_guided_prereg.md` (fixed before training). Scripts: `src/damage_rank.py`, "
         "`src/dmgsel_eval.py`; raw numbers: `results/damage_rank_W.json`, `results/dmgsel_W.json`.", "",
         "Setting identical to Table 5 / Fig. 6: hidden 256, 30% of development patients, 10 epochs, 50% target / 50% random "
         "profiles. Backbone = WXRbase, full = all six components (WXRfull). Seed ensembles; 95% CI by patient-cluster "
         "(Hallym) or hospital-cluster (low-testing eICU-CRD) bootstrap, B = 1000. Metric: death within 24 h.", ""]
    L += ["## 1. Damage ranking (label-free, backbone seeds 2–5, held-out development patients)", "",
          "| Target | Rank | Component | Δ 24-h AUROC (range) | Δ 24-h AUPRC | Observations removed |", "|---|---|---|---|---|---|"]
    for tg in ("H", "E"):
        for i, c in enumerate(rk[tg]["ranking"]):
            cc = rk[tg]["components"][c]
            L.append(f"| {TNAME[tg]} | {i + 1} | {NAME[c]} | {cc['mean'][0]:+.3f} ({cc['min'][0]:+.3f}, {cc['max'][0]:+.3f}) | "
                     f"{cc['mean'][1]:+.3f} | {cc['removed_share']:.0%} |")
    L += ["", "All six components were applicable at both targets (each removes ≥ 0.5% of the affected observations).", ""]
    for tg in ("H", "E"):
        if tg not in out:
            continue
        r = out[tg]
        L += [f"## 2. {TNAME[tg]}", "", f"Top-2 by damage: {' + '.join(NAME[c] for c in r['top2'])}; top-3 adds {NAME[r['top3'][2]]}.", "",
              "| 24-h | Backbone | Full | Top-2 | Top-3 | Top-2 − backbone | Full − backbone | Top-2 − full | Recovered (top-2 / top-3) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for f in ("auroc", "auprc"):
            x = r[f]; p = x["point"]
            g = lambda k: f"{p[k]:.3f}" if k in p else "–"
            rec = f"{x['recovered']:.0%} / {x.get('recovered3', float('nan')):.0%}" if x.get("recovered") is not None else "–"
            L.append(f"| {f.upper()} | {g('backbone')} | {g('full')} | {g('top2')} | {g('top3')} | "
                     f"{fmt(x['top2_minus_backbone']) if 'top2_minus_backbone' in x else '–'} | "
                     f"{fmt(x['full_minus_backbone']) if 'full_minus_backbone' in x else '–'} | "
                     f"{fmt(x['top2_minus_full']) if 'top2_minus_full' in x else '–'} | {rec} |")
        L += ["", f"O/E 24 h: " + ", ".join(f"{k} {v:.2f}" for k, v in r["oe24"].items()), ""]
        if "subsets_s23" in r["auroc"]:
            L += [f"Versus random selection (all {len(r['auroc']['subsets_s23'])} two-component subsets, seeds 2–3 ensembles):", "",
                  "| 24-h | Top-2 rank | Top-2 − mean of all subsets | Top-2 − mean of the other subsets | Spearman (summed damage vs metric) |",
                  "|---|---|---|---|---|"]
            for f in ("auroc", "auprc"):
                x = r[f]
                L.append(f"| {f.upper()} | {x['top2_rank']} of {len(x['subsets_s23'])} | {fmt(x['top2_minus_random_mean'])} | "
                         f"{fmt(x['top2_minus_other_mean'])} | {x['spearman_damage_vs_metric'][0]:+.2f} (p = {x['spearman_damage_vs_metric'][1]:.2f}) |")
            L += ["", "<details><summary>All subsets (24-h AUROC / AUPRC, seeds 2–3)</summary>", "",
                  "| Subset | AUROC | AUPRC |", "|---|---|---|"]
            inv = {v: k for k, v in CODE.items()}
            for k in sorted(r["auroc"]["subsets_s23"], key=lambda k: -r["auroc"]["subsets_s23"][k]):
                L.append(f"| {' + '.join(inv[ch] for ch in k)} | {r['auroc']['subsets_s23'][k]:.3f} | {r['auprc']['subsets_s23'][k]:.3f} |")
            L.append(f"| *backbone / full (seeds 2–3)* | {r['auroc'].get('backbone_s23', float('nan')):.3f} / {r['auroc'].get('full_s23', float('nan')):.3f} | "
                     f"{r['auprc'].get('backbone_s23', float('nan')):.3f} / {r['auprc'].get('full_s23', float('nan')):.3f} |")
            L += ["", "</details>", ""]
        ok_rec = (r["auroc"].get("recovered") or 0) >= 0.7
        ok_rand = any(r[f].get("top2_minus_random_mean", [0, 0, 0])[1] > 0 for f in ("auroc", "auprc"))
        L += [f"Pre-registered rule: recovered ≥ 70% in 24-h AUROC: **{'met' if ok_rec else 'not met'}**; "
              f"top-2 minus random-mean CI above 0 in AUROC or AUPRC: **{'met' if ok_rand else 'not met'}**.", ""]
    open(f"{R}/paper/notes/damage_guided_results.md", "w").write("\n".join(L) + "\n")
    print("written paper/notes/damage_guided_results.md")


if __name__ == "__main__":
    main()
