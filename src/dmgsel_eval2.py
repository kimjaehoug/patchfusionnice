"""Damage-guided selective re-observation v2 evaluation (paper/notes/damage_guided_prereg_v2.md).
Greedy prefixes k = 1..6 versus backbone and full; greedy k = 1 / 2 versus all singles / pairs (seeds 2-5 ensembles).
Usage: python dmgsel_eval2.py  -> results/dmgsel2_W.json, paper/notes/damage_guided_results_v2.md"""
import itertools
import json
import os
import shlex
import numpy as np
from sklearn.metrics import roc_auc_score as A, average_precision_score as AP
import train as T
from eval_hallym import frozen_preds, boot_idx
from bootstrap_par import boot_metric
from link_eval import low_tertile

R = T.ROOT
COMPS = ["absent", "streams", "sel", "cad", "doccad", "vitcad"]
CODE = {"absent": "a", "streams": "s", "sel": "l", "cad": "c", "doccad": "d", "vitcad": "v"}
KO = {"absent": "부재 변수(site)", "streams": "차팅 스트림(stay)", "sel": "검사 선택(stay)", "cad": "검사 cadence(patch)",
      "doccad": "차팅 cadence(patch)", "vitcad": "활력징후 cadence(patch)"}
TNAME = {"H": "Hallym", "E": "저검사 eICU-CRD"}
SEEDS = (2, 3, 4, 5)
B = 1000
code = lambda comps: "".join(CODE[c] for c in sorted(comps, key=COMPS.index))
name = lambda k: " + ".join(KO[c] for c in COMPS if CODE[c] in k)


def register():
    p = f"{R}/results/runs_quick.jsonl"
    have = {json.loads(l)["tag"] for l in open(p) if l.strip()}
    with open(p, "a") as fh:
        for jf in ("jobs_dmgsel.txt", "jobs_dmgsel2.txt"):
            for l in open(f"{R}/src/{jf}"):
                if not l.strip():
                    continue
                tag, rest = l.split(None, 1)
                a = shlex.split(rest)
                if tag in have or not os.path.exists(f"{R}/results/preds/{tag}.pt"):
                    continue
                have.add(tag)
                fh.write(json.dumps({"tag": tag, "model": a[a.index("--model") + 1], "seed": int(a[a.index("--seed") + 1]),
                                     "cfg": json.loads(a[a.index("--cfg") + 1]), "best_eps": [10]}) + "\n")


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    register()
    gr = json.load(open(f"{R}/results/damage_greedy_W.json"))["targets"]
    d = T.load("mimic_dyn")
    stats = T.fit_stats(d["x"])
    out = {}
    for tg in ("H", "E"):
        if tg == "H":
            h = T.load("hallym_dyn")
            msk, ys, cl = np.ones(len(h["y24"]), bool), h, h["subject_id"]
            get = lambda t: frozen_preds(t, h, stats, "cuda")
        else:
            e = T.load("eicu_dyn")
            msk = low_tertile(e); ys, cl = e, e["site"][msk]
            get = lambda t: np.load(f"{R}/results/preds/{t}.npz")["ext"]

        def ens(pfx):
            tags = [f"{pfx}_s{s}" for s in SEEDS]
            if not all(os.path.exists(f"{R}/results/preds/{t}.npz") for t in tags):
                return None
            return np.mean([get(t)[msk][:, 0] for t in tags], 0)

        order = gr[tg]["order"]
        P = {"backbone": ens("WXRbase"), "full": ens(f"WXRfull{tg}")}
        for k in range(1, 6):
            P[f"g{k}"] = ens(f"WD{tg}_{code(order[:k])}")
        singles = {code([c]): ens(f"WD{tg}_{code([c])}") for c in COMPS}
        pairs = {code(p): ens(f"WD{tg}_{code(p)}") for p in itertools.combinations(COMPS, 2)}
        singles = {k: v for k, v in singles.items() if v is not None}
        pairs = {k: v for k, v in pairs.items() if v is not None}
        y = ys["y24"][msk]
        idx = [i for i in boot_idx(cl, B, np.random.RandomState(0)) if 0 < y[i].sum() < len(i)]
        res = {"order": order, "n_singles": len(singles), "n_pairs": len(pairs)}
        for f, fn in (("auroc", A), ("auprc", AP)):
            pt = {k: float(fn(y, v)) for k, v in P.items() if v is not None}
            bm = {k: boot_metric(fn, y, v, idx) for k, v in P.items() if v is not None}
            r = {"point": pt, "curve": {}}
            gain = pt["full"] - pt["backbone"]
            for k in range(1, 7):
                key = "full" if k == 6 else f"g{k}"
                if key in pt:
                    r["curve"][k] = {"value": pt[key], "recovered": (pt[key] - pt["backbone"]) / gain,
                                     "minus_backbone": [pt[key] - pt["backbone"], *ci(bm[key] - bm["backbone"])],
                                     "minus_full": [pt[key] - pt["full"], *ci(bm[key] - bm["full"])]}
            for kk, pool in ((1, singles), (2, pairs)):
                gk = code(order[:kk])
                if gk in pool and len(pool) >= 3:
                    sp = {k: float(fn(y, v)) for k, v in pool.items()}
                    bp = {k: boot_metric(fn, y, v, idx) for k, v in pool.items()}
                    oth = [k for k in pool if k != gk]
                    r[f"k{kk}"] = {"greedy": gk, "values": sp, "rank": 1 + sum(v > sp[gk] for v in sp.values()), "n": len(sp),
                                   "minus_mean_all": [sp[gk] - np.mean(list(sp.values())), *ci(bp[gk] - np.mean([bp[k] for k in pool], 0))],
                                   "minus_mean_other": [sp[gk] - np.mean([sp[k] for k in oth]), *ci(bp[gk] - np.mean([bp[k] for k in oth], 0))],
                                   "best": max(sp, key=sp.get)}
            res[f] = r
        out[tg] = res
        print(tg, order, {f: {k: res[f][k]["rank"] for k in ("k1", "k2") if k in res[f]} for f in ("auroc", "auprc")}, flush=True)
    json.dump(out, open(f"{R}/results/dmgsel2_W.json", "w"), indent=1, default=float)
    write_md(out, gr)


def fmt(v):
    return f"{v[0]:+.3f} ({v[1]:+.3f} to {v[2]:+.3f})"


def write_md(out, gr):
    L = ["# Damage-guided selective re-observation v2: 결과", "",
         "사전 등록: `paper/notes/damage_guided_prereg_v2.md` (학습 전에 고정). 코드: `src/damage_greedy.py`, `src/dmgsel_eval2.py`. "
         "원자료: `results/damage_greedy_W.json`, `results/dmgsel2_W.json`.", "",
         "조건은 표 5 / Fig. 6과 같습니다: 은닉 256, 개발 환자 30%, 10 epoch, target/random 50:50, 시드 2–5 앙상블, 24-h 사망. "
         "95% CI는 Hallym 환자 클러스터, eICU 병원 클러스터 부트스트랩(B = 1000)입니다.", ""]
    L += ["## 1. 진단: greedy 조건부 손상 순서 (타깃 라벨 미사용)", "",
          "각 단계에서 이미 고른 구성요소를 적용한 상태로, 남은 구성요소 각각의 추가 손상(Δ 24-h AUROC)을 다시 잽니다.", ""]
    for tg in ("H", "E"):
        L += [f"### {TNAME[tg]}", "", "| 단계 | 선택 | 후보별 추가 Δ 24-h AUROC |", "|---|---|---|"]
        for i, stp in enumerate(gr[tg]["steps"]):
            cand = ", ".join(f"{KO[c]} {v['mean'][0]:+.3f}" for c, v in sorted(stp["candidates"].items(), key=lambda kv: kv[1]["mean"][0]))
            L.append(f"| {i + 1} | **{KO[stp['chosen']]}** | {cand} |")
        L.append("")
    for tg in ("H", "E"):
        if tg not in out:
            continue
        r = out[tg]
        L += [f"## {'2' if tg == 'H' else '3'}. {TNAME[tg]}", "", "### 이득 곡선 (greedy 상위 k개; k = 6은 전체 재관측)", "",
              "| k | 추가된 구성요소 | AUROC | 회복률 | − 백본 | − 전체 | AUPRC | 회복률 | − 전체 |", "|---|---|---|---|---|---|---|---|---|"]
        for k in range(1, 7):
            a, p = r["auroc"]["curve"].get(k), r["auprc"]["curve"].get(k)
            if a is None:
                continue
            L.append(f"| {k} | {KO[r['order'][k - 1]]} | {a['value']:.3f} | {a['recovered']:.0%} | {fmt(a['minus_backbone'])} | "
                     f"{fmt(a['minus_full']) if k < 6 else '–'} | {p['value']:.3f} | {p['recovered']:.0%} | {fmt(p['minus_full']) if k < 6 else '–'} |")
        L += ["", f"백본: AUROC {r['auroc']['point']['backbone']:.3f}, AUPRC {r['auprc']['point']['backbone']:.3f}", "",
              "### 같은 개수의 다른 조합과 비교", "", "| | 지표 | greedy 순위 | greedy − 전체 평균 | greedy − 나머지 평균 | 가장 좋은 조합 |", "|---|---|---|---|---|---|"]
        for kk in (1, 2):
            for f in ("auroc", "auprc"):
                x = r[f].get(f"k{kk}")
                if x:
                    L.append(f"| k = {kk} ({name(x['greedy'])}) | {f.upper()} | {x['rank']} / {x['n']} | {fmt(x['minus_mean_all'])} | "
                             f"{fmt(x['minus_mean_other'])} | {name(x['best'])} |")
        x2a, x2p = r["auroc"].get("k2"), r["auprc"].get("k2")
        if x2a and x2p:
            ok = any(x["minus_mean_all"][1] > 0 and x["rank"] <= 5 for x in (x2a, x2p))
            L += ["", f"사전 기준(k = 2가 쌍 평균보다 CI > 0이고 15개 중 5위 이내, AUROC 또는 AUPRC): **{'충족' if ok else '미충족'}**", ""]
            L += ["<details><summary>모든 쌍 (24-h AUROC / AUPRC, 시드 2–5)</summary>", "", "| 쌍 | AUROC | AUPRC |", "|---|---|---|"]
            for k in sorted(x2a["values"], key=lambda k: -x2a["values"][k]):
                L.append(f"| {name(k)}{' **(greedy)**' if k == x2a['greedy'] else ''} | {x2a['values'][k]:.3f} | {x2p['values'][k]:.3f} |")
            L += ["", "</details>", ""]
    open(f"{R}/paper/notes/damage_guided_results_v2.md", "w").write("\n".join(L) + "\n")
    print("written paper/notes/damage_guided_results_v2.md")


if __name__ == "__main__":
    main()
