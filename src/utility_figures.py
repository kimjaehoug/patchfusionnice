"""Figure 1 (calibration, 24-h death, risk deciles) and Figure 2 (decision curves) for low-testing eICU-CRD hospitals
and the Hallym hospitals; hidden-256 models, five-seed ensembles. Palette validated (all pairs) with the dataviz
validator (five slots, all pairs; the aqua-magenta pair is in the 6-8 CVD band, so line style is the required secondary encoding)."""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train as T
from link_eval import low_tertile

S = (2, 3, 4, 5, 6)
OUT = f"{T.ROOT}/paper/ESWA_draft/figures"
STY = {"proposed": ("Proposed", "#2a78d6", "-", 2.2), "backbone": ("Backbone (source-only)", "#008300", "--", 1.6),
       "generic": ("Generic counterfactual", "#e87ba4", ":", 1.8), "selection_only": ("Selection only", "#eda100", "-.", 1.6),
       "thinning": (os.environ.get("PF_THIN_LABEL", "Rate-matched thinning"), "#1baf7a", (0, (4, 1.5, 1, 1.5, 1, 1.5)), 1.6)}
TAGS = {"E": {"backbone": "W_base", "generic": "W_gen", "selection_only": "W_selE", "thinning": "W_thinE", "proposed": "W_propE"},
        "H": {"backbone": "W_base", "generic": "W_gen", "selection_only": "W_selH", "thinning": "W_thinH", "proposed": "W_propH"}}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
e = T.load("eicu_dyn"); h = T.load("hallym_dyn"); low = low_tertile(e)
HP = dict(np.load(f"{T.ROOT}/results/hallym_W_preds.npz"))
DATA = {}
for tgt, tags in TAGS.items():
    y = (e["y24"][low] if tgt == "E" else h["y24"]).astype(int)
    P = {g: (np.mean([np.load(f"{T.ROOT}/results/preds/{t}_s{s}.npz")["ext"] for s in S], 0)[low, 0] if tgt == "E" else HP[t][:, 0]) for g, t in tags.items()}
    DATA[tgt] = (y, P)
TITLE = {"E": "Low-testing eICU-CRD hospitals", "H": "Hallym hospitals"}


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)


plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
# Figure 1: calibration by risk decile
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
for ax, tgt in zip(axes, ("E", "H")):
    y, P = DATA[tgt]
    lim = 0
    for g in ("backbone", "generic", "selection_only", "thinning", "proposed"):
        p = P[g]; q = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.digitize(p, q[1:-1]), 0, 9)
        xs = np.array([p[b == k].mean() for k in range(10)]); ys = np.array([y[b == k].mean() for k in range(10)])
        lab, col, ls, lw = STY[g]
        ax.plot(100 * xs, 100 * ys, ls=ls, color=col, lw=lw, marker="o", ms=3.5, mec="#fcfcfb", mew=0.8, label=lab)
        lim = max(lim, xs.max(), ys.max())
    ax.plot([0, 100 * lim * 1.05], [0, 100 * lim * 1.05], color=MUTED, lw=1, ls=(0, (2, 2)), zorder=0, label="Perfect calibration")
    ax.set_xlim(0, 100 * lim * 1.05); ax.set_ylim(0, 100 * lim * 1.05)
    ax.set_title(TITLE[tgt], fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Predicted 24-h mortality (%)", color=INK); ax.set_ylabel("Observed 24-h mortality (%)", color=INK)
    style(ax)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f"{OUT}/Figure1_calibration{os.environ.get('PF_FIG_SUFFIX', '')}.png", dpi=300); plt.close(fig)
# Figure 2: decision curves
pts = np.linspace(0.01, 0.25, 49)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
for ax, tgt in zip(axes, ("E", "H")):
    y, P = DATA[tgt]; n = len(y); prev = y.mean()
    for g in ("backbone", "generic", "selection_only", "thinning", "proposed"):
        p = P[g]; nb = [(((p >= t) & (y == 1)).sum() - ((p >= t) & (y == 0)).sum() * t / (1 - t)) / n for t in pts]
        lab, col, ls, lw = STY[g]; ax.plot(100 * pts, 1000 * np.array(nb), ls=ls, color=col, lw=lw, label=lab)
    ta = prev - (1 - prev) * pts / (1 - pts)
    ax.plot(100 * pts, 1000 * ta, color=MUTED, lw=1, ls=(0, (2, 2)), label="Alert all")
    ax.axhline(0, color=MUTED, lw=0.8, label="Alert none")
    top = max(max(l.get_ydata().max() for l in ax.get_lines()[:5]), 1)
    ax.set_ylim(-1, top * 1.08); ax.set_xlim(1, 25)
    ax.set_title(TITLE[tgt], fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("Threshold probability (%)", color=INK); ax.set_ylabel("Net benefit (per 1,000 reassessments)", color=INK)
    style(ax)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f"{OUT}/Figure2_decision_curve{os.environ.get('PF_FIG_SUFFIX', '')}.png", dpi=300); plt.close(fig)
print("written figures")
