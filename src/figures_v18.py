"""Figures 2-5 (v17), in the style of Figure 1: Fig. 2 evaluation design (schematic), Fig. 3 main comparisons, Fig. 4
calibration by risk decile, Fig. 5 decision curves. Proposed in blue, the procedure's own components in orange where
they are schematic, comparators in gray. Values in Fig. 3 are read from results/*.json (see the comments).
Writes paper/ESWA_draft/figures/v17/Fig{2..5}_*.{png,pdf}."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import train as T
from link_eval import low_tertile

OUT = f"{T.ROOT}/paper/ESWA_draft/figures/v18"; os.makedirs(OUT, exist_ok=True)
INK, MUTED, FAINT, GRID = "#1b1b1b", "#5c5b57", "#9a9893", "#e6e4df"
OR, OR_F, BL, BL_F, GR_F, GRAY = "#e8622c", "#fbe3d7", "#2a6fdb", "#e3ecfa", "#f3f2ef", "#7a7a7a"
R = lambda f: json.load(open(f"{T.ROOT}/results/{f}"))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig); print("written", name)


def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(FAINT)
    ax.tick_params(colors=MUTED, labelsize=7, length=2.5)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)


# ============================================================ Fig. 2: evaluation design (schematic, serif)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
fig = plt.figure(figsize=(7.2, 4.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
def box(x, y, w, h, fc, ec, lw=0.9): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=lw))
def t(x, y, s, size=7.4, c=INK, ha="center", **k): ax.text(x, y, s, fontsize=size, color=c, ha=ha, va="center", **k)
t(1, 60.3, "(a)", 9, ha="left", weight="bold"); t(5, 60.3, "Data sources and their roles", 8.6, ha="left")
box(1, 44, 20, 13.5, GR_F, FAINT); t(11, 54.4, "MIMIC-IV", 8, weight="bold"); t(11, 49.4, "development: 50,764 stays\ntraining; held-out fold for\ndevelopment metrics, thresholds", 6.6, MUTED)
box(23.5, 44, 50, 13.5, "white", FAINT); t(48.5, 55.3, "eICU-CRD: 207 hospitals, 103,491 stays (unlabeled records used for profiles)", 7.0, INK)
for i, (lab, fc, ec) in enumerate([("low-testing tertile\nadaptation target", BL_F, BL), ("middle-testing tertile\nevaluated after design freeze", "white", FAINT), ("all hospitals\nsource-only and overall", "white", FAINT)]):
    box(25 + i * 16.0, 45.3, 15.0, 7.6, fc, ec); t(32.5 + i * 16.0, 49.1, lab, 6.5, INK if i == 0 else MUTED)
box(76, 44, 23, 13.5, BL_F, BL, 1.2); t(87.5, 54.4, "Hallym (4 Korean hospitals)", 7.6, weight="bold"); t(87.5, 49.4, "1,102 stays; seven variables\nnever recorded; independent\nevaluation, held out from design", 6.6, MUTED)
t(1, 40.0, "(b)", 9, ha="left", weight="bold"); t(5, 40.0, "Dynamic landmarks and prediction horizons", 8.6, ha="left")
x0, sc = 6, 0.72
ax.plot([x0, x0 + 108 * sc], [26, 26], color=INK, lw=0.9)
for hh in range(0, 109, 12):
    ax.plot([x0 + hh * sc] * 2, [25.2, 26.8], color=INK, lw=0.8); t(x0 + hh * sc, 23.4, "ICU start" if hh == 0 else f"{hh} h", 6.6, MUTED)
L = 60
ax.add_patch(Rectangle((x0 + (L - 48) * sc, 27.6), 48 * sc, 3.6, fc=BL_F, ec=BL, lw=0.8)); t(x0 + (L - 24) * sc, 29.4, "48-h look-back window at landmark L = 60 h", 6.8)
ax.plot([x0 + L * sc] * 2, [24.5, 37.5], color=BL, lw=0.8, ls=(0, (2, 2)))
for k, (hz, lab) in enumerate([(24, "death within 24 h"), (48, "within 48 h")]):
    yy = 33.0 + k * 2.8; ax.annotate("", xy=(x0 + (L + hz) * sc, yy), xytext=(x0 + L * sc, yy), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=7))
    t(x0 + (L + hz) * sc + 1.0, yy, lab, 6.6, MUTED, ha="left")
t(x0 + L * sc + 1, 38.6, "also within 7 days and before hospital discharge", 6.4, MUTED, ha="left")
t(50, 20.6, "Landmarks every 12 h from 12 to 72 h; one sample per landmark while the patient is alive and in ICU-level care; hours before it are empty.", 6.4, MUTED)
t(1, 16.6, "(c)", 9, ha="left", weight="bold"); t(5, 16.6, "Evaluation regimes and measures", 8.6, ha="left")
box(1, 1.5, 30, 12.5, "white", FAINT); t(16, 11.2, "Frozen source-only evaluation", 7.6, weight="bold"); t(16, 6.3, "trained on MIMIC-IV only and\napplied unchanged; the backbone\nand every comparator", 6.6, MUTED)
box(34, 1.5, 32, 12.5, OR_F, OR, 1.2); t(50, 11.2, "Profile-guided re-observation", 7.6, weight="bold"); t(50, 6.3, "profile from the target's unlabeled records;\nretrained on re-observed development data;\nno target label in loss, epochs or selection", 6.6, INK)
box(69, 1.5, 30, 12.5, "white", FAINT); t(84, 11.2, "Measures", 7.6, weight="bold"); t(84, 6.3, "AUROC, AUPRC; O/E, calibration intercept,\nslope, Brier score; alerts at a threshold fixed\non development data; cluster bootstrap", 6.6, MUTED)
save(fig, "Fig2_evaluation")

# ============================================================ data figures (sans)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
# ---------------- Fig. 3: main comparisons
F = R("formal_summary_W.json"); N = R("naive_remask_W.json"); LG = R("lgbm_reobs_W.json"); SS = R("semisynth_W.json"); TF = R("rev3_W.json")
PG = json.load(open("/tmp/claude-1001/-home-jbnu/165ef764-4028-4fb0-a653-eb1b5306aa65/scratchpad/ci/prop_vs_gen.json")) if os.path.exists("/tmp/claude-1001/-home-jbnu/165ef764-4028-4fb0-a653-eb1b5306aa65/scratchpad/ci/prop_vs_gen.json") else None
HP = np.load(f"{T.ROOT}/results/hallym_W_preds.npz")
from sklearn.metrics import roc_auc_score as A
h = T.load("hallym_dyn")
rate_H = A(h["y24"], HP["W_thinH"][:, 0])                       # five-seed rate-level, Hallym
rate_E = TF["thin_E"]["y24"]["thinning"][0]                    # five-seed rate-level, low-testing eICU-CRD
METH = [("Three-scale profile (proposed)", F["prop_E"]["eicu_low_y24"][0], F["prop_H"]["hallym_y24"][0], BL, "o", True),
        ("Rate-level imagination", rate_E, rate_H, BL, "o", True),
        ("Entry-wise masking, same rates*", N["eICU-low"]["y24"]["naive_remask"][0], N["Hallym"]["y24"]["naive_remask"][0], GRAY, "o", False),
        ("Generic (random observation processes)", F["generic"]["eicu_low_y24"][0], F["generic"]["hallym_y24"][0], GRAY, "o", True),
        ("Source-only backbone", F["base"]["eicu_low_y24"][0], F["base"]["hallym_y24"][0], GRAY, "o", True),
        ("Importance reweighting", F["rw_E"]["eicu_low_y24"][0], F["rw_H"]["hallym_y24"][0], GRAY, "o", True)]
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.95, left=0.215, right=0.97, top=0.94, bottom=0.10)
# (a) one dot plot, both sources; values tabulated on the right
ax = fig.add_subplot(gs[0, :]); style(ax)
ys = np.arange(len(METH))[::-1]
for (lab, e_, h_, c, mk, filled), yy in zip(METH, ys):
    ax.plot([e_, h_], [yy, yy], color=GRID, lw=1.0, zorder=1)
    ax.plot([e_], [yy], marker="o", ms=5.2, mfc=c if filled else "white", mec=c, mew=1.1, ls="none", zorder=3)
    ax.plot([h_], [yy], marker="s", ms=4.8, mfc=c if filled else "white", mec=c, mew=1.1, ls="none", zorder=3)
    w = "bold" if lab.startswith("Three") else "normal"
    ax.text(1.03, yy, f"{e_:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.3, color=INK, weight=w)
    ax.text(1.10, yy, f"{h_:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.3, color=INK, weight=w)
ax.text(1.03, len(METH) - 0.35, "eICU", transform=ax.get_yaxis_transform(), fontsize=5.9, color=MUTED)
ax.text(1.10, len(METH) - 0.35, "Korean", transform=ax.get_yaxis_transform(), fontsize=5.9, color=MUTED)
ax.set_yticks(ys); ax.set_yticklabels(["Three-scale profile", "Rate-level imagination", "Entry-wise masking*", "Generic (random processes)", "Source-only backbone", "Importance reweighting"], fontsize=6.6, color=INK)
for tl, (_, _, _, c, _, _) in zip(ax.get_yticklabels(), METH): tl.set_color(BL if c == BL else INK)
ax.set_xlim(0.84, 0.925); ax.set_xticks([0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92]); ax.set_ylim(-0.6, len(METH) - 0.2)
ax.set_xlabel("24-h AUROC", color=INK, fontsize=6.8)
ax.set_title("(a) Transport to the target hospitals, 24-h AUROC", fontsize=7.8, loc="left", color=INK, x=-0.27)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=MUTED, ms=4.5, label="low-testing eICU-CRD"), Line2D([], [], marker="s", ls="none", color=MUTED, ms=4.2, label="Korean hospitals")],
          fontsize=6.0, frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.1, ncol=2)
# (b) forest: where the gain comes from
ax = fig.add_subplot(gs[1, 0]); style(ax)
pg_E = PG["E"]["y24"]["proposed-generic_auroc"] if PG else [0.0163, 0.012, 0.0206]
pg_H = PG["H"]["y24"]["proposed-generic_auroc"] if PG else [0.0233, 0.0063, 0.0441]
ROWS = [("Knowing the target", [("E", pg_E, "o"), ("H", pg_H, "s")]),
        ("Re-observing vs masking", [("E", N["eICU-low"]["y24"]["rate_reobs-naive_remask_auroc"], "o"), ("H", N["Hallym"]["y24"]["rate_reobs-naive_remask_auroc"], "s")]),
        ("Structure: charting streams", [("S", SS["B"]["y24"]["three-rate_auroc"], "D")]),
        ("Structure: laboratory tests", [("S", SS["A"]["y24"]["three-rate_auroc"], "D")])]
yt = []
for i, (lab, pts) in enumerate(ROWS):
    base = len(ROWS) - 1 - i; yt.append(base)
    for j, (src, (v, lo_, hi_), mk) in enumerate(pts):
        yy = base + ((0.16 if j == 0 else -0.16) if len(pts) > 1 else 0)
        c = BL if src == "H" else (INK if src == "E" else GRAY)
        ax.plot([lo_, hi_], [yy, yy], color=c, lw=1.2, solid_capstyle="round")
        ax.plot([v], [yy], marker=mk, ms=4.6 if mk != "D" else 4.0, color=c, mec="white", mew=0.5, zorder=3)
ax.axvline(0, color=MUTED, lw=0.7); ax.set_xlim(-0.004, 0.05); ax.set_xticks([0, 0.02, 0.04])
ax.set_yticks(yt); ax.set_yticklabels([r[0] for r in ROWS], fontsize=6.6, color=INK); ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel("Δ 24-h AUROC (95% CI)", color=INK, fontsize=6.8)
ax.set_title("(b) Where the gain comes from", fontsize=7.8, loc="left", color=INK, x=-0.85)
ax.legend(handles=[Line2D([], [], marker="o", color=INK, lw=1.2, ms=4, label="eICU-CRD"), Line2D([], [], marker="s", color=BL, lw=1.2, ms=4, label="Korean"),
                   Line2D([], [], marker="D", color=GRAY, lw=1.2, ms=3.6, label="same-rate targets")], fontsize=5.8, frameon=False, loc="lower right", handlelength=1.2, borderaxespad=0.1)
# (c) dumbbell: source-only -> re-observed, two models, two targets
ax = fig.add_subplot(gs[1, 1]); style(ax)
yy, ticks, labs = 4.4, [], []
for tgt, tl in (("E", "eICU"), ("H", "Korean")):
    for name, key, c in (("PatchFusion", "pf", BL), ("LightGBM", "lgbm", GRAY)):
        a_, b_ = LG[tgt]["y24"][f"{key}_src"][0], LG[tgt]["y24"][f"{key}_three"][0]
        ax.plot([a_, b_], [yy, yy], color=c, lw=1.4, zorder=2)
        ax.plot([a_], [yy], marker="o", ms=4.8, mfc="white", mec=c, mew=1.1, zorder=3)
        ax.plot([b_], [yy], marker="o", ms=4.8, color=c, zorder=3)
        ax.text(b_ + 0.003, yy, f"{b_:.3f}", va="center", fontsize=6.2, color=INK)
        ticks.append(yy); labs.append(f"{name}, {tl}"); yy -= 1
    yy -= 0.5
ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=6.6, color=INK)
for tl in ax.get_yticklabels(): tl.set_color(BL if tl.get_text().startswith("Patch") else INK)
ax.set_xlim(0.845, 0.935); ax.set_xticks([0.85, 0.88, 0.91]); ax.set_ylim(min(ticks) - 0.6, max(ticks) + 0.6)
ax.set_xlabel("24-h AUROC", color=INK, fontsize=6.8)
ax.set_title("(c) Source-only (open) to re-observed (filled)", fontsize=7.8, loc="left", color=INK, x=-0.62)
fig.text(0.215, 0.012, "* three seeds (Supplementary S19); other points five-seed ensembles (a) or three-seed ensembles (c). Blue: the procedure of this work.", fontsize=5.8, color=MUTED)
save(fig, "Fig3_main_comparisons")

# ---------------- Figs. 4-5: calibration and decision curves
S = (2, 3, 4, 5, 6)
e = T.load("eicu_dyn"); low = low_tertile(e)
TAGS = {"E": {"backbone": "W_base", "generic": "W_gen", "selection_only": "W_selE", "thinning": "W_thinE", "proposed": "W_propE"},
        "H": {"backbone": "W_base", "generic": "W_gen", "selection_only": "W_selH", "thinning": "W_thinH", "proposed": "W_propH"}}
STY = {"proposed": ("Three-scale profile (proposed)", BL, "-", 2.0), "thinning": ("Rate-level imagination", "#86b0ee", (0, (4, 1.6)), 1.6),
       "selection_only": ("Selection-only imagination", "#b5b3ad", "-.", 1.3), "generic": ("Generic counterfactual training", GRAY, ":", 1.6),
       "backbone": ("Source-only backbone", "#3a3a3a", (0, (6, 2)), 1.4)}
DATA = {}
for tgt, tags in TAGS.items():
    y = (e["y24"][low] if tgt == "E" else h["y24"]).astype(int)
    P = {g: (np.mean([np.load(f"{T.ROOT}/results/preds/{tg}_s{s}.npz")["ext"] for s in S], 0)[low, 0] if tgt == "E" else HP[tg][:, 0]) for g, tg in tags.items()}
    DATA[tgt] = (y, P)
TITLE = {"E": "Low-testing eICU-CRD hospitals", "H": "Korean hospitals (independent)"}
ORDER = ("backbone", "generic", "selection_only", "thinning", "proposed")
def style2(ax):
    style(ax); ax.grid(axis="y", color=GRID, lw=0.6)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
for i, (ax, tgt) in enumerate(zip(axes, ("E", "H"))):
    y, P = DATA[tgt]; lim = 0
    for g in ORDER:
        p = P[g]; q = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.digitize(p, q[1:-1]), 0, 9)
        xs = np.array([p[b == k].mean() for k in range(10)]); ys = np.array([y[b == k].mean() for k in range(10)])
        lab, col, ls, lw = STY[g]
        ax.plot(100 * xs, 100 * ys, ls=ls, color=col, lw=lw, marker="o", ms=3.2, mec="white", mew=0.5, label=lab, zorder=3 if g == "proposed" else 2)
        lim = max(lim, xs.max(), ys.max())
    ax.plot([0, 100 * lim * 1.05], [0, 100 * lim * 1.05], color=FAINT, lw=0.9, ls=(0, (2, 2)), zorder=0, label="Perfect calibration")
    ax.set_xlim(0, 100 * lim * 1.05); ax.set_ylim(0, 100 * lim * 1.05)
    ax.set_title(f"({'ab'[i]}) {TITLE[tgt]}", fontsize=8.2, color=INK, loc="left")
    ax.set_xlabel("Predicted 24-h mortality (%)", color=INK, fontsize=7.2); ax.set_ylabel("Observed 24-h mortality (%)", color=INK, fontsize=7.2); style2(ax)
hh_, ll_ = axes[0].get_legend_handles_labels()
fig.legend(hh_, ll_, loc="lower center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.14, 1, 1)); save(fig, "Fig4_calibration")
pts = np.linspace(0.01, 0.25, 49)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
for i, (ax, tgt) in enumerate(zip(axes, ("E", "H"))):
    y, P = DATA[tgt]; n = len(y); prev = y.mean()
    for g in ORDER:
        p = P[g]; nb = [(((p >= tt) & (y == 1)).sum() - ((p >= tt) & (y == 0)).sum() * tt / (1 - tt)) / n for tt in pts]
        lab, col, ls, lw = STY[g]; ax.plot(100 * pts, 1000 * np.array(nb), ls=ls, color=col, lw=lw, label=lab, zorder=3 if g == "proposed" else 2)
    ta = prev - (1 - prev) * pts / (1 - pts)
    ax.plot(100 * pts, 1000 * ta, color=FAINT, lw=0.9, ls=(0, (2, 2)), label="Alert all"); ax.axhline(0, color=MUTED, lw=0.7, label="Alert none")
    top = max(max(l.get_ydata().max() for l in ax.get_lines()[:5]), 1)
    ax.set_ylim(-1, top * 1.08); ax.set_xlim(1, 25)
    ax.set_title(f"({'ab'[i]}) {TITLE[tgt]}", fontsize=8.2, color=INK, loc="left")
    ax.set_xlabel("Threshold probability (%)", color=INK, fontsize=7.2); ax.set_ylabel("Net benefit per 1,000 reassessments", color=INK, fontsize=7.2); style2(ax)
hh_, ll_ = axes[0].get_legend_handles_labels()
fig.legend(hh_, ll_, loc="lower center", ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.14, 1, 1)); save(fig, "Fig5_decision_curve")
