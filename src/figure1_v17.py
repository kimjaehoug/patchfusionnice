"""Figure 1 (v17): profile-guided re-observation. Style follows a two-tone schematic: components designed in this work in
orange, the model in light blue, everything else gray. Writes paper/ESWA_draft/figures/v17/Fig1_overview.{png,pdf}."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "ESWA_draft", "figures", "v17"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
INK, MUTED, FAINT, GRID = "#1b1b1b", "#5c5b57", "#9a9893", "#e6e4df"
OR, OR_F = "#e8622c", "#fbe3d7"          # designed in this work
BL, BL_F = "#2a6fdb", "#e3ecfa"          # the model
GR_F = "#f3f2ef"

fig = plt.figure(figsize=(7.2, 6.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 88); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=0.9, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, c=MUTED, lw=0.9, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, ls=ls, mutation_scale=8, shrinkA=0, shrinkB=0))


def t(x, y, s, size=8, c=INK, ha="center", va="center", **k):
    ax.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, **k)


# ------------------------------------------------------------------ (a) forward path
t(1, 86.2, "(a)", 9, ha="left", weight="bold")
ax.plot([20, 73], [83.6, 83.6], color=OR, lw=0.8, ls=(0, (4, 2.5)))
t(46.5, 85.0, "no outcome label of the target hospital is used", 7.2, OR)
# target records (mini dot grid)
box(1.5, 72.5, 13.5, 10.0, GR_F, FAINT)
rs = np.random.RandomState(3)
for r in range(3):
    for c in range(9):
        if rs.rand() < [0.9, 0.35, 0.6][r]:
            ax.add_patch(Circle((3.6 + c * 1.25, 80.6 - r * 1.75), 0.33, color=INK, lw=0))
t(8.25, 74.6, "target hospital,\nunlabeled records", 6.6, MUTED)
# development patients
box(1.5, 58, 13.5, 10.0, GR_F, FAINT)
t(8.25, 63.0, "development\npatients (MIMIC-IV,\nwith labels)", 7, MUTED)
# observation profile (designed)
box(20, 69.5, 23.5, 13.5, OR_F, OR, 1.1)
t(31.75, 80.6, "Observation profile", 8.6, INK, weight="bold")
t(21.2, 77.5, "site:  variables never recorded", 7, ha="left")
t(21.2, 75.3, "stay:  who is tested; which charting", 7, ha="left")
t(21.2, 73.5, "         streams exist", 7, ha="left")
t(21.2, 71.3, "patch: recording cadence within a stay", 7, ha="left")
t(31.75, 67.3, "damage analysis ranks which components\nharm a model trained elsewhere", 6.4, MUTED)
arrow(15.0, 76.8, 20.0, 76.8)
# re-observation (designed, central)
box(48, 58, 25, 25, OR_F, OR, 1.6)
t(60.5, 79.4, "Re-observation", 9.6, INK, weight="bold")
t(60.5, 74.9, "each development patient is\nobserved as the target hospital\nwould have observed them", 7.2)
t(60.5, 66.6, "measurements are removed and\ncarry-forward, differences, masks\nand availability are recomputed", 6.8, MUTED)
t(60.5, 60.6, "Section 4.2", 6.4, FAINT)
arrow(43.5, 76.3, 48.0, 76.3)
arrow(15.0, 63.0, 48.0, 63.0)
# training + model (supporting)
box(76.5, 62.5, 12.5, 16, BL_F, BL, 0.9)
t(82.75, 75.6, "model", 8.2, INK, weight="bold")
t(82.75, 72.3, "trained on factual\nand re-observed\nrecords", 6.4, INK)
t(82.75, 66.3, "PatchFusion encoder\nor LightGBM", 6.4, MUTED)
arrow(73.0, 70.5, 76.5, 70.5)

t(94.6, 72.8, "risk at the\ntarget hospital", 7.2)
t(94.6, 68.4, "death within 24 h,\n48 h, 7 days,\nbefore discharge", 6.3, MUTED)
arrow(89.0, 70.5, 89.9, 70.5)

# ------------------------------------------------------------------ (b) re-observation, not masking
t(1, 53.2, "(b)", 9, ha="left", weight="bold"); t(5, 53.2, "Re-observing measurements, not masking entries", 8.6, ha="left")
X0, DX = 17.0, 2.35                       # hours 1..12
ROWS = [("development\nrecord", 46.5), ("re-observed\n(this work)", 38.0), ("entry-wise\nmasking", 29.5)]
meas = [0, 4, 8]                          # measurement hours (index), carried forward for 4 h
for lab, y in ROWS:
    t(15.2, y, lab, 6.8, MUTED, ha="right")
    for hh in range(12):
        ax.add_patch(Rectangle((X0 + hh * DX, y - 1.1), DX * 0.92, 2.2, fc="white", ec=GRID, lw=0.5))
for hh in range(12):
    t(X0 + hh * DX + DX * 0.46, 50.0, f"{hh + 1}", 6.3, FAINT)
t(X0 - 0.4, 50.0, "hour", 6.3, FAINT, ha="right")


def run(y, start, n, fc, ec=None):
    for k in range(n):
        ax.add_patch(Rectangle((X0 + (start + k) * DX, y - 1.1), DX * 0.92, 2.2, fc=fc, ec=ec or fc, lw=0.5))


# development record: three measurements, each carried forward for 4 h
for m in meas:
    run(46.5, m, 4, "#c9c7c1"); ax.add_patch(Circle((X0 + m * DX + DX * 0.46, 46.5), 0.55, color=INK, lw=0))
# re-observed: the measurement at hour 5 is removed with its carried-forward copies
for m in (0, 8):
    run(38.0, m, 4, "#c9c7c1"); ax.add_patch(Circle((X0 + m * DX + DX * 0.46, 38.0), 0.55, color=INK, lw=0))
ax.add_patch(Circle((X0 + 4 * DX + DX * 0.46, 38.0), 0.55, fc="white", ec=OR, lw=1.0))
ax.plot([X0 + 4 * DX, X0 + 8 * DX - 0.2], [36.2, 36.2], color=OR, lw=0.8)
t(X0 + 6 * DX, 35.0, "measurement at hour 5 removed with its carried-forward copies", 6.3, OR)
# entry-wise masking: entries dropped independently; holes inside runs, carried values without their measurement
for m in meas:
    run(29.5, m, 4, "#c9c7c1")
for hh in (1, 4, 6, 10):
    ax.add_patch(Rectangle((X0 + hh * DX, 28.4), DX * 0.92, 2.2, fc="white", ec=GRID, lw=0.5))
    ax.plot([X0 + hh * DX + 0.5, X0 + hh * DX + DX * 0.92 - 0.5], [28.8, 30.2], color=FAINT, lw=0.7)
    ax.plot([X0 + hh * DX + 0.5, X0 + hh * DX + DX * 0.92 - 0.5], [30.2, 28.8], color=FAINT, lw=0.7)
for m in (0, 8):
    ax.add_patch(Circle((X0 + m * DX + DX * 0.46, 29.5), 0.55, color=INK, lw=0))
t(X0 + 6 * DX, 26.6, "hour 5 dropped, but its carried values at hours 6 and 8 remain", 6.3, MUTED)
# legend for (b)
ax.add_patch(Circle((18.0, 22.6), 0.55, color=INK, lw=0)); t(19.2, 22.6, "measurement", 6.2, MUTED, ha="left")
ax.add_patch(Rectangle((29.3, 21.6), 2.0, 2.0, fc="#c9c7c1", ec="#c9c7c1")); t(31.9, 22.6, "carried-forward value", 6.2, MUTED, ha="left")
# result for (b)
t(48.2, 46.5, "Korean hospitals, 24-h AUROC\ngain over the source-only model", 6.4, MUTED, ha="left")
t(48.2, 38.0, "+0.045", 9.5, INK, ha="left", weight="bold"); t(55.2, 38.0, "(0.021 to 0.069)", 6.4, MUTED, ha="left")
t(48.2, 29.5, "+0.016", 9.5, MUTED, ha="left"); t(55.2, 29.5, "(−0.011 to 0.042)", 6.4, MUTED, ha="left")
t(48.2, 22.6, "same target rates in both rows", 6.2, FAINT, ha="left")

# ------------------------------------------------------------------ (c) same rates, different structure
t(66.5, 53.2, "(c)", 9, ha="left", weight="bold"); t(70.5, 53.2, "Same rate, different structure", 8.6, ha="left")
GX = [69.0, 84.5]; GW, GH = 0.95, 1.55
t(83.4, 51.2, "both: stream recorded in 40% of hours", 6.4, MUTED)
rs = np.random.RandomState(7)
stays_x = [True, False, True, False, False, True, False, False, True, False]
for gi, (gx, title, sub) in enumerate(zip(GX, ("hospital X", "hospital Y"), ("stream absent in\n6 of 10 stays", "every stay charted\nin 40% of hours"))):
    t(gx + 6.6, 48.4, title, 7.2, INK)
    for r in range(10):
        for c in range(14):
            on = stays_x[r] if gi == 0 else (rs.rand() < 0.4)
            ax.add_patch(Rectangle((gx + c * GW, 45.6 - r * GH), GW * 0.85, GH * 0.8, fc=INK if on else "white", ec=GRID, lw=0.35))
    t(gx + 6.6, 28.3, sub, 6.4, MUTED)
t(83.4, 52.0 - 1.5, "", 6)
t(69.0, 24.6, "coarsest profile (rates):", 6.6, MUTED, ha="left"); t(98.5, 24.6, "identical", 6.6, INK, ha="right")
t(69.0, 22.2, "three-scale profile:", 6.6, MUTED, ha="left"); t(98.5, 22.2, "different", 6.6, OR, ha="right", weight="bold")
t(69.0, 19.4, "Targets built with these two structures (Supplementary S20):", 6.3, MUTED, ha="left")
t(69.0, 17.2, "three-scale minus rate-level, 24-h AUROC +0.007 (0.003 to 0.010)", 6.3, INK, ha="left")
t(69.0, 15.2, "laboratory selection, buffered by carry-forward: +0.000", 6.3, MUTED, ha="left")
# footer: the orange convention
ax.add_patch(Rectangle((1.5, 9.0), 2.6, 1.8, fc=OR_F, ec=OR, lw=0.9)); t(4.8, 9.9, "designed in this work", 6.4, MUTED, ha="left")
ax.add_patch(Rectangle((19.5, 9.0), 2.6, 1.8, fc=BL_F, ec=BL, lw=0.9)); t(22.8, 9.9, "model trained by the procedure (any model; Section 4.4)", 6.4, MUTED, ha="left")
ax.set_ylim(7, 88)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig1_overview.{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
print("written", OUT)
