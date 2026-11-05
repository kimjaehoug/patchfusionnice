"""Schematic figures: Fig. 1 method overview (observation imagination + PatchFusion encoder), Fig. 2 evaluation design."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "/home/jbnu/patchfusionnice/paper/ESWA_draft/figures"
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8984"
FILL, ACC, ACCF = "#f2f1ee", "#2a78d6", "#e3eefb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})


def box(ax, x, y, w, h, text, fc=FILL, ec=LINE, bold=False, size=7.5, color=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=LINE, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, shrinkA=0, shrinkB=0))


def label(ax, x, y, t):
    ax.text(x, y, t, fontsize=9, fontweight="bold", color=INK, ha="left", va="top")


# ---------------------------------------------------------------- Figure 1
fig = plt.figure(figsize=(7.2, 6.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
label(ax, 0.01, 0.995, "(a) Patch-level observation imagination")
yb, hb = 0.68, 0.27
box(ax, 0.01, yb, 0.15, hb, "Target hospital\n\nunlabeled\nrecords\n(no outcome\nlabels)", fc="#ffffff", size=6.8)
arrow(ax, 0.16, yb + hb / 2, 0.19, yb + hb / 2)
# profile with three scales
ax.add_patch(FancyBboxPatch((0.19, yb), 0.27, hb, boxstyle="round,pad=0.004,rounding_size=0.012", fc=FILL, ec=LINE, lw=0.9))
ax.text(0.325, yb + hb - 0.03, "Label-free observation profile", ha="center", fontsize=7.5, fontweight="bold", color=INK)
for k, (sc, desc) in enumerate((("Site", "variables never recorded"), ("Stay", "test selection by acuity;\ncharting streams present"), ("Patch", "testing, charting and\nvital-sign cadence"))):
    yy = yb + hb - 0.085 - k * 0.06
    box(ax, 0.20, yy, 0.06, 0.05, sc, fc=ACCF if sc == "Patch" else "#ffffff", ec=ACC if sc == "Patch" else LINE, bold=True, size=7)
    ax.text(0.27, yy + 0.025, desc, ha="left", va="center", fontsize=6.6, color=MUTED, linespacing=1.15)
arrow(ax, 0.46, yb + hb / 2, 0.49, yb + hb / 2)
box(ax, 0.49, yb, 0.17, hb, "Re-observe\ndevelopment patients\n(MIMIC-IV) as the\ntarget would\n\nselection per stay,\ncadence per 6-h patch,\nre-impute", fc=ACCF, ec=ACC, size=6.8)
arrow(ax, 0.66, yb + hb / 2, 0.69, yb + hb / 2)
box(ax, 0.69, yb, 0.14, hb, "Train PatchFusion\non factual +\nimagined views\n\n50% target,\n50% random\nprofiles", fc="#ffffff", size=6.8)
arrow(ax, 0.83, yb + hb / 2, 0.86, yb + hb / 2)
box(ax, 0.86, yb, 0.13, hb, "Apply\nunchanged to\nthe target\nhospital\n\ndeath within\n24 h to\nin-hospital", fc="#ffffff", size=6.8)
# (b) encoder
label(ax, 0.01, 0.585, "(b) PatchFusion encoder at a landmark L")
y0, cw = 0.43, 0.0145
ax.text(0.01, y0 + 0.032, "48-h input\n(88 channels)", fontsize=6.6, color=MUTED, va="center")
x0 = 0.11
for t in range(48):
    ax.add_patch(Rectangle((x0 + t * cw, y0 + 0.01), cw * 0.9, 0.045, fc="#ffffff" if (t // 6) % 2 else FILL, ec=LINE, lw=0.4))
for p in range(9):
    xx = x0 + p * 6 * cw - cw * 0.05
    ax.plot([xx, xx], [y0 - 0.005, y0 + 0.07], color=ACC, lw=1.2)
ax.text(x0 + 48 * cw + 0.01, y0 + 0.032, "state reset\nevery 6 h", fontsize=6.6, color=ACC, va="center")
# BiLSTM per patch
yl = 0.31
for p in range(8):
    box(ax, x0 + p * 6 * cw + 0.003, yl, 6 * cw - 0.008, 0.06, "BiLSTM", fc=ACCF, ec=ACC, size=6.3)
    arrow(ax, x0 + p * 6 * cw + 3 * cw, y0 + 0.008, x0 + p * 6 * cw + 3 * cw, yl + 0.06, color=LINE, lw=0.7)
ax.text(x0 + 48 * cw + 0.01, yl + 0.03, "shared weights,\nno state across patches", fontsize=6.6, color=MUTED, va="center")
# tokens
yt = 0.21
for k in range(16):
    ax.add_patch(Rectangle((x0 + k * 3 * cw + 0.004, yt), 3 * cw - 0.008, 0.04, fc="#ffffff", ec=ACC, lw=0.7))
    if k % 2 == 0:
        arrow(ax, x0 + (k // 2) * 6 * cw + 3 * cw, yl, x0 + (k // 2) * 6 * cw + 3 * cw, yt + 0.04, color=LINE, lw=0.7)
ax.text(x0 + 48 * cw + 0.01, yt + 0.02, "16 tokens\n(one per 3-h block)", fontsize=6.6, color=MUTED, va="center")
box(ax, x0, 0.105, 48 * cw, 0.06, "Global attention (2 layers, rotary positions) → mean pooling", fc=FILL)
arrow(ax, x0 + 24 * cw, yt, x0 + 24 * cw, 0.165)
box(ax, x0, 0.01, 48 * cw, 0.065, "Competing-risk head: death in (L, L+24], (L+24, L+48], (L+48, L+168] h and before discharge", fc="#ffffff")
arrow(ax, x0 + 24 * cw, 0.105, x0 + 24 * cw, 0.075)
fig.savefig(f"{OUT}/Figure1_overview.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig = plt.figure(figsize=(7.2, 4.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
label(ax, 0.01, 0.995, "(a) Data sources and roles")
box(ax, 0.01, 0.71, 0.22, 0.2, "MIMIC-IV\n(development)\n\n50,764 stays\ntraining; held-out fold\nfor development metrics\nand alert thresholds", fc="#ffffff")
ax.add_patch(FancyBboxPatch((0.25, 0.71), 0.46, 0.2, boxstyle="round,pad=0.004,rounding_size=0.012", fc=FILL, ec=LINE, lw=0.9))
ax.text(0.48, 0.885, "eICU-CRD: 207 hospitals, 103,491 stays", ha="center", fontsize=7.2, fontweight="bold", color=INK)
box(ax, 0.26, 0.725, 0.14, 0.13, "Low testing\nadaptation target", fc=ACCF, ec=ACC)
box(ax, 0.41, 0.725, 0.14, 0.13, "Middle testing\nconfirmation:\nnot tuned for", fc="#ffffff")
box(ax, 0.56, 0.725, 0.14, 0.13, "All hospitals\nsource-only and\noverall results", fc="#ffffff")
box(ax, 0.73, 0.71, 0.26, 0.2, "Hallym (4 Korean hospitals)\n1,102 stays\n\nadaptation target;\n7 variables never recorded", fc=ACCF, ec=ACC)
# (b) landmarks
label(ax, 0.01, 0.645, "(b) Dynamic landmarks and horizons")
yt, xs, xe = 0.47, 0.06, 0.94
sc = lambda hrs: xs + (xe - xs) * hrs / 120
ax.plot([xs, xe], [yt, yt], color=INK, lw=1)
for L_ in (0, 12, 24, 36, 48, 60, 72):
    ax.plot([sc(L_)] * 2, [yt - 0.012, yt + 0.012], color=INK, lw=1)
    ax.text(sc(L_), yt - 0.03, "ICU start" if L_ == 0 else f"{L_} h", ha="center", va="top", fontsize=6.6, color=MUTED)
ax.text(sc(108), yt - 0.03, "hours since ICU-level care began", ha="center", va="top", fontsize=6.3, color=MUTED)
L = 60
ax.add_patch(Rectangle((sc(L - 48), yt + 0.03), sc(L) - sc(L - 48), 0.055, fc=ACCF, ec=ACC, lw=0.8))
ax.text((sc(L - 48) + sc(L)) / 2, yt + 0.0575, "48-h look-back window at landmark L = 60 h", ha="center", va="center", fontsize=6.6, color=INK)
ax.plot([sc(L)] * 2, [yt, yt + 0.16], color=ACC, lw=0.8, ls=(0, (2, 2)))
for k, (hrs, t) in enumerate(((24, "death within 24 h"), (48, "within 48 h"))):
    yy = yt + 0.045 + k * 0.045
    ax.add_patch(FancyArrowPatch((sc(L), yy), (sc(L + hrs), yy), arrowstyle="-|>", mutation_scale=7, color=LINE, lw=1))
    ax.text(sc(L + hrs) + 0.006, yy, t, fontsize=6.6, color=MUTED, va="center")
ax.text(sc(L) + 0.006, yt + 0.135, "also within 7 days and before hospital discharge", fontsize=6.6, color=MUTED, va="center")
ax.text(0.06, yt - 0.085, "Landmarks every 12 h from 12 to 72 h; one sample per landmark while the patient is alive and in the ICU; hours before ICU-level care are empty.", ha="left", fontsize=6.3, color=MUTED)
# (c) regimes and measures
label(ax, 0.01, 0.37, "(c) Evaluation regimes and measures")
box(ax, 0.01, 0.05, 0.3, 0.27, "Frozen source-only evaluation\n\ntrained on MIMIC-IV only,\napplied unchanged\n\nencoder design study (Table 2),\nbackbone", fc="#ffffff")
box(ax, 0.33, 0.05, 0.3, 0.27, "Target-label-free adaptation\n\nprofile from the target's\nunlabeled records; retrained on\nre-observed development data;\nno target label in training,\nmodel selection or epochs", fc=ACCF, ec=ACC)
box(ax, 0.65, 0.05, 0.34, 0.27, "Measures\n\nDiscrimination: AUROC, AUPRC\nCalibration: O/E, intercept, slope, Brier\nAlerts: threshold fixed on development data;\nlandmark and stay level; net benefit\nUncertainty: cluster bootstrap\n(hospitals / patients)", fc="#ffffff")
fig.savefig(f"{OUT}/Figure2_evaluation.png", dpi=300); plt.close(fig)
print("written schematics")
