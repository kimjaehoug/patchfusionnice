"""Fig. 1 (novelty-focused): three-scale observation imagination on a record grid, aligned 6-h units in the encoder,
and the contrast with reweighting and random augmentation. Schematic; the record is synthetic."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "/home/jbnu/patchfusionnice/paper/ESWA_draft/figures/Figure1_overview_v12.png"
INK, MUTED, LINE, FILL = "#0b0b0b", "#52514e", "#8a8984", "#f2f1ee"
SITE, STAY, PATCH = "#008300", "#e87ba4", "#2a78d6"          # validated categorical slots (dataviz validator)
PATCHF = "#e3eefb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2})
rng = np.random.RandomState(3)
ROWS = ["Heart rate", "SpO2", "Lactate", "Creatinine", "GCS", "Urine output"]
T = 48
obs = np.zeros((len(ROWS), T), bool)
obs[0] = True; obs[1] = rng.rand(T) < 0.9
obs[2, [3, 9, 15, 20, 27, 33, 40, 45]] = True
obs[3, [5, 17, 29, 41]] = True
obs[4] = rng.rand(T) < 0.45; obs[5] = rng.rand(T) < 0.7
# imagined: site = GCS never recorded; stay = lactate not tested in this stay; patch = cadence thinned within patches
img = obs.copy(); how = np.zeros_like(obs, int)                    # 1 site, 2 stay, 3 patch
img[4] = False; how[4][obs[4]] = 1
img[2] = False; how[2][obs[2]] = 2
for p in range(8):
    for r in (1, 5):
        hrs = [t for t in range(p * 6, p * 6 + 6) if obs[r, t]]
        drop = [t for t in hrs if rng.rand() < 0.55] if p in (1, 3, 4, 6) else []
        for t in drop:
            img[r, t] = False; how[r, t] = 3


def box(ax, x, y, w, h, text, fc="#ffffff", ec=LINE, size=7.2, bold=False, color=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.01", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight="bold" if bold else "normal", linespacing=1.22)


def arrow(ax, x0, y0, x1, y1, color=LINE, lw=1.1, ms=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0))


def grid(ax, x0, y0, w, h, M, how=None, title=None):
    cw, rh = w / T, h / len(ROWS)
    for p in range(8):
        ax.add_patch(Rectangle((x0 + p * 6 * cw, y0), 6 * cw, h, fc=FILL if p % 2 == 0 else "#ffffff", ec="none"))
    for p in range(9):
        ax.plot([x0 + p * 6 * cw] * 2, [y0 - 0.006, y0 + h + 0.006], color=PATCH, lw=0.9)
    for r in range(len(ROWS)):
        yy = y0 + h - (r + 0.5) * rh
        for t in range(T):
            xx = x0 + (t + 0.5) * cw
            if M[r, t]:
                ax.plot(xx, yy, "o", ms=2.0, color=INK, mec="none")
            elif how is not None and how[r, t]:
                col = {1: SITE, 2: STAY, 3: PATCH}[how[r, t]]
                ax.plot(xx, yy, "o", ms=2.3, mfc="none", mec=col, mew=0.7)
    if title:
        ax.text(x0 + w / 2, y0 + h + 0.02, title, ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=INK)


fig = plt.figure(figsize=(7.2, 7.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
# ---------------- (a) imagination on the record
ax.text(0.01, 0.995, "(a) Approximating how the target hospital would observe a development patient, at three scales", fontsize=8.6, fontweight="bold", va="top")
gx, gy, gw, gh = 0.115, 0.715, 0.3, 0.2
for r, nm in enumerate(ROWS):
    ax.text(gx - 0.01, gy + gh - (r + 0.5) * gh / len(ROWS), nm, ha="right", va="center", fontsize=6.6, color=MUTED)
grid(ax, gx, gy, gw, gh, obs, title="Development record (MIMIC-IV), 48 h")
gx2 = 0.555
for r, nm in enumerate(ROWS):
    ax.text(gx2 - 0.01, gy + gh - (r + 0.5) * gh / len(ROWS), nm, ha="right", va="center", fontsize=6.6, color=MUTED)
grid(ax, gx2, gy, gw, gh, img, how, title="Imagined record (approximating the target)")
arrow(ax, gx + gw + 0.012, gy + gh / 2, gx2 - 0.075, gy + gh / 2, color=INK, lw=1.3, ms=10)
ax.text((gx + gw + gx2 - 0.075) / 2 + 0.006, gy + gh / 2 + 0.035, "target\nprofile", ha="center", fontsize=6.8, color=INK)
ax.text((gx + gw + gx2 - 0.075) / 2 + 0.006, gy + gh / 2 - 0.05, "no outcome\nlabels", ha="center", fontsize=6.3, color=MUTED)
# brackets for scales on the right of the imagined grid
rh = gh / len(ROWS)
for r, col, lab in ((4, SITE, "Site: never\nrecorded here"), (2, STAY, "Stay: not tested\nat this acuity"), (5, PATCH, "Patch: lower\nhourly cadence")):
    yy = gy + gh - (r + 0.5) * rh
    ax.plot([gx2 + gw + 0.005, gx2 + gw + 0.02], [yy, yy], color=col, lw=1.2)
    ax.text(gx2 + gw + 0.024, yy, lab, va="center", fontsize=6.2, color=col, fontweight="bold", linespacing=1.1)
ax.plot([], [], "o", ms=3, color=INK, label="recorded"); ax.plot([], [], "o", ms=3, mfc="none", mec=MUTED, label="removed by imagination")
ax.legend(loc="upper left", bbox_to_anchor=(0.115, 0.705), ncol=2, frameon=False, fontsize=6.4, handletextpad=0.3, columnspacing=1.2)
ax.text(gx, 0.663, "Profile estimated from the target's unlabeled records; each training chunk uses one target hospital's profile (or a random profile).", fontsize=6.4, color=MUTED)
# ---------------- (b) aligned unit in the encoder
ax.text(0.01, 0.635, "(b) State resets every 6 h: an imagined change acts only on tokens of patches whose input changed", fontsize=8.6, fontweight="bold", va="top")
ex, ew = 0.13, 0.62
cw = ew / T
yin = 0.545
for t in range(T):
    changed = (how[:, t] == 3).any()
    ax.add_patch(Rectangle((ex + t * cw, yin), cw * 0.92, 0.03, fc=PATCHF if changed else "#ffffff", ec=PATCH if changed else LINE, lw=0.5))
for p in range(9):
    ax.plot([ex + p * 6 * cw] * 2, [yin - 0.008, yin + 0.04], color=PATCH, lw=1.2)
ax.text(ex - 0.01, yin + 0.015, "hourly input", ha="right", va="center", fontsize=6.6, color=MUTED)
ybl = 0.455
for p in range(8):
    hit = p in (1, 3, 4, 6)
    box(ax, ex + p * 6 * cw + 0.002, ybl, 6 * cw - 0.006, 0.05, "BiLSTM", fc=PATCHF if hit else "#ffffff", ec=PATCH, size=6.2)
    arrow(ax, ex + p * 6 * cw + 3 * cw, yin - 0.002, ex + p * 6 * cw + 3 * cw, ybl + 0.05, lw=0.7, ms=6)
ax.text(ex - 0.01, ybl + 0.025, "reset BiLSTM\n(shared weights)", ha="right", va="center", fontsize=6.6, color=MUTED)
ytk = 0.375
for k in range(16):
    hit = (k // 2) in (1, 3, 4, 6)
    ax.add_patch(Rectangle((ex + k * 3 * cw + 0.003, ytk), 3 * cw - 0.006, 0.035, fc=PATCH if hit else "#ffffff", ec=PATCH, lw=0.7))
    if k % 2 == 0:
        arrow(ax, ex + (k // 2) * 6 * cw + 3 * cw, ybl, ex + (k // 2) * 6 * cw + 3 * cw, ytk + 0.035, lw=0.7, ms=6)
ax.text(ex - 0.01, ytk + 0.017, "16 tokens", ha="right", va="center", fontsize=6.6, color=MUTED)
ax.text(ex + ew + 0.012, ytk + 0.017, "filled: patches whose\ncadence was thinned;\nsite- and stay-scale\nremovals reach all patches", va="center", fontsize=6.2, color=PATCH, linespacing=1.1)
box(ax, ex, 0.29, ew, 0.045, "Global attention (2 layers, rotary positions) → mean pooling", fc=FILL, size=6.8)
arrow(ax, ex + ew / 2, ytk, ex + ew / 2, 0.335, ms=7)
box(ax, ex, 0.225, ew, 0.045, "Competing-risk head: death within 24 h, 48 h, 7 days, before discharge", size=6.8)
arrow(ax, ex + ew / 2, 0.29, ex + ew / 2, 0.27, ms=7)
ax.text(ex + ew + 0.012, yin + 0.015, "patch-scale imagination\n(cadence) shaded", va="center", fontsize=6.2, color=PATCH, linespacing=1.1)
ax.text(ex + ew + 0.012, ybl + 0.025, "no recurrent state\ncrosses a boundary", va="center", fontsize=6.2, color=PATCH, linespacing=1.1)
# ---------------- (c) contrast
ax.text(0.01, 0.19, "(c) How the target hospital's observation process enters training", fontsize=8.6, fontweight="bold", va="top")
cy, chh = 0.015, 0.145
box(ax, 0.01, cy, 0.315, chh, "Importance reweighting\n\nweights development samples towards\nthe target's observation patterns;\ncannot represent a variable the target\nnever records (no support)")
box(ax, 0.3425, cy, 0.315, chh, "Random augmentation\n\nthins or masks observations at random;\nnot matched to how the target hospital\nselects, charts and samples")
box(ax, 0.675, cy, 0.315, chh, "Patch-level imagination (this work)\n\ntarget profile at site, stay and patch\nscale, estimated without labels; with the\nreset, a change acts only on patches\nwhose input it changes", fc=PATCHF, ec=PATCH, lw=1.2)
fig.savefig(OUT, dpi=300); plt.close(fig)
print("written", OUT)
