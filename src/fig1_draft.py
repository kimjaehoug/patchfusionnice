"""Figure 1, step 2 (structure draft): story = (a) problem -> (b) the procedure in three numbered steps -> (c) what
re-observation produces. Plain boxes, arrows and dot timelines only; styling comes in step 3.
Writes paper/ESWA_draft/figures/draft/Fig1_draft.png."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "ESWA_draft", "figures", "draft"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
INK, MUTED, FAINT, GRID, OR, OR_F, BL, BL_F = "#1b1b1b", "#5c5b57", "#a3a19c", "#dddbd6", "#e8622c", "#fde6da", "#2a6fdb", "#e2ecfb"
fig = plt.figure(figsize=(7.2, 4.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")


def box(x, y, w, h, fc="white", ec=GRID, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, c=INK, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=c, lw=lw, shrinkA=0, shrinkB=0))


def t(x, y, s, size=7, c=INK, ha="center", **k):
    ax.text(x, y, s, fontsize=size, color=c, ha=ha, va="center", **k)


def timeline(x, y, w, rows, label=None):
    """rows: list of 0/1 arrays (one per variable); draws dots on a faint line"""
    n = len(rows[0]); dx = w / n
    for r, row in enumerate(rows):
        yy = y - r * 1.5
        ax.plot([x, x + w], [yy, yy], color=GRID, lw=0.5, zorder=1)
        for j, v in enumerate(row):
            if v: ax.add_patch(Circle((x + (j + 0.5) * dx, yy), 0.33, color=INK, lw=0, zorder=2))
    if label: t(x - 0.8, y - (len(rows) - 1) * 0.75, label, 6.4, MUTED, ha="right")


rs = np.random.RandomState(1)
N = 16
dense = [np.ones(N, int), (rs.rand(N) < 0.85).astype(int), (rs.rand(N) < 0.6).astype(int)]
sparse = [np.ones(N, int), (rs.rand(N) < 0.15).astype(int), np.zeros(N, int)]

# ---------------------------------------------------------------- (a) problem
t(0.8, 62.0, "(a) The problem", 8.2, ha="left", weight="bold")
t(0.8, 59.0, "Hospitals record the same patient differently", 6.8, MUTED, ha="left")
timeline(10.0, 54.6, 17.0, dense, "development\nhospital")
timeline(10.0, 46.0, 17.0, sparse, "target\nhospital")
box(9.0, 33.6, 9.5, 5.4, BL_F, BL); t(13.75, 36.3, "model", 7)
t(13.75, 31.6, "learned on dense records", 6.0, MUTED)
arrow(18.5, 36.3, 22.0, 36.3)
t(26.0, 37.4, "applied to", 6.0, MUTED); t(26.0, 35.2, "sparse records", 6.0, MUTED)
t(18.5, 27.0, "→ loses accuracy at the target", 6.8, OR)
ax.plot([31.5, 31.5], [24, 63], color=GRID, lw=0.8)

# ---------------------------------------------------------------- (b) procedure, three steps
t(33.5, 62.0, "(b) Our procedure: read → re-observe → train", 8.2, ha="left", weight="bold")
lanes = [("①", "Read how the target hospital observes", "no outcome labels", 55.0),
         ("②", "Re-observe development patients the same way", "labels kept", 44.0),
         ("③", "Train on both records, deploy at the target", "", 33.0)]
for num, title, note, y in lanes:
    t(33.5, y + 3.2, num, 8.4, OR if num != "③" else BL, ha="left", weight="bold")
    t(36.3, y + 3.2, title, 7.0, ha="left")
    if note: t(98.8, y + 3.2, note, 6.2, OR, ha="right")
# step 1: target record -> profile
timeline(36.5, 55.6, 14.0, sparse); t(43.5, 50.9, "target records", 6.0, MUTED)
arrow(51.5, 54.1, 56.5, 54.1)
box(56.5, 50.8, 21.0, 6.6, OR_F, OR, 1.1)
t(67.0, 55.6, "observation profile", 7.0, weight="bold")
t(67.0, 52.6, "what · for whom · how often", 6.2, MUTED)
# step 2: development record -> re-observe -> re-observed record
timeline(36.5, 44.6, 14.0, dense); t(43.5, 39.9, "development records", 6.0, MUTED)
arrow(51.5, 43.1, 60.0, 43.1)
box(60.0, 40.2, 14.0, 5.8, OR_F, OR, 1.1); t(67.0, 43.1, "re-observe", 7.0, weight="bold")
arrow(67.0, 50.8, 67.0, 46.0, c=OR)
arrow(74.0, 43.1, 80.5, 43.1)
reob = [dense[0], dense[1] * (rs.rand(N) < 0.18), np.zeros(N, int)]
timeline(81.0, 44.6, 14.0, reob); t(88.0, 39.9, "looks like the target", 6.0, OR)
# step 3: both records -> model -> target hospital
box(36.5, 28.8, 14.0, 5.6, "white", GRID); t(43.5, 31.6, "development +\nre-observed", 6.2)
arrow(50.5, 31.6, 60.0, 31.6)
box(60.0, 28.8, 14.0, 5.6, BL_F, BL); t(67.0, 31.6, "model", 7.0)
arrow(74.0, 31.6, 80.5, 31.6)
box(80.5, 28.8, 14.5, 5.6, "white", GRID); t(87.75, 31.6, "target hospital", 6.4)
ax.plot([0.8, 99.2], [24.0, 24.0], color=GRID, lw=0.8)

# ---------------------------------------------------------------- (c) what re-observation produces
t(0.8, 21.6, "(c) What re-observation produces", 8.2, ha="left", weight="bold")
t(0.8, 18.8, "A test removed takes its carried-forward hours with it, so the record looks like one a real hospital keeps", 6.6, MUTED, ha="left")
CW = 1.55
def strip(x, y, cells, title, tc=INK):
    for j, c in enumerate(cells):
        fc = {0: "white", 1: INK, 2: "#cfcdc7", 3: OR, 4: "#f6c9b4"}[c]
        ax.add_patch(Rectangle((x + j * CW, y), CW * 0.9, 2.6, fc=fc, ec=GRID, lw=0.4))
    t(x + len(cells) * CW / 2, y + 4.4, title, 6.8, tc)
base = [1, 2, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 2, 2, 0, 0]
reo = [1, 2, 2, 2, 0, 3, 3, 3, 3, 0, 1, 2, 2, 2, 0, 0]
msk = [1, 4, 2, 4, 0, 1, 4, 2, 4, 0, 4, 2, 4, 2, 0, 0]
strip(3.0, 8.0, base, "development record")
strip(36.0, 8.0, reo, "re-observed: a hospital-like record", OR)
strip(69.0, 8.0, msk, "random masking: holes no hospital makes", MUTED)
for k, (c, lab) in enumerate([(INK, "test"), ("#cfcdc7", "carried-forward value"), (OR, "removed with its carried values"), ("#f6c9b4", "random hole")]):
    ax.add_patch(Rectangle((3.0 + k * 24.0, 2.2), 1.4, 1.6, fc=c, ec=GRID, lw=0.4)); t(5.0 + k * 24.0, 3.0, lab, 6.0, MUTED, ha="left")
fig.savefig(os.path.join(OUT, "Fig1_draft.png"), dpi=250, bbox_inches="tight", pad_inches=0.04)
print("written")
