"""Fig. 1 for the Journal of Medical Systems manuscript: the JBHI overview with journal lettering (Arial-metric sans,
contrast of at least 4.5:1) in the design language of the ESWA Fig. 1 (orange = components of this work, blue = observed data and model; orange slashes = removed observations;
solid orange cells = recomputed carry-forward values). (a) pipeline, (b) site/stay/patch interventions,
(c) measurement-level re-observation versus entry-wise masking. Writes paper/JOMS/figures/Fig1.{pdf,eps,png}."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "JOMS")
OUT = os.path.join(ROOT, "figures")
for f in ("Regular", "Bold", "Italic", "BoldItalic"):
    font_manager.fontManager.addfont(os.path.join(ROOT, "fonts", f"LiberationSans-{f}.ttf"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold", "mathtext.fallback": "stixsans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED, FAINT, EDGE = "#1b1b1b", "#5c5b57", "#9a9893", "#b9b7b2"
OR, OR_F = "#e8622c", "#fbe3d7"
BL, BL_F = "#2a6fdb", "#e3ecfa"
W, H = 100.0, 56.0
fig = plt.figure(figsize=(6.85, 6.85 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc="white", ec=EDGE, lw=0.8, ls="-", r=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(x0, y0, x1, y1, c=INK, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.5, color=c, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


TXT = {OR: "#a33a0b", FAINT: MUTED, BL: "#1f5fc4"}          # lettering contrast of at least 4.5:1


def t(x, y, s, size=7.5, c=INK, ha="center", va="center", **k):
    ax.text(x, y, s, fontsize=size, color=TXT.get(c, c), ha=ha, va=va, **k)


def cell(x, y, s, state, letter=None):
    """state: 0 empty, 1 observed (blue), 2 removed (orange slash), 3 recomputed value (solid orange), 4 carried (light blue)"""
    fc, ec = {0: ("white", "#d6d4cf"), 1: (BL_F, BL), 2: ("white", OR), 3: (OR_F, OR), 4: (BL_F, BL)}[state]
    ax.add_patch(Rectangle((x, y), s, s, fc=fc, ec=ec, lw=0.55))
    if state == 2:
        ax.plot([x + 0.12 * s, x + 0.88 * s], [y + 0.12 * s, y + 0.88 * s], color=OR, lw=0.7)
    if letter:
        t(x + s / 2, y + s / 2, letter, 5.6, OR if state == 3 else BL)


def grid(x, y, pattern, s=1.1, gap=0.15):
    rows = len(pattern)
    for r, row in enumerate(pattern):
        for c, v in enumerate(row):
            cell(x + c * (s + gap), y + (rows - 1 - r) * (s + gap), s, v)


def hospital(x, y, s=1.0, c=OR):
    ax.add_patch(Rectangle((x, y), 6 * s, 5 * s, fc="white", ec=c, lw=1.0))
    ax.add_patch(Rectangle((x + 2.4 * s, y + 5 * s), 1.2 * s, 2.2 * s, fc=c, ec=c, lw=0))
    ax.add_patch(Rectangle((x + 1.9 * s, y + 5.55 * s), 2.2 * s, 1.1 * s, fc=c, ec=c, lw=0))
    for (i, j) in ((1, 3), (3.9, 3), (1, 1.2), (3.9, 1.2)):
        ax.add_patch(Rectangle((x + i * s, y + j * s), 1.1 * s, 1.0 * s, fc=c, ec=c, lw=0))
    ax.add_patch(Rectangle((x + 2.5 * s, y), 1.0 * s, 1.6 * s, fc=c, ec=c, lw=0))


def net(x, y, s=1.0, c=BL):
    cols = [[0.0, 1.6, 3.2], [0.0, 1.6, 3.2], [0.0, 1.6, 3.2]]
    pts = [[(x + i * 2.4 * s, y + v * s) for v in col] for i, col in enumerate(cols)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#c9d7f2", lw=0.35, zorder=1)
    for col in pts:
        for p in col:
            ax.add_patch(Circle(p, 0.42 * s, fc="white", ec=c, lw=0.8, zorder=2))


rs = np.random.RandomState(4)
SRC = (rs.rand(5, 8) < 0.72).astype(int)
# ================================================================ (a) pipeline
t(0.6, 54.6, "(a) Profile-guided re-observation", 8.4, ha="left")
# target hospital
hospital(2.2, 41.5, 0.95)
t(5.05, 38.6, "Target hospital", 7.2, OR)
t(5.05, 36.7, "unlabeled records", 6.6, MUTED)
t(5.05, 35.0, "no outcome labels", 6.6, MUTED)
arrow(9.6, 44.5, 12.6, 44.5, c=OR)
# observation profile
box(12.8, 34.0, 23.0, 18.2, fc=OR_F, ec=OR, lw=1.0)
t(24.3, 50.3, "Estimate the observation profile", 7.1, OR)
t(24.3, 47.3, r"$\hat{\pi}_h=(A_h,\ p_h,\ \rho_h,\ c_h)$", 8.6)
# icons: absence grid, selection sigmoid, cadence dots
grid(14.0, 38.6, [[1, 1, 0, 1], [2, 2, 2, 2], [0, 1, 1, 1]], s=1.05)
ax.plot([20.6, 20.6, 27.0], [43.6, 38.5, 38.5], color=INK, lw=0.55)
xs = np.linspace(20.9, 26.8, 40)
ax.plot(xs, 38.9 + 4.2 / (1 + np.exp(-(xs - 23.9) * 1.6)), color=OR, lw=1.0)
for r, ks in enumerate(([1] * 7, [1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1])):
    yy = 42.8 - r * 1.8
    ax.plot([28.6, 34.6], [yy, yy], color=OR, lw=0.5)
    for j, v in enumerate(ks):
        ax.add_patch(Circle((28.6 + j * 1.0, yy), 0.3, fc=OR if v else "white", ec=OR, lw=0.5))
for xx, a_, b_ in ((16.4, "Site", "absence"), (23.9, "Stay", "selection"), (31.6, "Patch", "cadence")):
    t(xx, 36.6, a_, 6.9, INK)
    t(xx, 35.1, b_, 6.3, MUTED)
# diagnosis (auxiliary)
box(12.8, 27.4, 23.0, 5.0, fc="white", ec=FAINT, lw=0.7, ls=(0, (3, 2)))
t(24.3, 31.0, "Damage diagnosis (auxiliary)", 6.9, MUTED)
t(24.3, 29.0, "one component at a time", 6.3, FAINT)
ax.plot([24.3, 24.3], [34.0, 32.4], color=FAINT, lw=0.6, ls=(0, (2, 1.5)))
arrow(35.8, 44.5, 38.6, 44.5, c=OR)
# re-observe measurements
box(38.8, 34.0, 23.6, 18.2, fc=OR_F, ec=OR, lw=1.0)
t(50.6, 50.3, "Re-observe measurements", 7.3, OR)
t(44.0, 47.6, "Development", 6.4, MUTED)
t(57.2, 47.6, "Re-observed", 6.4, MUTED)
grid(39.9, 39.2, SRC, s=1.0, gap=0.14)
REO = SRC.copy()
REO[1, :] = np.where(SRC[1, :] == 1, 2, 0)                  # absent variable
REO[3, [1, 3, 5]] = np.where(SRC[3, [1, 3, 5]] == 1, 2, 0)   # thinned cadence
REO[[0, 4], 6] = np.where(SRC[[0, 4], 6] == 1, 2, 0)
grid(52.9, 39.2, REO, s=1.0, gap=0.14)
arrow(49.4, 41.9, 52.3, 41.9, c=OR)
t(50.6, 37.3, r"$\tilde{R}_i=T_{\hat{\pi}}(R_i)$", 8.4)
t(50.6, 35.2, "same patient, source label kept", 6.3, MUTED)
t(50.6, 29.8, r"development patient $R_i$", 6.7, BL)
arrow(50.6, 31.0, 50.6, 33.8, c=BL)
arrow(62.4, 44.5, 65.0, 44.5, c=OR)
# rebuild the input
box(65.2, 34.0, 18.4, 18.2, fc="white", ec=EDGE, lw=0.9)
t(74.4, 50.3, "Rebuild the hourly input", 7.2)
t(66.4, 46.8, "Raw", 6.4, MUTED, ha="left")
ax.plot([70.2, 82.0], [46.8, 46.8], color=FAINT, lw=0.5)
for xx, lab, col in ((70.6, "a", BL), (76.1, "b", OR), (81.6, "c", BL)):
    ax.add_patch(Circle((xx, 46.8), 0.5, fc=BL_F if col == BL else "white", ec=col, lw=0.8))
    t(xx, 48.3, lab, 6.0, col)
ax.plot([75.75, 76.45], [46.45, 47.15], color=OR, lw=0.8)
s_ = 1.26
t(66.4, 43.1, "Before", 6.4, MUTED, ha="left")
for j, l in enumerate("aaabbbbc"):
    cell(71.2 + j * (s_ + 0.12), 42.45, s_, 4, l)
t(66.4, 40.2, "After", 6.4, MUTED, ha="left")
for j, l in enumerate("aaaaaaac"):
    cell(71.2 + j * (s_ + 0.12), 39.55, s_, 3 if 3 <= j <= 6 else 4, l)
arrow(76.1, 46.0, 76.1, 44.15, c=OR, lw=0.6)
t(74.4, 37.3, r"$\tilde{X}_i=g(\tilde{R}_i)$", 8.4)
t(74.4, 35.2, "carry-forward, masks rebuilt", 6.1, MUTED)
arrow(83.6, 44.5, 86.0, 44.5)
# model
net(86.6, 42.9, 0.85)
t(88.6, 40.6, r"$f_\theta$", 9.0, BL)
t(89.2, 38.4, "factual + re-observed,", 5.9, MUTED)
t(89.2, 36.9, r"shared $\theta$, same outcome", 5.9, MUTED)
arrow(91.4, 44.5, 93.4, 44.5)
t(96.8, 45.6, "Risk at", 6.8)
t(96.8, 44.1, "the target", 6.8)
t(96.8, 42.4, r"$F_1,\ldots,F_4$", 7.6)
ax.plot([0.6, 99.4], [25.6, 25.6], color="#e6e4df", lw=0.6)

# ================================================================ (b) site / stay / patch
t(0.6, 24.2, "(b) What the profile changes", 8.4, ha="left")
before = [[1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 1, 1]]
cols = [("Site", "variable never recorded", [[1, 1, 1, 1, 1, 1], [2, 0, 2, 2, 0, 2], [1, 1, 1, 0, 1, 1]],
         r"$v\in A_h\ \Rightarrow\ \tilde{O}_{tv}=0$", None),
        ("Stay", "who is tested; which streams", [[1, 1, 1, 1, 1, 1], [2, 0, 2, 2, 0, 2], [2, 2, 2, 0, 2, 2]],
         r"$q^{\mathrm{test}}_{hv}=\min\!\left(1,\frac{p_{hv}(a)}{p_{\mathrm{dev},v}(a)}\right)$",
         r"$q^{\mathrm{stream}}_{hv}=\min\!\left(1,\rho_{hv}/\rho_{\mathrm{dev},v}\right)$"),
        ("Patch", "how often, within the stay", [[1, 2, 1, 2, 1, 2], [1, 0, 2, 1, 0, 2], [1, 2, 1, 0, 2, 1]],
         r"$q^{\mathrm{cad}}_{hv}=\min\!\left(1,c_{hv}/c_{\mathrm{dev},v}\right)$", None)]
for k, (name, sub, after, f1, f2) in enumerate(cols):
    x0 = 1.0 + k * 20.0
    t(x0, 21.6, name, 7.6, OR, ha="left")
    t(x0, 19.8, sub, 6.3, MUTED, ha="left")
    grid(x0, 13.2, before, s=1.05, gap=0.13)
    arrow(x0 + 8.0, 14.9, x0 + 9.8, 14.9, c=OR)
    grid(x0 + 10.4, 13.2, after, s=1.05, gap=0.13)
    t(x0 + 8.6, 9.6, f1, 7.6)
    if f2:
        t(x0 + 8.6, 6.2, f2, 7.6)
ax.plot([61.2, 61.2], [3.6, 24.6], color="#e6e4df", lw=0.6)

# ================================================================ (c) re-observation versus entry masking
t(62.4, 24.2, "(c) Re-observation is not entry masking", 8.4, ha="left")
X0, D = 70.6, 1.92
rows = [("development", 18.6), ("re-observed", 13.8), ("entry-masked", 9.0)]
for hh in range(12):
    t(X0 + hh * D + D * 0.45, 21.6, f"{hh + 1}", 5.6, FAINT)
for lab, y in rows:
    t(69.6, y + 0.85, lab, 6.5, OR if lab == "re-observed" else MUTED, ha="right")
    for hh in range(12):
        ax.add_patch(Rectangle((X0 + hh * D, y), D * 0.9, 1.7, fc="white", ec="#d6d4cf", lw=0.5))


def run(y, start, n, fc, ec):
    for k in range(n):
        ax.add_patch(Rectangle((X0 + (start + k) * D, y), D * 0.9, 1.7, fc=fc, ec=ec, lw=0.5))


def meas(y, h, removed=False):
    xc = X0 + h * D + D * 0.45
    if removed:
        ax.add_patch(Circle((xc, y + 0.85), 0.5, fc="white", ec=OR, lw=0.8))
        ax.plot([xc - 0.35, xc + 0.35], [y + 0.5, y + 1.2], color=OR, lw=0.8)
    else:
        ax.add_patch(Circle((xc, y + 0.85), 0.5, fc=BL, ec=BL, lw=0))


for m in (0, 4, 8):                                         # development: measurements carried forward 4 h
    run(18.6, m, 4, BL_F, BL); meas(18.6, m)
for m in (0, 8):                                            # re-observed: hour-5 measurement removed with its copies
    run(13.8, m, 4, BL_F, BL); meas(13.8, m)
meas(13.8, 4, removed=True)
run(13.8, 4, 4, "white", OR)
for m in (0, 4, 8):                                         # entry-wise masking: independent holes, carried values kept
    run(9.0, m, 4, BL_F, BL)
for hh in (1, 4, 6, 10):
    x = X0 + hh * D
    ax.add_patch(Rectangle((x, 9.0), D * 0.9, 1.7, fc="white", ec=FAINT, lw=0.5))
    ax.plot([x + 0.3, x + D * 0.9 - 0.3], [9.25, 10.45], color=FAINT, lw=0.6)
    ax.plot([x + 0.3, x + D * 0.9 - 0.3], [10.45, 9.25], color=FAINT, lw=0.6)
for m in (0, 8):
    meas(9.0, m)
t(X0 + 6 * D, 12.6, "a measurement and its carried copies removed; input rebuilt", 6.0, OR)
t(X0 + 6 * D, 7.8, "independent holes; carried values of removed tests remain", 6.0, MUTED)
ax.add_patch(Circle((63.4, 4.6), 0.5, fc=BL, ec=BL, lw=0)); t(64.4, 4.6, "measurement", 6.2, MUTED, ha="left")
ax.add_patch(Rectangle((73.4, 3.75), 1.7, 1.7, fc=BL_F, ec=BL, lw=0.5)); t(75.6, 4.6, "carried-forward value", 6.2, MUTED, ha="left")

for ext, kw in (("pdf", {}), ("eps", {}), ("png", {"dpi": 1200})):
    fig.savefig(os.path.join(OUT, f"Fig1.{ext}"), bbox_inches="tight", pad_inches=0.02, **kw)
print("written", OUT)
