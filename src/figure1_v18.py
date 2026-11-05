"""Figure 1 (v18): profile-guided re-observation, drawn after the reference style: a compact forward path with small
glyphs and annotations under each block (a), and zooms on the two components designed in this work (b, c).
Orange = designed in this work; light blue = model; gray = data and the rest. Schematic: records are synthetic.
Writes paper/ESWA_draft/figures/v18/Fig1_overview.{png,pdf}."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "ESWA_draft", "figures", "v18"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7})
INK, MUTED, FAINT, GRID = "#1b1b1b", "#55534f", "#8f8d88", "#d9d7d2"
OR, OR_F, OR_L = "#e8622c", "#fde6da", "#f4a582"
BL, BL_F = "#2a6fdb", "#e2ecfb"
GR_F = "#f4f3f0"

fig = plt.figure(figsize=(7.2, 5.25))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 73); ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=0.8, r=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))


def arr(x0, y0, x1, y1, c=INK, lw=0.7, ls="-", cs="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, color=c, lw=lw, ls=ls, connectionstyle=cs, shrinkA=0, shrinkB=0))


def t(x, y, s, size=7, c=INK, ha="center", va="center", **k):
    ax.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, **k)


def grid(x, y, M, cw=0.55, rh=0.75, removed=None, carry=None):
    """M: rows x cols of 0/1 (recorded); removed: same shape, 1 = removed by re-observation; carry: carried-forward cells."""
    R, C = M.shape
    for r in range(R):
        for c in range(C):
            xx, yy = x + c * cw, y - r * rh
            if carry is not None and carry[r, c]:
                ax.add_patch(Rectangle((xx, yy - rh * 0.42), cw, rh * 0.84, fc="#e7e5e0", ec="none"))
            if M[r, c]:
                ax.add_patch(Circle((xx + cw / 2, yy), min(cw, rh) * 0.27, color=INK, lw=0))
            if removed is not None and removed[r, c]:
                ax.add_patch(Circle((xx + cw / 2, yy), min(cw, rh) * 0.27, fc="white", ec=OR, lw=0.7))


rs = np.random.RandomState(4)
VARS = ["HR", "SpO$_2$", "lactate", "creat.", "GCS", "UO"]
C = 20
dev = np.zeros((6, C), int)
dev[0] = 1; dev[1] = (rs.rand(C) < 0.9); dev[2, [1, 7, 13, 18]] = 1; dev[3, [4, 15]] = 1
dev[4] = (rs.rand(C) < 0.55); dev[5] = (rs.rand(C) < 0.75)
tgt = np.zeros((6, C), int)
tgt[0] = 1; tgt[1] = (rs.rand(C) < 0.25); tgt[3, [9]] = 1; tgt[5] = (rs.rand(C) < 0.35)

# ====================================================================== (a) forward path
t(0.6, 71.6, "(a)", 8.5, ha="left", weight="bold")
YC = 62.0
def glyph(x, y, M, w=9.0, h=9.6):
    R_, C_ = M.shape; cw, rh = w / C_, h / R_
    for r in range(R_):
        for c in range(C_):
            if M[r, c]: ax.add_patch(Circle((x + (c + 0.5) * cw, y + h - (r + 0.5) * rh), 0.2, color=INK, lw=0))
G = 14
# target records
t(5.1, 69.4, "Target hospital $h$", 7.2)
rbox(0.6, 56.9, 9.0, 10.4, GR_F, GRID); glyph(0.6, 57.3, tgt[:, :G])
t(5.1, 55.4, r"$\mathbf{X}_h$, no outcome labels", 6.2, MUTED)
# development records
t(5.1, 52.0, "Development", 7.2)
rbox(0.6, 39.6, 9.0, 10.4, GR_F, GRID); glyph(0.6, 40.0, dev[:, :G])
t(5.1, 38.1, r"MIMIC-IV, $(\mathbf{X}, y)$", 6.2, MUTED)
# profile (designed)
t(21.0, 69.4, "Observation profile", 7.6, weight="bold")
rbox(12.6, 56.9, 16.8, 10.4, OR_F, OR, 1.1)
t(21.0, 65.2, r"$\hat{\pi}_h=(A_h,\ \alpha_h,\ \rho_h,\ c_h)$", 7.4)
for k, (sc, lab) in enumerate([("site", r"$A_h$  never recorded"), ("stay", r"$\alpha_h,\rho_h$  who is tested, streams"), ("patch", r"$c_h$  cadence within a stay")]):
    yy = 62.2 - k * 2.0
    t(13.4, yy, sc, 6.2, MUTED, ha="left"); t(16.6, yy, lab, 6.3, ha="left")
t(21.0, 55.4, "from unlabeled records; see (b)", 6.2, MUTED)
t(21.0, 52.8, "damage analysis ranks the components\none at a time (Section 6.2)", 5.9, FAINT)
arr(9.6, YC, 12.6, YC)
# re-observation (designed)
t(41.5, 69.4, "Re-observation", 7.6, weight="bold")
rbox(32.4, 44.6, 18.2, 22.7, OR_F, OR, 1.4)
t(41.5, 64.6, r"$\tilde{\mathbf{X}} = T_{\hat{\pi}_h}(\mathbf{X})$", 8.2)
steps = [("stay", "tests kept per window"), ("patch", "tests thinned by 3-h block"), ("stay", "streams kept per window"),
         ("patch", "charting, vitals thinned by hour"), ("site", "absent variables removed")]
for k, (sc, lab) in enumerate(steps):
    yy = 61.0 - k * 2.3
    t(33.2, yy, sc, 5.9, OR, ha="left"); t(36.4, yy, lab, 6.2, ha="left")
ax.plot([33.2, 49.8], [49.6, 49.6], color=OR_L, lw=0.6)
t(41.5, 47.3, "carry-forward, differences, masks and\navailability recomputed: a valid record", 5.9, MUTED)
t(41.5, 43.1, "see (c)", 6.2, MUTED)
arr(29.4, YC, 32.4, YC)
arr(9.6, 46.4, 32.4, 46.4)
# factual view bypasses re-observation
ax.plot([9.6, 56.2], [41.4, 41.4], color=FAINT, lw=0.6); arr(56.2, 41.4, 56.2, 56.9, c=FAINT, lw=0.6)
t(31.0, 40.2, r"factual $\mathbf{X}$", 6.0, FAINT)
t(52.9, 64.6, r"$\tilde{\mathbf{X}}$", 6.6, OR, ha="left")
# model (light blue)
t(67.2, 69.4, "Model (here the PatchFusion encoder)", 7.2)
rbox(55.2, 56.9, 24.0, 10.4, BL_F, BL, 0.9)
for k in range(8):
    rbox(56.2 + k * 1.5, 61.6, 1.25, 2.8, "white", BL, 0.6, 0.3)
t(62.1, 65.5, "BiLSTM per 6-h patch", 6.0); t(62.1, 59.8, "16 tokens", 5.8, MUTED)
rbox(69.6, 60.0, 4.2, 5.6, "white", BL, 0.6, 0.4); t(71.7, 62.8, "attn\n×2", 6.0)
rbox(75.0, 60.0, 3.4, 5.6, "white", BL, 0.6, 0.4); t(76.7, 62.8, "CR\nhead", 6.0)
arr(68.2, 62.8, 69.6, 62.8, lw=0.6); arr(73.8, 62.8, 75.0, 62.8, lw=0.6)
t(57.6, 55.4, "trained on both views with labels $y$; any model", 6.2, MUTED, ha="left")
arr(50.6, YC, 55.2, YC)
# outputs
t(89.6, 69.4, "Risk at hospital $h$", 7.2)
for k, lab in enumerate([r"$F_{24\,\mathrm{h}}$", r"$F_{48\,\mathrm{h}}$", r"$F_{7\,\mathrm{d}}$", r"$F_{\mathrm{disch}}$"]):
    t(83.4 + k * 4.1, YC + 0.4, lab, 7.2)
t(89.6, 58.6, "death before each horizon;\ndischarge alive competing", 6.0, MUTED)
arr(79.2, YC, 81.4, YC)
ax.plot([12.6, 50.6], [71.3, 71.3], color=OR, lw=0.7, ls=(0, (3, 2)))
t(31.6, 72.5, "no outcome label of the target hospital is used", 6.2, OR)

# ====================================================================== (b) profile zoom
t(0.6, 35.6, "(b)", 8.5, ha="left", weight="bold"); t(4.0, 35.6, "Observation profile: three scales from unlabeled records", 7.4, ha="left")
ax.add_patch(Rectangle((5.0, 31.9), 1.6, 1.0, fc="#cfcdc7", ec="none")); t(7.0, 32.4, "development hospital", 6.0, MUTED, ha="left")
ax.add_patch(Rectangle((19.0, 31.9), 1.6, 1.0, fc=OR, ec="none")); t(21.0, 32.4, "target hospital $h$", 6.0, MUTED, ha="left")
def mini(x0, y0, w, h):
    a = fig.add_axes([x0 / 100, y0 / 73, w / 100, h / 73])
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color(FAINT); a.spines[sp].set_linewidth(0.6)
    a.tick_params(labelsize=5.6, colors=MUTED, length=1.8, width=0.5, pad=1)
    return a
# site: share of windows with each variable, development vs target
a1 = mini(5.0, 7.0, 13.4, 21.0)
names = ["HR", "SpO$_2$", "creat.", "lactate", "GCS", "UO"]
dv = [1.0, 0.99, 0.86, 0.62, 0.93, 0.95]; tg = [1.0, 0.98, 0.40, 0.0, 0.0, 0.0]
yy = np.arange(len(names))[::-1]
a1.barh(yy + 0.18, dv, height=0.34, color="#cfcdc7", label="development")
a1.barh(yy - 0.18, tg, height=0.34, color=OR, label="target")
for k, v in enumerate(tg):
    if v == 0: a1.text(0.03, yy[k] - 0.18, "×  never recorded", fontsize=5.4, color=OR, va="center")
a1.set_yticks(yy); a1.set_yticklabels(names); a1.set_xlim(0, 1.05); a1.set_xticks([0, 0.5, 1])
a1.set_xlabel("share of windows", fontsize=5.8, color=MUTED, labelpad=1)
a1.set_title("site: $A_h$", fontsize=6.6, color=INK, loc="left", pad=2)

# stay: selection model
a2 = mini(23.6, 7.0, 12.4, 21.0)
aw = np.linspace(-2.5, 2.5, 100); s = lambda z: 1 / (1 + np.exp(-z))
a2.plot(aw, s(0.9 * aw + 2.0), color="#a9a7a1", lw=1.2, label="development")
a2.plot(aw, s(0.9 * aw - 0.4), color=OR, lw=1.4, label="target")
a2.annotate("", xy=(0.0, s(-0.4)), xytext=(0.0, s(2.0)), arrowprops=dict(arrowstyle="-|>", color=OR, lw=0.6, mutation_scale=5))
a2.text(0.15, 0.62, r"$\alpha_h$ shift", fontsize=5.6, color=OR)
a2.set_xlabel("window acuity $a_w$", fontsize=5.8, color=MUTED, labelpad=1); a2.set_ylabel("P(tested)", fontsize=5.8, color=MUTED, labelpad=1)
a2.set_ylim(0, 1.02); a2.set_xticks([]); a2.set_yticks([0, 0.5, 1])
a2.set_title(r"stay: $\sigma(\beta_v a_w+\alpha_{hv})$, $\rho_{hv}$", fontsize=6.6, color=INK, loc="left", pad=2)

# patch: cadence among tested windows (16 three-hour blocks)
t(40.8, 29.6, "patch: $c_{hv}$", 6.6, INK, ha="left")
for row, (lab, p) in enumerate([("development", 0.65), ("target", 0.25)]):
    rr = np.random.RandomState(10 + row).rand(16) < p
    y0 = 25.5 - row * 6.0
    t(40.8, y0 + 2.0, lab, 5.8, MUTED, ha="left")
    for k in range(16):
        ax.add_patch(Rectangle((40.8 + k * 0.42, y0 - 0.9), 0.36, 1.8, fc=(OR if row else "#a9a7a1") if rr[k] else "white", ec=GRID, lw=0.35))
t(44.2, 11.2, "3-h blocks with a test,\namong tested windows", 5.8, MUTED)
t(44.2, 7.6, "also: charting and vital-sign\ncadence by hour", 5.8, MUTED)

# ====================================================================== (c) re-observation zoom
t(51.0, 35.6, "(c)", 8.5, ha="left", weight="bold"); t(54.4, 35.6, "Re-observation of one development record (20 h shown)", 7.4, ha="left")
CW, RH = 0.78, 2.75
X0L, X0R, Y0 = 56.6, 76.2, 29.6
for k, v in enumerate(VARS):
    t(X0L - 0.6, Y0 - k * RH, v, 6.0, MUTED, ha="right")
t(X0L + C * CW / 2, 32.6, r"development $\mathbf{X}$", 6.6)
t(X0R + C * CW / 2, 32.6, r"re-observed $\tilde{\mathbf{X}}$", 6.6)
carry_dev = np.zeros_like(dev)
for r in (2, 3):
    last = -99
    for c in range(C):
        if dev[r, c]: last = c
        elif c - last < 4: carry_dev[r, c] = 1
grid(X0L, Y0, dev, cw=CW, rh=RH, carry=carry_dev)
rem = np.zeros_like(dev)
rem[2] = dev[2]                                        # stay: lactate not tested in this window at the target's acuity-dependent rate
rem[4] = dev[4]                                        # site: GCS never recorded at the target
rem[1] = dev[1] * (np.random.RandomState(8).rand(C) < 0.7)   # patch: SpO2 thinned to target cadence
rem[5] = dev[5] * (np.random.RandomState(9).rand(C) < 0.5)   # patch: urine output thinned by hour
kept = dev * (1 - rem)
carry_re = np.zeros_like(dev)
for r in (3,):
    last = -99
    for c in range(C):
        if kept[r, c]: last = c
        elif c - last < 4: carry_re[r, c] = 1
grid(X0R, Y0, kept, cw=CW, rh=RH, removed=rem, carry=carry_re)
arr(X0L + C * CW + 0.6, 32.6, X0R - 0.6, 32.6, c=OR, lw=0.9)
t((X0L + C * CW + X0R) / 2, 34.0, r"$T_{\hat{\pi}_h}$", 6.8, OR)
for k, lab in ((1, "patch: cadence"), (2, "stay: not tested"), (4, "site: never recorded"), (5, "patch: cadence")):
    t(X0R + C * CW + 0.5, Y0 - k * RH, lab, 5.6, OR, ha="left")
t(X0L, 11.0, "", 5)
# legend for (c)
yL = 10.2
ax.add_patch(Circle((57.0, yL), 0.36, color=INK, lw=0)); t(57.8, yL, "recorded", 5.9, MUTED, ha="left")
ax.add_patch(Circle((63.6, yL), 0.36, fc="white", ec=OR, lw=0.7)); t(64.4, yL, "removed by re-observation", 5.9, MUTED, ha="left")
ax.add_patch(Rectangle((78.4, yL - 0.5), 1.0, 1.0, fc="#e7e5e0", ec="none")); t(79.8, yL, "carried-forward value", 5.9, MUTED, ha="left")
t(56.6, 6.8, "Dropping the lactate tests also drops their carried-forward hours, so the record stays a valid hourly record.", 5.9, MUTED, ha="left")
# footer
ax.add_patch(Rectangle((0.8, 1.2), 2.0, 1.4, fc=OR_F, ec=OR, lw=0.8)); t(3.4, 1.9, "designed in this work", 6.0, MUTED, ha="left")
ax.add_patch(Rectangle((16.0, 1.2), 2.0, 1.4, fc=BL_F, ec=BL, lw=0.8)); t(18.6, 1.9, "model trained by the procedure", 6.0, MUTED, ha="left")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig1_overview.{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)
print("written", OUT)
