"""Figure 1 (v19): visual-first overview of profile-guided re-observation. Records are drawn as data (curves, observation
dots, carry-forward steps), the profile as three small charts, the model as a token map, the output as risk curves, and
panel (b) compares observation maps. Orange = this work; blue = model; gray/black = data. Schematic, synthetic records.
Writes paper/ESWA_draft/figures/v19/Fig1_overview.{png,pdf}."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "ESWA_draft", "figures", "v19"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5})
INK, MUTED, FAINT, GRID = "#1b1b1b", "#5c5b57", "#a3a19c", "#e4e2dd"
OR, OR_F, BL, BL_F = "#e8622c", "#fde6da", "#2a6fdb", "#e2ecfb"
H = 48; hh = np.arange(H)
rs = np.random.RandomState(2)

# ---------------------------------------------------------------- synthetic patient (development) and observation patterns
hr = 92 + 9 * np.sin(hh / 7.0) + np.cumsum(rs.randn(H)) * 0.8
sp = 95 - 2.5 * np.sin(hh / 9.0 + 1) + rs.randn(H) * 0.6
lac_t = np.array([3, 11, 19, 27, 35, 43]); lac_v = np.array([2.1, 2.6, 3.4, 2.9, 2.4, 2.0])
obs_dev = {"hr": np.ones(H, bool), "sp": rs.rand(H) < 0.92, "gcs": rs.rand(H) < 0.5, "uo": rs.rand(H) < 0.8}
# what the target hospital records (another patient's pattern) and the re-observation of the development patient
obs_tgt = {"hr": np.ones(H, bool), "sp": rs.rand(H) < 0.12, "gcs": np.zeros(H, bool), "uo": rs.rand(H) < 0.3}
keep = {"hr": np.ones(H, bool), "sp": obs_dev["sp"] & (rs.rand(H) < 0.14), "gcs": np.zeros(H, bool), "uo": obs_dev["uo"] & (rs.rand(H) < 0.38)}
lac_keep = np.array([False, False, True, False, False, False])          # one lactate test kept in this window


def style_trace(a):
    for sp_ in a.spines.values(): sp_.set_visible(False)
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(-1, H)


def record(a, obs, lac_mask, title, removed=None, lac_removed=None, title_c=INK):
    """four stacked traces: HR (line + dots), SpO2 (line + dots), lactate (dots + carried step), GCS dots"""
    style_trace(a); a.set_ylim(-0.6, 4.3)
    rows = {"hr": 3.4, "sp": 2.3, "lac": 1.2, "gcs": 0.2}
    for key, y0, sig in (("hr", rows["hr"], hr), ("sp", rows["sp"], sp)):
        z = (sig - sig.mean()) / (sig.std() * 3.2) + y0
        a.plot(hh, z, color=GRID, lw=0.9, zorder=1)
        o = obs[key]; a.scatter(hh[o], z[o], s=3.2, color=INK, lw=0, zorder=3)
        if removed is not None:
            r_ = removed[key]; a.scatter(hh[r_], z[r_], s=4.5, facecolor="white", edgecolor=OR, lw=0.45, zorder=3)
    # lactate: values carried forward for 6 h
    y0 = rows["lac"]; lz = (lac_v - 2.5) / 3.0 + y0
    for k, (t0, v) in enumerate(zip(lac_t, lz)):
        if lac_mask[k]:
            a.plot([t0, min(t0 + 6, H - 1)], [v, v], color="#bdbbb5", lw=2.2, solid_capstyle="butt", zorder=2)
            a.scatter([t0], [v], s=7, color=INK, lw=0, zorder=3)
        elif lac_removed is not None and lac_removed[k]:
            a.scatter([t0], [v], s=7, facecolor="white", edgecolor=OR, lw=0.5, zorder=3)
    y0 = rows["gcs"]; o = obs["gcs"]
    a.scatter(hh[o], np.full(o.sum(), y0), s=3.0, color=INK, lw=0, marker="s", zorder=3)
    if removed is not None:
        r_ = removed["gcs"]; a.scatter(hh[r_], np.full(r_.sum(), y0), s=4.0, facecolor="white", edgecolor=OR, lw=0.45, marker="s", zorder=3)
    a.set_title(title, fontsize=6.8, color=title_c, loc="left", pad=1.5)


def fa(x0, y0, x1, y1, c=INK, lw=0.8, cs="arc3"):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, color=c, lw=lw, connectionstyle=cs))


fig = plt.figure(figsize=(7.2, 4.7))
fig.text(0.005, 0.975, "(a)", fontsize=8.5, weight="bold", va="top")
# variable labels for the record panels
LAB = [("HR", 3.4), ("SpO$_2$", 2.3), ("lactate", 1.2), ("GCS", 0.2)]
# target record (top left) and development record (bottom left)
aT = fig.add_axes([0.065, 0.765, 0.205, 0.17]); record(aT, obs_tgt, np.array([False, False, True, False, False, False]), "target hospital $h$ · no labels")
aD = fig.add_axes([0.065, 0.53, 0.205, 0.17]); record(aD, obs_dev, np.ones(6, bool), "development patient · labeled")
for a in (aT, aD):
    for lab, y in LAB: a.text(-2.5, y, lab, fontsize=5.6, color=MUTED, ha="right", va="center")
# profile: three small charts read from the target hospital's unlabeled records
px, pw = 0.335, 0.075
fig.text(px + 0.115, 0.958, r"profile $\hat{\pi}_h$", fontsize=7.2, color=OR, ha="center", weight="bold")
a1 = fig.add_axes([px, 0.80, pw, 0.12])
a1.barh([3, 2, 1, 0], [1, 0.15, 0.55, 0], color=[INK, OR, OR, OR], height=0.6); a1.barh([3, 2, 1, 0], [1, 0.92, 0.9, 0.5], color="none", edgecolor=FAINT, height=0.6, lw=0.4)
a1.text(0.06, 0, "×", color=OR, fontsize=6.5, va="center")
a1.set_xlim(0, 1.05); a1.set_title("site", fontsize=6, color=MUTED, pad=1); a1.axis("off")
a2 = fig.add_axes([px + pw + 0.012, 0.80, pw, 0.12])
z = np.linspace(-3, 3, 60); s = lambda q: 1 / (1 + np.exp(-q))
a2.plot(z, s(z + 2), color=FAINT, lw=1.0); a2.plot(z, s(z - 0.6), color=OR, lw=1.3)
a2.set_title("stay", fontsize=6, color=MUTED, pad=1); a2.axis("off")
a3 = fig.add_axes([px + 2 * (pw + 0.012), 0.80, pw, 0.12]); a3.axis("off"); a3.set_xlim(0, 16); a3.set_ylim(0, 2)
for k in range(16):
    a3.add_patch(plt.Rectangle((k, 1.15), 0.8, 0.7, color=FAINT if np.random.RandomState(k).rand() < 0.7 else "white", ec=GRID, lw=0.3))
    a3.add_patch(plt.Rectangle((k, 0.2), 0.8, 0.7, color=OR if np.random.RandomState(50 + k).rand() < 0.25 else "white", ec=GRID, lw=0.3))
a3.set_title("patch", fontsize=6, color=MUTED, pad=1)
fa(0.272, 0.85, px - 0.006, 0.85, c=MUTED)
# re-observation operator
fig.add_artist(FancyBboxPatch((0.405, 0.575), 0.13, 0.12, boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure, fc=OR_F, ec=OR, lw=1.3))
fig.text(0.47, 0.648, r"$T_{\hat{\pi}_h}$", fontsize=11, color=OR, ha="center", va="center")
fig.text(0.47, 0.598, "re-observe", fontsize=6.4, color=OR, ha="center", va="center", weight="bold")
fa(0.47, 0.755, 0.47, 0.697, c=OR, lw=1.0)
fa(0.272, 0.61, 0.405, 0.61, c=INK)
# re-observed development record
aR = fig.add_axes([0.585, 0.53, 0.205, 0.17])
removed = {k: obs_dev[k] & ~keep[k] for k in obs_dev}
record(aR, keep, lac_keep, r"re-observed $\tilde{\mathbf{X}}$ · labels kept", removed=removed, lac_removed=~lac_keep, title_c=OR)
fa(0.535, 0.61, 0.582, 0.61, c=OR, lw=1.0)
fig.text(0.587, 0.505, "dropped as hospital $h$ would not record them;\ncarry-forward recomputed", fontsize=5.6, color=MUTED, va="top")
# model: token map (factual + re-observed) and competing-risk output
aM = fig.add_axes([0.825, 0.565, 0.06, 0.14])
tok = np.random.RandomState(5).rand(16, 10)
aM.imshow(tok, cmap=LinearSegmentedColormap.from_list("b", ["#cfe0fa", BL, "#163f86"]), aspect="auto")
aM.set_xticks([]); aM.set_yticks([]); [sp_.set_color(BL) for sp_ in aM.spines.values()]
aM.set_title("model", fontsize=6.4, color=BL, pad=1.5)
fig.text(0.855, 0.548, "16 patch tokens", fontsize=5.4, color=MUTED, ha="center", va="top")
fa(0.792, 0.635, 0.823, 0.635, c=INK)
aO = fig.add_axes([0.915, 0.565, 0.075, 0.14])
tt = np.linspace(0, 7, 80)
for k, (lam, al) in enumerate(((0.35, 1.0), (0.18, 0.75), (0.08, 0.5))):
    aO.plot(tt, 1 - np.exp(-lam * tt), color=BL, lw=1.0, alpha=al)
for sp_ in ("top", "right"): aO.spines[sp_].set_visible(False)
aO.spines["left"].set_color(FAINT); aO.spines["bottom"].set_color(FAINT); aO.set_xticks([]); aO.set_yticks([])
aO.set_title("risk", fontsize=6.4, color=INK, pad=1.5); aO.set_xlabel("days", fontsize=5.4, color=MUTED, labelpad=1)
fa(0.887, 0.635, 0.913, 0.635, c=INK)
# factual view also enters the model
fa(0.168, 0.53, 0.855, 0.565, c=FAINT, lw=0.6, cs="arc3,rad=0.26")
fig.text(0.33, 0.452, r"factual $\mathbf{X}$ with labels $y$", fontsize=5.6, color=FAINT, ha="center")
fig.text(px + 0.115, 0.775, "unlabeled records only", fontsize=5.8, color=OR, ha="center", va="top")

# ---------------------------------------------------------------- (b) observation maps
fig.text(0.005, 0.405, "(b)", fontsize=8.5, weight="bold", va="top")
VARS = ["HR", "SpO$_2$", "lactate", "creat.", "GCS", "UO"]
r2 = np.random.RandomState(11)
dev = np.zeros((6, H), int)
dev[0] = 1; dev[1] = r2.rand(H) < 0.92; dev[2, [3, 11, 19, 27, 35, 43]] = 1; dev[3, [6, 30]] = 1; dev[4] = r2.rand(H) < 0.5; dev[5] = r2.rand(H) < 0.8
def carried(m, rows=(2, 3), pers=6):
    c = np.zeros_like(m)
    for r in rows:
        last = -99
        for j in range(H):
            if m[r, j]: last = j
            elif j - last <= pers: c[r, j] = 1
    return c
tgt = np.zeros((6, H), int); tgt[0] = 1; tgt[1] = r2.rand(H) < 0.12; tgt[3, [20]] = 1; tgt[5] = r2.rand(H) < 0.3
reo = dev.copy(); rem = np.zeros_like(dev)
for r, p in ((1, 0.14), (5, 0.38)):
    k_ = r2.rand(H) < p; rem[r] = dev[r] & ~k_; reo[r] = dev[r] & k_
rem[2] = dev[2]; reo[2] = 0; rem[2, 19] = 0; reo[2, 19] = 1          # lactate: one test kept in the window
rem[4] = dev[4]; reo[4] = 0                                            # GCS never recorded
msk = dev.copy(); hole = np.zeros_like(dev); cd = carried(dev)
for r in range(6):
    drop = r2.rand(H) < (0.6 if r in (1, 2, 3, 4, 5) else 0.0)
    if r == 4: drop[:] = True
    hole[r] = (dev[r] | cd[r]) & drop; msk[r] = dev[r] & ~drop
def code(m, removed=None, carry=None, holes=None):
    z = np.zeros_like(m)
    if carry is not None: z[carry.astype(bool)] = 2
    z[m.astype(bool)] = 1
    if removed is not None: z[removed.astype(bool)] = 3
    if holes is not None: z[(holes.astype(bool)) & (z == 0)] = 4
    return z
CM = ListedColormap(["white", INK, "#cfcdc7", OR, "#f3c3ad"])
maps = [("development patient", code(dev, carry=cd)), ("target hospital $h$", code(tgt, carry=carried(tgt))),
        ("re-observed (this work)", code(reo, removed=rem, carry=carried(reo))), ("entry-wise masking", code(msk, carry=(cd & ~hole.astype(bool)).astype(int), holes=hole))]
W0, Wm, gap = 0.065, 0.155, 0.016
for k, (title, z) in enumerate(maps):
    a = fig.add_axes([W0 + k * (Wm + gap), 0.10, Wm, 0.235])
    a.imshow(z, cmap=CM, vmin=0, vmax=4, aspect="auto", interpolation="nearest")
    a.set_xticks([0, 23, 47]); a.set_xticklabels(["0", "24", "48 h"], fontsize=5.4, color=MUTED); a.tick_params(length=1.5, pad=1)
    a.set_yticks(range(6)); a.set_yticklabels(VARS if k == 0 else [], fontsize=5.6, color=MUTED)
    for sp_ in a.spines.values(): sp_.set_color(OR if k == 2 else GRID); sp_.set_linewidth(1.1 if k == 2 else 0.5)
    a.set_title(title, fontsize=6.6, color=OR if k == 2 else INK, pad=2)
# result inset: Korean hospitals
aB = fig.add_axes([0.855, 0.13, 0.12, 0.19])
aB.barh([1, 0], [0.045, 0.016], color=[OR, "#c9c7c1"], height=0.55)
aB.errorbar([0.045, 0.016], [1, 0], xerr=[[0.024, 0.027], [0.024, 0.026]], fmt="none", ecolor=MUTED, elinewidth=0.6, capsize=1.5)
aB.set_yticks([1, 0]); aB.set_yticklabels(["re-observed", "masking"], fontsize=5.8, color=INK)
aB.axvline(0, color=MUTED, lw=0.5); aB.set_xlim(-0.015, 0.075); aB.set_xticks([0, 0.04]); aB.tick_params(labelsize=5.4, colors=MUTED, length=1.5)
for sp_ in ("top", "right", "left"): aB.spines[sp_].set_visible(False)
aB.spines["bottom"].set_color(FAINT)
aB.set_title("Korean hospitals:\nΔ 24-h AUROC", fontsize=6.2, color=INK, pad=2)
# legend for (b)
lx, ly = 0.065, 0.03
for k, (c, lab) in enumerate([(INK, "recorded"), ("#cfcdc7", "carried forward"), (OR, "removed by re-observation"), ("#f3c3ad", "hole left by masking")]):
    fig.add_artist(plt.Rectangle((lx + k * 0.17, ly), 0.012, 0.022, transform=fig.transFigure, color=c, ec=GRID, lw=0.3))
    fig.text(lx + k * 0.17 + 0.017, ly + 0.011, lab, fontsize=5.8, color=MUTED, va="center")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig1_overview.{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)
print("written", OUT)
