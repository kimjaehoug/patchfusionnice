"""Figures 3-7 of the ESWA submission, drawn from the values of the manuscript tables they replace or summarise.
Fig. 3 observation profiles (old Table 2), Fig. 4 component damage (old Table 3), Fig. 5 gain over the backbone
(summary of Table 2, old Table 4), Fig. 6 patch-scale steps (old Tables 5 and 6), Fig. 7 source-target trade-off
(old Table 7). Style follows the calibration and decision-curve figures (DejaVu Serif; method colours of
utility_figures.py); targets use the source colours of Fig. 2 (eICU-CRD blue, Hallym orange).
Writes PDF (vector), 1200-dpi PNG and a 600-dpi PNG for embedding into OUT."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "paper", "ESWA_draft", "figures", "tablefigs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42,
                     "svg.fonttype": "none"})
INK, MUTED, FAINT, GRID, BAND = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df", "#f6f5f2"
EICU, HAL = "#2a6fdb", "#e8622c"                     # source colours of Fig. 2
JOMS = bool(os.environ.get("PF_JOMS"))               # journal lettering: Arial-metric sans, contrast >= 4.5:1, no transparency
HAL_T, EICU_T = ("#a33a0b", "#1f5fc4") if JOMS else (HAL, EICU)
if JOMS:
    from matplotlib import font_manager
    FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "JOMS", "fonts")
    for f_ in ("Regular", "Bold", "Italic", "BoldItalic"):
        font_manager.fontManager.addfont(os.path.join(FD, f"LiberationSans-{f_}.ttf"))
    plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold", "ps.fonttype": 42})
M = {"backbone": ("Backbone (source-only)", "#008300"), "reweight": ("Importance reweighting", "#8a8984"),
     "generic": ("Generic counterfactual training", "#e87ba4"), "selection": ("Selection-only imagination", "#eda100"),
     "rate": ("Rate-level imagination", "#1baf7a"), "nopatch": ("Proposed without patch scale", "#8db8ee"),
     "proposed": ("Proposed (three-scale profile)", "#2a78d6")}


def style(ax, grid="x"):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=7.2, width=0.6, length=2.5)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def title(ax, s, x=0.0):
    ax.set_title(s, loc="left", color=INK, x=x, pad=6)


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 1200})) + ((("eps", {}),) if JOMS else ()):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03, **kw)
    fig.savefig(os.path.join(OUT, f"{name}_600dpi.png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# =============================================================== Fig. 3  observation profiles at three scales
def fig3():
    rows = ["MIMIC-IV (development)", "eICU-CRD, low testing", "Hallym, Kangnam", "Hallym, Chuncheon", "Hallym, Hallym",
            "Hallym, Dongtan"]
    rcol = [INK, EICU_T, HAL_T, HAL_T, HAL_T, HAL_T]
    Y = np.array([0, 1.3, 2.6, 3.6, 4.6, 5.6])           # gaps between sources
    nan = np.nan
    share = np.array([[.96, .97, .96, 1.00, .97], [.85, .85, .81, .49, .55], [.93, .93, .93, nan, nan],
                      [.32, .33, .99, nan, nan], [.01, .01, .99, nan, nan], [.01, .03, .99, nan, nan]])
    cad = np.array([[.29, .33, .28, .34, .60, .98], [.21, .23, .19, .13, .15, .94], [.20, .22, .21, nan, nan, .10],
                    [.10, .11, .23, nan, nan, .17], [.10, .11, .25, nan, nan, .12], [.08, .23, .24, nan, nan, .10]])
    absent = [0, 0, 7, 8, 7, 7]
    cmap = LinearSegmentedColormap.from_list("prof", ["#fffaf6", "#f9c9ad", "#e8622c", "#9c3510"])
    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.7, 5, 6], wspace=0.07, left=0.165, right=0.995, top=0.78, bottom=0.04)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
    for ax in (a, b, c):
        ax.set_ylim(Y[-1] + 0.55, -0.55); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    # (a) site: absent variables
    a.barh(Y, absent, height=0.72, color=HAL, alpha=1.0 if JOMS else 0.9)
    for y, v in zip(Y, absent):
        a.text(v + 0.4, y, f"{v}", va="center", ha="left", fontsize=7.2, color=INK if v else MUTED)
    a.set_xlim(0, 11); a.set_xticks([]); a.axvline(0, color=MUTED, lw=0.7)
    for y, r, col in zip(Y, rows, rcol):
        a.text(-0.6, y, r, ha="right", va="center", fontsize=7.4, color=col)
    a.text(0, -0.95, "of 22 variables", fontsize=6.6, color=MUTED, ha="left", va="center")
    title(a, "(a) Site\nabsent variables")

    def heat(ax, M_, cols, groups, ttl):
        for i, y in enumerate(Y):
            for j in range(M_.shape[1]):
                v = M_[i, j]
                if np.isnan(v):
                    ax.add_patch(Rectangle((j - 0.47, y - 0.4), 0.94, 0.8, fc="#f1f0ec", ec="#d6d4cf", lw=0.4, hatch="//////"))
                    ax.text(j, y, "absent", ha="center", va="center", fontsize=6.4, color=MUTED)
                else:
                    ax.add_patch(Rectangle((j - 0.47, y - 0.4), 0.94, 0.8, fc=cmap(v), ec="white", lw=0.6))
                    ax.text(j, y, f"{v:.2f}", ha="center", va="center", fontsize=7.2, color="white" if v > 0.55 else INK)
        ax.set_xlim(-0.55, M_.shape[1] - 0.45)
        ax.set_xticks([])
        for j, cname in enumerate(cols):
            ax.text(j, -0.75, cname, ha="center", va="bottom", fontsize=7.0, color=INK)
        for g0, g1, gname in groups:
            ax.plot([g0 - 0.42, g1 + 0.42], [-1.55, -1.55], color=MUTED, lw=0.6, clip_on=False)
            ax.text((g0 + g1) / 2, -1.7, gname, ha="center", va="bottom", fontsize=6.5, color=MUTED)
        title(ax, ttl)
    heat(b, share, ["BUN", "Sodium", "WBC", "GCS", "Urine\noutput"], [(0, 2, "laboratory"), (3, 4, "charting streams")],
         "(b) Stay\nshare of windows with the variable")
    heat(c, cad, ["BUN", "Sodium", "WBC", "GCS", "Urine\noutput", "SpO$_2$"],
         [(0, 2, "laboratory, 3-h blocks"), (3, 5, "charting and vitals, hours")],
         "(c) Patch\ncadence within those windows")
    for ax in (a, b, c):                       # titles above the column headers
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=36, color=INK, linespacing=1.35)
    save(fig, "Fig3")


# =============================================================== Fig. 4  component damage
def fig4():
    R = [("site", "Absent variables", "H", (-.131, -.157, -.091), (-.179, -.225, -.129), (-.109, -.122, -.093)),
         ("stay", "Charting streams\nremoved per stay", "E", (-.046, -.057, -.031), (-.057, -.093, -.036), (-.043, -.048, -.037)),
         ("stay", "Laboratory selection", "H", (-.014, -.015, -.013), (-.084, -.093, -.071), (-.016, -.018, -.015)),
         ("stay", "Laboratory selection", "E", (-.005, -.007, -.003), (-.033, -.042, -.028), (-.010, -.011, -.009)),
         ("patch", "Charting cadence\n(GCS, urine output)", "E", (-.030, -.040, -.018), (-.058, -.097, -.035), (-.028, -.034, -.023)),
         ("patch", "Laboratory cadence", "H", (-.004, -.006, -.001), (-.050, -.054, -.044), (-.006, -.007, -.005)),
         ("patch", "Laboratory cadence", "E", (-.001, -.003, .001), (-.020, -.022, -.017), (-.004, -.005, -.003)),
         ("patch", "Vital-sign cadence", "H", (-.002, -.006, .000), (-.026, -.047, -.015), (-.003, -.004, -.000))]
    Y = np.array([0, 1.5, 2.5, 3.5, 5.0, 6.0, 7.0, 8.0])
    fig = plt.figure(figsize=(7.15, 3.0))
    gs = fig.add_gridspec(1, 3, wspace=0.13, left=0.315, right=0.99, top=0.86, bottom=0.15)
    lims = [(-0.225, 0.012), (-0.31, 0.012), (-0.175, 0.012)]
    titles = ["(a) Δ 24-h AUROC", "(b) Δ 24-h AUPRC", "(c) Δ in-hospital AUROC"]
    axes = [fig.add_subplot(gs[k]) for k in range(3)]
    for k, ax in enumerate(axes):
        style(ax)
        ax.set_ylim(Y[-1] + 0.6, -0.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        for (y0, y1) in ((-0.6, 0.6), (4.4, 8.6)):
            ax.axhspan(y0, y1, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        for y, r in zip(Y, R):
            m, lo, hi = r[3 + k]; col = HAL if r[2] == "H" else EICU
            ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot(m, y, marker="s" if r[2] == "H" else "o", ms=4.6, color=col, mec="white", mew=0.6, zorder=3)
            ax.text(lo - 0.035 * (lims[k][1] - lims[k][0]), y, f"{m:+.3f}".replace("-", "−"), va="center",
                    ha="right", fontsize=6.6, color=INK)
        ax.set_xlim(*lims[k])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace("-", "−")))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05 if k == 2 else 0.1))
        title(ax, titles[k])
    a = axes[0]
    tr = matplotlib.transforms.blended_transform_factory(fig.transFigure, a.transData)
    for y, r in zip(Y, R):
        a.text(0.228, y, r[1], ha="right", va="center", fontsize=7.2, color=INK, transform=tr)
        a.text(0.234, y, "Hallym" if r[2] == "H" else "eICU-CRD", ha="left", va="center", fontsize=6.6,
               color=HAL_T if r[2] == "H" else EICU_T, transform=tr)
    # scale brackets
    for (y0, y1, s) in ((0, 0, "Site"), (1.5, 3.5, "Stay"), (5.0, 8.0, "Patch")):
        xb = 0.055; a = axes[0]
        a.plot([xb, xb], [y0 - 0.32, y1 + 0.32], color=MUTED, lw=1.0, clip_on=False, transform=tr)
        a.text(xb - 0.006, (y0 + y1) / 2, s, ha="right", va="center", fontsize=7.4, color=INK, transform=tr)
    h = [Line2D([], [], color=EICU, marker="o", ms=4.6, lw=1.4, label="component set to the low-testing eICU-CRD value"),
         Line2D([], [], color=HAL, marker="s", ms=4.6, lw=1.4, label="component set to the Hallym value")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7.2, bbox_to_anchor=(0.62, -0.01))
    save(fig, "Fig4")


# =============================================================== Fig. 5  gain over the backbone (Table 2)
def fig5():
    cats = ["24 h", "In-hospital", "24 h", "48 h", "7 days", "In-hospital"]
    T = {"backbone": [(.914, .318), (.848, .457), (.859, .095), (.826, .130), (.824, .233), (.720, .446)],
         "reweight": [(.884, .221), (.826, .396), (.848, .076), (.825, .106), (.818, .241), (.699, .425)],
         "generic": [(.921, .330), (.856, .465), (.867, .128), (.841, .170), (.829, .256), (.716, .445)],
         "rate": [(.927, .340), (.861, .471), (.898, .198), (.868, .250), (.834, .289), (.723, .449)],
         "nopatch": [(.921, .329), (.858, .464), (.876, .149), (.845, .195), (.824, .268), (.704, .426)],
         "proposed": [(.929, .346), (.862, .475), (.890, .172), (.855, .209), (.823, .271), (.703, .427)]}
    order = ["reweight", "generic", "rate", "nopatch", "proposed"]
    X = np.array([0, 1, 2.6, 3.6, 4.6, 5.6])
    off = np.linspace(-0.3, 0.3, len(order))
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.3), sharex=True, gridspec_kw={"hspace": 0.28, "top": 0.855,
                                                                                    "bottom": 0.10, "left": 0.09, "right": 0.99})
    for k, (ax, lab) in enumerate(zip(axes, ["(a) Change in AUROC over the backbone", "(b) Change in AUPRC over the backbone"])):
        style(ax, grid="y")
        ax.axvspan(-0.5, 1.5, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=M["backbone"][1], lw=1.0, ls=(0, (4, 2)), zorder=1)
        for mi, m in enumerate(order):
            d = np.array([T[m][j][k] - T["backbone"][j][k] for j in range(6)])
            big = m == "proposed"
            for j in range(6):
                ax.plot([X[j] + off[mi]] * 2, [0, d[j]], color=M[m][1], lw=0.9 if big else 0.7, alpha=1.0 if JOMS else 0.55, zorder=2)
            ax.plot(X + off[mi], d, ls="none", marker=dict(reweight="v", generic="s", rate="^", nopatch="D", proposed="o")[m] if JOMS else "o", ms=5.2 if big else 4.2, color=M[m][1], mec="white", mew=0.6,
                    zorder=3, label=M[m][0])
            if big:
                for j in range(6):
                    ax.text(X[j] + off[mi] + 0.07, d[j], f"{d[j]:+.3f}".replace("-", "−"), fontsize=6.3, color=EICU_T if JOMS else M[m][1],
                            va="bottom" if abs(d[j]) < 0.004 * (1 + 3 * k) else "center", ha="left")
        ax.set_ylabel("Δ AUROC" if k == 0 else "Δ AUPRC", color=INK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}".replace("-", "−") if v else "0"))
        title(ax, lab)
        ax.set_xlim(-0.5, 6.1)
    axes[1].set_xticks(X); axes[1].set_xticklabels(cats, color=INK, fontsize=7.4)
    for x0, s, col in ((0.5, "eICU-CRD, all hospitals", EICU_T), (4.1, "Hallym hospitals (independent evaluation)", HAL_T)):
        axes[1].text(x0, -0.17, s, transform=matplotlib.transforms.blended_transform_factory(axes[1].transData, axes[1].transAxes),
                     ha="center", va="top", fontsize=7.6, color=col)
    h, l = axes[0].get_legend_handles_labels()
    h = [Line2D([], [], color=M["backbone"][1], lw=1.0, ls=(0, (4, 2)))] + h; l = ["Backbone (zero line)"] + l
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.2, bbox_to_anchor=(0.54, 1.0))
    save(fig, "Fig5")


# =============================================================== Fig. 6  patch-scale steps (old Tables 5 and 6)
def fig6():
    hz = ["24 h", "48 h", "7 days", "In-hosp."]
    dA = {"H": [(.025, .012, .038), (.018, .005, .033), (.011, .001, .023), (.001, -.008, .012)],
          "E": [(.006, .003, .009), (.007, .004, .010), (.008, .004, .011), (.006, .003, .010)]}
    dP = {"H": [(.035, .019, .054), (.037, .017, .056), (.019, .004, .034), (.006, -.006, .018)],
          "E": [(.010, .002, .018), (.011, .004, .019), (.017, .011, .024), (.014, .007, .021)]}
    oe = {"H": [(.40, .88), (.41, .76), (.49, .66), (1.08, 1.21)], "E": [(.75, .90), (.77, .93), (.76, .92), (.72, .84)]}
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.72], hspace=0.75, wspace=0.36, left=0.075, right=0.99, top=0.86,
                          bottom=0.075)
    X = np.arange(4); dx = {"E": -0.13, "H": 0.13}
    col = {"E": EICU, "H": HAL}; mk = {"E": "o", "H": "s"}
    for k, (D, lab, yl) in enumerate(((dA, "(a) Δ AUROC, with − without", "Δ AUROC"), (dP, "(b) Δ AUPRC, with − without", "Δ AUPRC"))):
        ax = fig.add_subplot(gs[0, k]); style(ax, grid="y")
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        for t in ("E", "H"):
            for j, (m, lo, hi) in enumerate(D[t]):
                ax.plot([X[j] + dx[t]] * 2, [lo, hi], color=col[t], lw=1.4, solid_capstyle="round")
                ax.plot(X[j] + dx[t], m, marker=mk[t], ms=4.6, color=col[t], mec="white", mew=0.6)
        ax.set_xticks(X); ax.set_xticklabels(hz, color=INK); ax.set_xlim(-0.5, 3.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}".replace("-", "−") if v else "0"))
        ax.set_ylabel(yl, color=INK); title(ax, lab)
    ax = fig.add_subplot(gs[0, 2]); style(ax, grid="y")
    ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    for t in ("E", "H"):
        for j, (w0, w1) in enumerate(oe[t]):
            x = X[j] + dx[t]
            ax.annotate("", xy=(x, w1), xytext=(x, w0), arrowprops=dict(arrowstyle="-|>", color=col[t], lw=1.1,
                                                                          mutation_scale=7, shrinkA=2.6, shrinkB=2.6))
            ax.plot(x, w0, marker=mk[t], ms=4.6, mfc="white", mec=col[t], mew=1.0)
            ax.plot(x, w1, marker=mk[t], ms=4.6, color=col[t], mec="white", mew=0.6)
    ax.set_xticks(X); ax.set_xticklabels(hz, color=INK); ax.set_xlim(-0.5, 3.5); ax.set_ylim(0.3, 1.3)
    ax.set_ylabel("O/E", color=INK); title(ax, "(c) O/E, without → with")
    ax.text(-0.42, 1.02, "O/E = 1", fontsize=6.4, color=MUTED, ha="left", va="bottom")
    h = [Line2D([], [], color=EICU, marker="o", ms=4.6, lw=1.4, label="eICU-CRD, low testing (hospital bootstrap)"),
         Line2D([], [], color=HAL, marker="s", ms=4.6, lw=1.4, label="Hallym (patient bootstrap)"),
         Line2D([], [], color=MUTED, marker="o", ms=4.6, mfc="white", lw=0, label="without"),
         Line2D([], [], color=MUTED, marker="o", ms=4.6, lw=0, label="with patch-scale steps")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=7.0, bbox_to_anchor=(0.53, 1.0),
               columnspacing=1.4, handletextpad=0.5)
    # (d) robustness to lowered charting cadence (old Table 6)
    th = [("Backbone", .944, .918, "−0.027", "−0.026", "#008300"),
          ("Generic counterfactual training", .944, .929, "−0.015", "−0.014", "#e87ba4"),
          ("Profile matching, stay scale only", .948, .936, "−0.012", "−0.012", "#8db8ee"),
          ("Profile matching with charting cadence", .949, .937, "−0.012", "−0.012", "#2a78d6")]
    pos = gs[1, :].get_position(fig)
    ax = fig.add_axes([0.335, pos.y0, 0.43, pos.height]); style(ax)
    Yd = np.arange(len(th))
    for y, (n, f0, f1, a24, ain, c_) in zip(Yd, th):
        ax.annotate("", xy=(f1, y), xytext=(f0, y), arrowprops=dict(arrowstyle="-|>", color=c_, lw=1.2, mutation_scale=7,
                                                                    shrinkA=2.8, shrinkB=2.8))
        ax.plot(f0, y, "o", ms=4.8, mfc="white", mec=c_, mew=1.0)
        ax.plot(f1, y, "o", ms=4.8, color=c_, mec="white", mew=0.6)
    ax.set_yticks(Yd); ax.set_yticklabels([r[0] for r in th], color=INK, fontsize=7.2); ax.set_ylim(len(th) - 0.5, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.912, 0.952); ax.set_xticks([0.915, 0.925, 0.935, 0.945])
    ax.set_xlabel("24-h AUROC, held-out development patients", color=INK)
    tr = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for y, r in zip(Yd, th):
        fig.text(0, 0, "")
        ax.text(0.865, y, r[3], ha="center", va="center", fontsize=7.0, color=INK, transform=tr)
        ax.text(0.945, y, r[4], ha="center", va="center", fontsize=7.0, color=INK, transform=tr)
    ax.text(0.905, -1.35, "Δ AUROC after thinning", ha="center", va="center", fontsize=6.8, color=MUTED, transform=tr)
    ax.text(0.865, -0.75, "24 h", ha="center", va="center", fontsize=6.8, color=MUTED, transform=tr)
    ax.text(0.945, -0.75, "in-hospital", ha="center", va="center", fontsize=6.8, color=MUTED, transform=tr)
    fig.text(0.075 - 0.065, pos.y1 + 0.045, "(d) Charting cadence lowered to the low-testing eICU-CRD rate:"
             " factual (open) → thinned (filled)", ha="left", va="bottom", fontsize=8.5, color=INK)
    save(fig, "Fig6")


# =============================================================== Fig. 7  source-target trade-off (old Table 7)
def fig7():
    cats = ["MIMIC-IV development\n(held-out fold)", "eICU-CRD,\nlow testing", "Hallym"]
    base = {"auroc": [.953, .890, .859], "auprc": [.446, .264, .095]}
    # development differences and 95% patient-cluster bootstrap CI from results/dev_tradeoff_W.json (unrounded)
    dev = {"auroc": [0.0001, 0.0000, 0.0006], "auprc": [(-.0029, -.0168, .0122), (-.0186, -.0362, -.0004), (-.0222, -.0383, -.0055)]}
    mods = [("Generic counterfactual training", "#e87ba4", "o", True, [(.953, .443), (.899, .271), (.867, .128)]),
            ("Proposed, eICU-CRD target", "#2a78d6", "o", True, [(.953, .427), (.916, .294), (.893, .157)]),
            ("Proposed, Hallym target", "#2a78d6", "D", False, [(.953, .423), (.907, .285), (.890, .172)])]
    X = np.array([0, 1.25, 2.25]); off = [-0.28, 0, 0.28]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), gridspec_kw={"wspace": 0.24, "left": 0.08, "right": 0.99,
                                                                     "top": 0.79, "bottom": 0.21})
    for k, (ax, key, lab) in enumerate(zip(axes, ("auroc", "auprc"), ("(a) Δ 24-h AUROC over the backbone",
                                                                     "(b) Δ 24-h AUPRC over the backbone"))):
        style(ax, grid="y")
        ax.axvspan(-0.55, 0.55, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=M["backbone"][1], lw=1.0, ls=(0, (4, 2)))
        for mi, ((n, c_, mk, fill, v), o) in enumerate(zip(mods, off)):
            d = np.array([v[j][k] - base[key][j] for j in range(3)])
            if key == "auprc":
                m0, lo, hi = dev["auprc"][mi]; d[0] = m0
                ax.plot([X[0] + o] * 2, [lo, hi], color=c_, lw=1.4, solid_capstyle="round", zorder=2)
                lx, ly, ha, va = [(X[0] + o, lo - 0.0025, "center", "top"), (X[0] + o, lo - 0.0025, "center", "top"),
                                  (X[0] + o + 0.08, m0, "left", "center")][mi]
                ax.text(lx, ly, f"{m0:+.3f}".replace("-", "−"), fontsize=6.2, ha=ha, va=va,
                        color="#c0507e" if c_ == "#e87ba4" else c_)
            else:
                d[0] = dev["auroc"][mi]
            ax.plot(X + o, d, ls="none", marker=mk, ms=5, color=c_, mfc=c_ if fill else "white", mec="white" if fill else c_,
                    mew=0.6 if fill else 1.1, label=n, zorder=3)
        ax.set_xticks(X); ax.set_xticklabels(cats, color=INK, fontsize=7.2); ax.set_xlim(-0.62, 2.6)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}".replace("-", "−") if v else "0"))
        title(ax, lab)
        bt = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(0, 0.97, "source", transform=bt, ha="center", va="top", fontsize=6.6, color=MUTED)
        ax.text(1.75, 0.97, "targets", transform=bt, ha="center", va="top", fontsize=6.6, color=MUTED)
    axes[0].set_ylim(-0.006, 0.04); axes[1].set_ylim(-0.05, 0.088)
    axes[0].text(0, 0.0035, "unchanged\n(all 0.953)", ha="center", va="bottom", fontsize=6.3, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    h = [Line2D([], [], color=M["backbone"][1], lw=1.0, ls=(0, (4, 2)))] + h; l = ["Backbone (zero line)"] + l
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7.0, bbox_to_anchor=(0.53, 1.0))
    save(fig, "Fig7")


if __name__ == "__main__":
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["3", "4", "5", "6", "7"]
    for w in which: globals()[f"fig{w}"]()
    print("written", OUT)
