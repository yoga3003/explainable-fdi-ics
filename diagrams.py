"""Colour diagrams for the paper (run after fdi_pipeline.py, analyze.py, extras.py and, for Figure 2, hai22_pipeline.py).
  Figure 1  study pipeline
  Figure 2  HAI grouping hierarchy: process -> control loop -> data point
  Figure 3  exact group Shapley over four processes (with a real case-study attribution)
  Figure 4  trust gate versus duration filter
  Figure 12 deployment architecture
Numbers shown in the diagrams are read from the pipeline outputs.

Usage:
    python diagrams.py --results results --results22 results22
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
W_IN = 7.09  # 180 mm
INK, MUTED = "#1f2937", "#4b5563"
PROC = {"P1": "#2a6fd6", "P2": "#e8902a", "P3": "#3f9c55", "P4": "#8b6fc4"}
PROC_LIGHT = {"P1": "#dbe7fb", "P2": "#fcebd6", "P3": "#dcf0e1", "P4": "#ebe4f7"}
PROC_NAME = {"P1": "Boiler", "P2": "Turbine", "P3": "Water treatment", "P4": "HIL simulation"}
TEAL, TEAL_L = "#138a8a", "#d6f0ef"
BLUE, BLUE_L = "#2a6fd6", "#dbe7fb"
ORANGE, ORANGE_L = "#e8902a", "#fcebd6"
GREEN, GREEN_L = "#3f9c55", "#dcf0e1"
PURPLE, PURPLE_L = "#8b6fc4", "#ebe4f7"
RED, RED_L = "#c8433f", "#f8dedd"
GREY_L = "#f3f4f6"


def save(fig, F, name):
    fig.savefig(f"{F}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{F}/{name}.tif", dpi=300, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


class Canvas:
    """Drawing helper in a top-left coordinate system (x right, y down)."""

    def __init__(self, w, h, inch_h):
        self.W, self.H = w, h
        self.fig, self.ax = plt.subplots(figsize=(W_IN, inch_h))
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.fig.subplots_adjust(0, 0, 1, 1)

    def Y(self, y):
        return self.H - y

    def box(self, x, y, w, h, title=None, lines=(), fc="white", ec=INK, lw=1.0, tsize=8, lsize=7, tcolor=INK, r=7, bold=True):
        self.ax.add_patch(FancyBboxPatch((x, self.Y(y + h)), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fc, ec=ec, lw=lw, zorder=2))
        n = (1 if title else 0) + len(lines)
        top = y + h / 2 - (n - 1) * 6.5
        k = 0
        if title:
            self.ax.text(x + w / 2, self.Y(top), title, ha="center", va="center", fontsize=tsize,
                         weight="bold" if bold else "normal", color=tcolor, zorder=3); k = 1
        for j, l in enumerate(lines):
            self.ax.text(x + w / 2, self.Y(top + 13 * (j + k)), l, ha="center", va="center", fontsize=lsize, color=MUTED, zorder=3)

    def band(self, x, y, w, h, fc, label=None, lc=MUTED):
        self.ax.add_patch(FancyBboxPatch((x, self.Y(y + h)), w, h, boxstyle="round,pad=0,rounding_size=9", fc=fc, ec="none", zorder=0))
        if label:
            self.ax.text(x + 9, self.Y(y + 11), label, fontsize=6.8, weight="bold", color=lc, va="center", zorder=1)

    def arrow(self, x1, y1, x2, y2, color=INK, lw=1.1, style="-|>", ls="-"):
        self.ax.annotate("", xy=(x2, self.Y(y2)), xytext=(x1, self.Y(y1)), zorder=4,
                         arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))

    def line(self, xs, ys, color=INK, lw=1.1):
        self.ax.plot(xs, [self.Y(v) for v in ys], color=color, lw=lw, zorder=1, solid_capstyle="round")

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", 7); kw.setdefault("color", MUTED); kw.setdefault("ha", "center"); kw.setdefault("va", "center")
        self.ax.text(x, self.Y(y), s, zorder=5, **kw)


# ----------------------------------------------------------------------------------------------- Figure 1
def fig_pipeline(R, F):
    a = json.load(open(f"{R}/dataset_audit.json"))
    c = Canvas(760, 400, 3.75)
    c.band(6, 6, 748, 108, TEAL_L, "1  DATA (HAI 21.03, 1 Hz, 57 sensors)", TEAL)
    c.band(6, 124, 748, 108, BLUE_L, "2  MODELS AND ATTRIBUTION (normal data only)", BLUE)
    c.band(6, 242, 748, 152, GREY_L, "3  EVALUATION AGAINST RECORDED ATTACK LABELS", INK)
    W, xs = 222, [24, 269, 514]
    c.box(xs[0], 30, W, 70, "Training files", ["normal operation only", f"{a['fit_rows']:,} fitting rows"], ec=TEAL, lw=1.4)
    c.box(xs[1], 30, W, 70, "Calibration tail", ["last 20% of final training file", f"{a['cal_rows']:,} rows"], ec=TEAL, lw=1.4)
    c.box(xs[2], 30, W, 70, "Test files", [f"{a['test_rows']:,} rows", f"{a['events']} real attacks, labels held back"], ec=TEAL, lw=1.4)
    c.box(xs[0], 148, W, 70, "Five detectors", ["PCA, MLP autoencoder, isolation", "forest, Z-max, USAD (deep)"], ec=BLUE, lw=1.4)
    c.box(xs[1], 148, W, 70, "Score and alarm", ["causal 60-s mean;", "threshold = q-quantile (q = 0.999)"], ec=BLUE, lw=1.4)
    c.box(xs[2], 148, W, 70, "Process attribution", ["exact group Shapley (16 evaluations)", "vs residual, Z-mean, Kernel SHAP"], ec=ORANGE, lw=1.8, fc="#fff7ee")
    for x in xs:
        c.arrow(x + W / 2, 100, x + W / 2, 148, color=MUTED)
    c.arrow(xs[0] + W, 183, xs[1], 183); c.arrow(xs[1] + W, 183, xs[2], 183)
    outs = [("Detection", ["point F1, AUC-PR, event recall,", "delay, false alarms per hour", "Tables 3, 10"], GREEN, GREEN_L),
            ("Localization (RQ1, RQ2)", ["top process vs attacked process;", "stealth tiers, oracle, baseline", "Figs 5, 6, 9, 11; Tables 4-6, 11-13"], ORANGE, ORANGE_L),
            ("Trust gate (RQ3)", ["alert after 10 stable seconds", "vs duration filter", "Fig. 7; Table 7"], PURPLE, PURPLE_L)]
    for x, (t, l, ec, fc) in zip(xs, outs):
        c.box(x, 286, W, 92, t, l, fc=fc, ec=ec, lw=1.6)
    c.line([xs[1] + W / 2, xs[1] + W / 2], [218, 266], color=MUTED)
    c.line([xs[0] + W / 2, xs[2] + W / 2], [266, 266], color=MUTED)
    c.line([xs[2] + W / 2, xs[2] + W / 2], [218, 266], color=MUTED)
    for x in xs:
        c.arrow(x + W / 2, 266, x + W / 2, 286, color=MUTED)
    save(c.fig, F, "Fig1_pipeline")


# ----------------------------------------------------------------------------------------------- Figure 2
def fig_hierarchy(R, R22, F):
    a = json.load(open(f"{R}/dataset_audit.json")); b = json.load(open(f"{R22}/audit22.json"))
    loops = b["loop_groups"]
    c = Canvas(760, 330, 3.2)
    c.box(190, 8, 380, 40, "HAI testbed (boiler, turbine, water treatment, HIL)", fc=GREY_L, ec=INK, tsize=7.5)
    c.text(14, 28, "Level", ha="left", weight="bold", color=INK, fontsize=7)
    c.text(14, 96, "Process", ha="left", weight="bold", color=INK, fontsize=7.5)
    c.text(14, 108, "4 groups", ha="left", fontsize=6.5)
    c.text(14, 196, "Control loop", ha="left", weight="bold", color=INK, fontsize=7.5)
    c.text(14, 208, "11 groups", ha="left", fontsize=6.5)
    c.text(14, 286, "Data point", ha="left", weight="bold", color=INK, fontsize=7.5)
    c.text(14, 298, f"{b['features_used']} points", ha="left", fontsize=6.5)
    by_p = {p: [L for L in loops if L.startswith(p)] for p in PROC}
    wt = {L: (1.6 if len(by_p[L[:2]]) == 1 else 1.0) for L in loops}
    x0 = 96; unit = (752 - x0) / sum(wt.values())
    c.text(14, 120, "sensors:", ha="left", fontsize=6)
    c.text(14, 130, "21.03 / 22.04", ha="left", fontsize=6)
    two = {"P3": ["Water", "treatment"], "P4": ["HIL", "simulation"]}
    x = x0
    for p in PROC:
        w = sum(wt[L] for L in by_p[p]) * unit
        px = x
        if p in two:
            c.box(px + 3, 68, w - 6, 62, p, two[p] + [f"{a['group_sizes'][p]} / {b['process_groups'][p]}"],
                  fc=PROC_LIGHT[p], ec=PROC[p], lw=1.6, tsize=7.5, lsize=6.2)
        else:
            c.box(px + 3, 68, w - 6, 62, f"{p}  {PROC_NAME[p]}", [f"{a['group_sizes'][p]} / {b['process_groups'][p]} sensors"],
                  fc=PROC_LIGHT[p], ec=PROC[p], lw=1.6, tsize=7.5, lsize=6.4)
        c.arrow(380, 48, px + w / 2, 68, color=MUTED, lw=0.8)
        cx = px
        for L in by_p[p]:
            lw_ = wt[L] * unit
            c.box(cx + 3, 170, lw_ - 6, 48, L, [f"{loops[L]} points"], fc="white", ec=PROC[p], lw=1.3, tsize=6.2, lsize=5.9)
            c.arrow(px + w / 2, 130, cx + lw_ / 2, 170, color=PROC[p], lw=0.7)
            m = min(loops[L], 6)
            for k in range(m):
                dx = cx + lw_ / 2 + (k - (m - 1) / 2) * 7
                c.ax.add_patch(Circle((dx, c.Y(272)), 2.5, fc=PROC[p], ec="none", zorder=3))
            if loops[L] > 6:
                c.text(cx + lw_ / 2, 285, f"+{loops[L] - 6}", fontsize=5.8)
            c.line([cx + lw_ / 2, cx + lw_ / 2], [218, 266], color=PROC[p], lw=0.5)
            cx += lw_
        x += w
    c.text(380, 318, "Localization is scored at each level: correct process, correct control loop, and target point in the top 1, 3 or 5",
           fontsize=6.8, color=INK)
    save(c.fig, F, "Fig2_hierarchy")


# ----------------------------------------------------------------------------------------------- Figure 3
def fig_shapley(R, F):
    cs = pd.read_csv(f"{R}/E7_case_studies.csv")
    ev = 3 if (cs.event == 3).any() else int(cs.event.iloc[0])
    sub = cs[cs.event == ev].set_index("method"); target = sub.targets.iloc[0]
    fig = plt.figure(figsize=(W_IN, 3.3))
    # panel a: coalition lattice
    ax = fig.add_axes([0.0, 0.0, 0.65, 1.0]); ax.set_xlim(0, 500); ax.set_ylim(0, 300); ax.axis("off")
    ax.text(4, 288, "a", fontsize=10, weight="bold", color=INK)
    ax.text(22, 288, "Coalitions S of the four processes: 2⁴ = 16 detector evaluations per second", fontsize=7.5, color=INK, va="center")
    import itertools
    G = list(PROC)
    levels = [list(itertools.combinations(G, k)) for k in range(5)]
    pos = {}
    for k, combs in enumerate(levels):
        y = 255 - k * 52
        for i, S in enumerate(combs):
            x = 300 + (i - (len(combs) - 1) / 2) * 66
            pos[S] = (x, y)
    for k in range(4):
        for S in levels[k]:
            for T in levels[k + 1]:
                if set(S) <= set(T):
                    g = (set(T) - set(S)).pop()
                    ax.plot([pos[S][0], pos[T][0]], [pos[S][1] - 9, pos[T][1] + 9], color=PROC[g], lw=0.6, alpha=0.55, zorder=1)
    for S, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - 27, y - 9), 54, 18, boxstyle="round,pad=0,rounding_size=4", fc="white", ec="#9ca3af", lw=0.7, zorder=2))
        for j, g in enumerate(G):
            ax.add_patch(plt.Rectangle((x - 22 + j * 11.5, y - 5), 9, 10, fc=PROC[g] if g in S else "#e5e7eb", ec="none", zorder=3))
    ax.text(235 + 2.5 * 70 + 14, 255, "v(∅): every process\nat training median", fontsize=6.3, color=MUTED, va="center", ha="left")
    ax.text(300 + 32, 47, "v(G) = s(x): observed sample", fontsize=6.3, color=MUTED, va="center", ha="left")
    for j, g in enumerate(G):
        ax.add_patch(plt.Rectangle((12, 200 - j * 18), 9, 10, fc=PROC[g], ec="none"))
        ax.text(26, 205 - j * 18, f"{g} observed", fontsize=6.5, va="center", color=INK)
    ax.add_patch(plt.Rectangle((12, 128), 9, 10, fc="#e5e7eb", ec="none"))
    ax.text(26, 133, "replaced by median", fontsize=6.5, va="center", color=INK)
    ax.text(12, 95, "φ(g) = weighted mean of\nv(S ∪ {g}) − v(S)\nover the 8 coalitions\nwithout g (colored\nedges, one colour\nper process g)", fontsize=6.4, color=INK, va="top", linespacing=1.35)
    # panel b: real attribution for one event
    ax2 = fig.add_axes([0.72, 0.14, 0.27, 0.62])
    meths = ["Residual", "Group Shapley"]
    w = 0.38
    for i, m in enumerate(meths):
        vals = [sub.loc[m, f"share_{g}"] for g in G]
        for j, g in enumerate(G):
            ax2.bar(j + (i - 0.5) * w, vals[j], w * 0.92, color=PROC[g] if m == "Group Shapley" else PROC_LIGHT[g],
                    ec=PROC[g], lw=0.8, hatch=None if m == "Group Shapley" else "////", zorder=3)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(G, fontsize=7.5)
    ax2.set_ylabel("Mean attribution share", fontsize=7); ax2.tick_params(labelsize=6.5)
    ax2.set_ylim(0, 0.8); ax2.grid(axis="y", color="#eeeeee", zorder=0)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(fc="white", ec=INK, hatch="////", label="Residual"), Patch(fc=INK, label="Group Shapley")],
               fontsize=6.3, frameon=False, loc="upper right")
    fig.text(0.675, 0.94, "b", fontsize=10, weight="bold", color=INK)
    fig.text(0.70, 0.94, f"Event {ev}: {target} attacked (PCA detector)", fontsize=7.3, color=INK, va="center")
    fig.text(0.70, 0.865, "residual points to P3,\ngroup Shapley to P2", fontsize=6.6, color=MUTED, va="center")
    save(fig, F, "Fig3_group_shapley")


# ----------------------------------------------------------------------------------------------- Figure 4
def fig_trust_gate(F):
    c = Canvas(760, 372, 3.5)
    c.band(6, 6, 478, 360, PURPLE_L, "TRUST GATE (this study)", PURPLE)
    c.band(494, 6, 260, 360, GREY_L, "DURATION FILTER (baseline)", INK)
    c.box(26, 30, 218, 44, "Alarm second t", ["smoothed score above threshold"], ec=INK, lw=1.2)
    c.box(26, 98, 218, 52, "Five noisy copies of the input", ["Gaussian noise, \u03c3 \u2208 {0.05, 0.2, 0.5}", "(z-units)"], ec=BLUE, fc=BLUE_L, lw=1.3)
    c.box(26, 174, 218, 44, "Residual attribution", ["top process of each copy"], ec=ORANGE, fc=ORANGE_L, lw=1.3)
    cx, cy = 135, 264
    c.ax.add_patch(plt.Polygon([(cx, c.Y(cy - 36)), (cx + 96, c.Y(cy)), (cx, c.Y(cy + 36)), (cx - 96, c.Y(cy))], fc="white", ec=PURPLE, lw=1.5, zorder=2))
    c.text(cx, cy - 12, "Original top process", fontsize=7, color=INK, weight="bold")
    c.text(cx, cy, "kept in \u2265 4 of 5", fontsize=7, color=INK, weight="bold")
    c.text(cx, cy + 12, "copies?", fontsize=7, color=INK, weight="bold")
    for y1, y2 in [(74, 98), (150, 174), (218, 228)]:
        c.arrow(135, y1, 135, y2)
    c.box(290, 26, 176, 52, "Raise alert", ["once the segment holds", "10 stable seconds"], ec=GREEN, fc="white", lw=1.8, tcolor=GREEN)
    c.box(290, 110, 176, 52, "Stable count + 1", ["stable seconds so far", "in this alarm segment"], ec=GREEN, fc=GREEN_L, lw=1.3)
    c.box(290, 296, 176, 52, "Not counted", ["count unchanged; check", "the next alarm second"], ec=RED, fc=RED_L, lw=1.3)
    c.line([231, 378], [264, 264], color=GREEN); c.arrow(378, 264, 378, 162, color=GREEN)
    c.text(262, 255, "yes", color=GREEN, weight="bold")
    c.arrow(378, 110, 378, 74, color=GREEN)
    c.line([135, 135], [300, 322], color=RED); c.arrow(135, 322, 290, 322, color=RED)
    c.text(150, 312, "no", color=RED, weight="bold")
    c.box(526, 30, 196, 44, "Alarm second t", ["smoothed score above threshold"], ec=INK, lw=1.2)
    c.box(526, 130, 196, 52, "Consecutive count + 1", ["no use of the explanation"], ec=MUTED, fc="white", lw=1.2)
    c.box(526, 236, 196, 52, "Raise alert", ["after k seconds,", "k \u2208 {1, 5, 10, \u2026, 120}"], ec=INK, fc="white", lw=1.6)
    c.arrow(624, 74, 624, 130, color=MUTED); c.arrow(624, 182, 624, 236, color=MUTED)
    c.text(624, 318, "Compared on the full recall vs", fontsize=6.6)
    c.text(624, 330, "false-alarm curve (Figure 7)", fontsize=6.6)
    save(c.fig, F, "Fig4_trust_gate_flow")


# ----------------------------------------------------------------------------------------------- Figure 12
def fig_architecture(F):
    c = Canvas(760, 372, 3.6)
    c.band(8, 6, 744, 112, TEAL_L, "OFFLINE  (normal operation only)", TEAL)
    c.band(8, 128, 744, 128, BLUE_L, "ONLINE  (every second)", BLUE)
    c.band(8, 266, 744, 100, GREY_L, "OPERATOR AND RESPONSE", INK)
    W4, G4 = 168, 18
    xs4 = [22 + k * (W4 + G4) for k in range(4)]
    off = [("Historian data", ["1-Hz sensor and actuator", "tags (OPC, Modbus)"]), ("Fit detector", ["standardize; any model:", "PCA, AE, USAD, IForest"]),
           ("Calibrate", ["threshold q on a held-out", "normal tail"]), ("Process map", ["sensor → process / loop", "(P1 … P4)"])]
    for x, (t, l) in zip(xs4, off):
        c.box(x, 30, W4, 70, t, l, ec=TEAL, lw=1.4)
    for k in range(3):
        c.arrow(xs4[k] + W4, 65, xs4[k + 1], 65, color=TEAL)
    W5, G5 = 132, 15
    xs5 = [22 + k * (W5 + G5) for k in range(5)]
    on = [("Live tags", ["buffered and", "standardized"], INK, "white"), ("Detector score", ["causal 60-s mean,", "compared with q"], BLUE, "white"),
          ("Group Shapley", ["16 evaluations;", "top = location"], ORANGE, ORANGE_L), ("Trust gate", ["alert after 10", "stable seconds"], PURPLE, PURPLE_L),
          ("Operator alert", ["process, share,", "stability"], GREEN, GREEN_L)]
    for x, (t, l, ec, fc) in zip(xs5, on):
        c.box(x, 152, W5, 70, t, l, ec=ec, fc=fc, lw=1.6)
    for k in range(4):
        c.arrow(xs5[k] + W5, 187, xs5[k + 1], 187)
    c.text(xs5[1] + W5 / 2, 236, "below threshold: no alert", fontsize=6.3)
    c.text(xs5[3] + W5 / 2, 236, "unstable: hold and re-check", fontsize=6.3)
    c.arrow(xs4[1] + W4 / 2, 100, xs5[1] + W5 / 2, 152, color=TEAL, ls="--")
    c.arrow(xs4[2] + W4 / 2, 100, xs5[1] + W5 * 0.8, 152, color=TEAL, ls="--")
    c.arrow(xs4[3] + W4 / 2, 100, xs5[2] + W5 / 2, 152, color=TEAL, ls="--")
    c.box(150, 290, 210, 60, "Inspect the named process", ["check its loops and points", "(hierarchical ranking)"], ec=INK, lw=1.2)
    c.box(400, 290, 210, 60, "Log for audit", ["alert, attribution share,", "stability, timestamps"], ec=INK, lw=1.2)
    ax_ = xs5[4] + W5 / 2
    c.line([ax_, ax_], [222, 276], color=GREEN); c.line([255, ax_], [276, 276], color=GREEN)
    c.arrow(255, 276, 255, 290, color=GREEN); c.arrow(505, 276, 505, 290, color=GREEN)
    save(c.fig, F, "Fig12_architecture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--results22", default="results22")
    a = ap.parse_args()
    F = f"{a.results}/figures"; os.makedirs(F, exist_ok=True)
    fig_pipeline(a.results, F)
    if os.path.exists(f"{a.results22}/audit22.json"):
        fig_hierarchy(a.results, a.results22, F)
    fig_shapley(a.results, F)
    fig_trust_gate(F)
    fig_architecture(F)
    print("diagrams ok ->", F)


if __name__ == "__main__":
    main()
