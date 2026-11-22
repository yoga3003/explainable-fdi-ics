"""Aggregate pipeline outputs into the paper's tables (mean ± SD over seeds),
paired significance tests, and 300-dpi figures (PNG + TIFF for Frontiers)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results"
F = f"{R}/figures"; os.makedirs(F, exist_ok=True)
ORDER = ["PCA", "MLP-AE", "IForest", "Z-max", "USAD"]
fmt = lambda m, s: f"{m:.3f} ± {s:.3f}" if not np.isnan(s) and s > 0 else f"{m:.3f}"

def agg(df, keys, cols):
    g = df.groupby(keys, sort=False)
    out = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd")).reset_index()
    out["n_seeds"] = g.size().values
    return out

# ---- Table 2: detection
d = pd.read_csv(f"{R}/E1_detection.csv")
ORDER = [o for o in ORDER if o in set(d.detector)]
cols = ["precision", "recall", "f1", "auc_pr", "event_recall", "median_delay_s", "false_alarms_per_hour"]
t2 = agg(d, ["detector"], cols); t2.to_csv(f"{R}/T2_detection_summary.csv", index=False)

# ---- Table 3: localization (single-target events), plus oracle
pe = pd.read_csv(f"{R}/E2_localization_per_event.csv")
loc = pd.read_csv(f"{R}/E2_localization.csv")
t3 = agg(loc, ["detector", "method"], ["hit_rate_single", "ci_low", "ci_high", "detected_single", "hit_rate_multi"])
orc = pe[pe.single].groupby(["detector", "method", "seed"]).hit_oracle.mean().reset_index()
orc = orc.groupby(["detector", "method"]).hit_oracle.agg(["mean", "std"]).reset_index().rename(
    columns={"mean": "oracle_hit_mean", "std": "oracle_hit_sd"})
t3 = t3.merge(orc, on=["detector", "method"]); t3.to_csv(f"{R}/T3_localization_summary.csv", index=False)

# ---- paired test: GroupShapley vs Residual / Z-mean (exact McNemar on events;
#      per-event outcome = majority over seeds, detected single-target events only)
tests = []
for det in ORDER:
    sub = pe[(pe.detector == det) & pe.single]
    piv = sub.pivot_table(index="event", columns="method", values="hit", aggfunc="mean")
    for other in ["Residual", "Z-mean"]:
        if other not in piv or "GroupShapley" not in piv:
            continue
        p = piv[["GroupShapley", other]].dropna()
        a = (p.GroupShapley >= 0.5); b = (p[other] >= 0.5)
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
        tests.append(dict(detector=det, comparison=f"GroupShapley vs {other}", events=len(p),
                          gs_only_correct=n01, other_only_correct=n10, exact_mcnemar_p=pv))
# Kernel SHAP (per sensor, summed to processes) vs group Shapley on the same sampled alarm seconds
if os.path.exists(f"{R}/E11_kernelshap_per_event.csv"):
    ks = pd.read_csv(f"{R}/E11_kernelshap_per_event.csv"); ks = ks[ks.single]
    s_ = ks.groupby(["detector", "method", "seed"]).hit.agg(["mean", "count"]).reset_index()
    t3c = s_.groupby(["detector", "method"]).agg(hit_mean=("mean", "mean"), hit_sd=("mean", "std"), events=("count", "mean")).reset_index()
    t3c.to_csv(f"{R}/T3c_kernelshap_summary.csv", index=False)
    for det in ORDER:
        piv = ks[ks.detector == det].pivot_table(index="event", columns="method", values="hit", aggfunc="mean")
        if piv.empty:
            continue
        a = piv["GroupShapley (same seconds)"] >= 0.5; b = piv["KernelSHAP"] >= 0.5
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
        tests.append(dict(detector=det, comparison="GroupShapley vs KernelSHAP (same seconds)", events=len(piv),
                          gs_only_correct=n01, other_only_correct=n10, exact_mcnemar_p=pv))
pd.DataFrame(tests).to_csv(f"{R}/T3b_mcnemar.csv", index=False)

# ---- Table 4: stealth tiers
st = pd.read_csv(f"{R}/E3_stealth_tiers.csv")
t4 = agg(st, ["detector", "method", "tier"], ["detection_rate", "localization_rate_all", "localization_rate_detected"])
t4.to_csv(f"{R}/T4_stealth_summary.csv", index=False)
# oracle localization by tier (explanation independent of detection)
ev = pd.read_csv(f"{R}/events.csv")
ot = pe.groupby(["detector", "method", "seed", "stealth_tier"]).hit_oracle.mean().reset_index()
ot = ot.groupby(["detector", "method", "stealth_tier"]).hit_oracle.agg(["mean", "std"]).reset_index()
ot.to_csv(f"{R}/T4b_oracle_by_tier.csv", index=False)

# ---- Table 5: trust gate
g = pd.read_csv(f"{R}/E4_trust_gate.csv")
g["noise_sd"] = g["noise_sd"].fillna(-1)
t5 = agg(g, ["detector", "thr_quantile", "policy", "noise_sd"],
         ["event_recall", "false_alarms_per_hour", "median_delay_s", "f1", "stable_share_attack", "stable_share_normal"])
t5["noise_sd"] = t5["noise_sd"].replace(-1, np.nan)
t5.to_csv(f"{R}/T5_trust_gate_summary.csv", index=False)

# ---------------------------------------------------------------- figures
# Figures 1-4 exactly as in the paper, 300 dpi, PNG + TIFF (Frontiers: 180 mm wide)
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, LIGHT, DARK = "#2a6fd6", "#bdbcae", "#e3e2dd", "#555555"
W_IN = 7.09  # 180 mm

def save(fig, name):
    fig.savefig(f"{F}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{F}/{name}.tif", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

audit = json.load(open(f"{R}/dataset_audit.json"))

# ---- Figure 1: study pipeline
fig, ax = plt.subplots(figsize=(W_IN, 3.9)); ax.set_xlim(0, 760); ax.set_ylim(0, 380); ax.axis("off")
def box(cx, y, h, title, lines, main=False):
    ax.add_patch(FancyBboxPatch((cx - 113, 380 - y - h), 226, h, boxstyle="round,pad=0,rounding_size=8",
                                fc="#e6eefb" if main else "white", ec=BLUE if main else DARK, lw=1.6 if main else 0.9))
    ax.text(cx, 380 - y - 20, title, ha="center", va="center", fontsize=8.5, weight="bold")
    for j, l in enumerate(lines):
        ax.text(cx, 380 - y - 38 - 15 * j, l, ha="center", va="center", fontsize=8, color="#333333")
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, 380 - y2), xytext=(x1, 380 - y1), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.9))
c1, c2, c3 = 129, 374, 619
box(c1, 30, 70, "Training data", ["HAI 21.03, normal only", f"{audit['fit_rows']:,} rows"])
box(c2, 30, 70, "Four detectors", ["PCA, MLP-AE,", "IForest, Z-max"])
box(c3, 30, 70, "Alarm threshold", ["q-quantile of scores on", f"held-out tail, {audit['cal_rows']:,} rows"])
box(c1, 145, 70, "Test data", [f"{audit['test_rows']:,} rows", f"{audit['events']} real attack events"])
box(c2, 145, 70, "Score and alarm", ["causal 60-s mean,", "alarm above threshold"])
box(c1, 270, 84, "Detection", ["point F1, event recall,", "false alarms per hour", "Table 3"])
box(c2, 270, 84, "Localization (RQ1, RQ2)", ["residual, Z-mean, group", "Shapley vs labels; stealth", "Fig. 2–3, Tables 4–6"], main=True)
box(c3, 270, 84, "Trust gate (RQ3)", ["alarm after 10 stable s", "vs duration filter", "Table 7, Fig. 4"])
arrow(c1 + 113, 65, c2 - 113, 65); arrow(c2 + 113, 65, c3 - 113, 65)
ax.plot([c3, c3, c2 + 120], [380 - 100, 380 - 180, 380 - 180], color=DARK, lw=0.9); arrow(c2 + 120, 180, c2 + 113, 180)
arrow(c1 + 113, 180, c2 - 113, 180)
ax.plot([c2, c2], [380 - 215, 380 - 245], color=DARK, lw=0.9)
ax.plot([c1, c3], [380 - 245, 380 - 245], color=DARK, lw=0.9)
for x in (c1, c2, c3): arrow(x, 245, x, 270)
save(fig, "Fig1_pipeline")

# ---- Figure 2: localization hit rate by detector x method
fig, ax = plt.subplots(figsize=(W_IN, 3.2))
meths = [("Residual", "Residual", LIGHT), ("Z-mean", "Z-mean", GREY), ("GroupShapley", "Group Shapley", BLUE)]; w = 0.27
for j, (m, lab, col) in enumerate(meths):
    for i, det in enumerate(ORDER):
        r = t3[(t3.detector == det) & (t3.method == m)]
        if len(r):
            v = r.hit_rate_single_mean.iloc[0] * 100; x = i + (j - 1) * w
            ax.bar(x, v, w * 0.92, color=col, label=lab if i == 1 else None)
            ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
maj = audit["majority_class_hit_rate"] * 100
ax.axhline(maj, ls="--", c="#888888", lw=0.9, label=f"Always P1 baseline ({maj:.1f}%)")
labels = [f"{d}\n{t3[t3.detector == d].detected_single_mean.iloc[0]:.0f} detected" for d in ORDER]
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(labels); ax.set_ylim(0, 108); ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
save(fig, "Fig5_localization")

# ---- Figure 3: stealth tiers, MLP-AE (detected, group Shapley, residual)
tiers = ["high stealth", "medium", "low stealth"]; tl = ["High stealth", "Medium", "Low stealth"]
fig, ax = plt.subplots(figsize=(W_IN, 3.2))
sub = t4[t4.detector == "MLP-AE"]
series = [("Detected", lambda t: sub[(sub.method == "GroupShapley") & (sub.tier == t)].detection_rate_mean.iloc[0], GREY),
          ("Localized, group Shapley", lambda t: sub[(sub.method == "GroupShapley") & (sub.tier == t)].localization_rate_detected_mean.iloc[0], BLUE),
          ("Localized, residual", lambda t: sub[(sub.method == "Residual") & (sub.tier == t)].localization_rate_detected_mean.iloc[0], LIGHT)]
n_ev = ev.stealth_tier.value_counts()
for j, (lab, fn, col) in enumerate(series):
    for i, t in enumerate(tiers):
        v = fn(t) * 100; x = i + (j - 1) * 0.27
        ax.bar(x, v, 0.25, color=col, label=lab if i == 0 else None)
        ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{a}\n{n_ev[t]} events" for a, t in zip(tl, tiers)])
ax.set_ylim(0, 108); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "Fig6_stealth")

# ---- Figure 4: trust gate vs duration-filter frontier, q = 0.99
c = pd.read_csv(f"{R}/E4_duration_curve.csv")
fig, axs = plt.subplots(1, 2, figsize=(W_IN, 3.0), sharey=True)
best = {"PCA": 0.05, "MLP-AE": 0.5}
for ax, det in zip(axs, ["PCA", "MLP-AE"]):
    cc = c[(c.detector == det) & (c.thr_quantile == 0.99)].groupby("k")[["false_alarms_per_hour", "event_recall"]].mean().sort_values("false_alarms_per_hour")
    ax.plot(cc.false_alarms_per_hour, cc.event_recall * 100, "o-", ms=3, c=GREY, lw=1.2, label="Duration filter, k = 1–120 s")
    for k in (1, 60):
        ax.annotate(f"k = {k} s", (cc.loc[k].false_alarms_per_hour, cc.loc[k].event_recall * 100), textcoords="offset points",
                    xytext=(0, -12 if det == "PCA" else 7), ha="right" if k == 1 else "center", fontsize=7, color="#555555")
    gg = t5[(t5.detector == det) & (t5.thr_quantile == 0.99) & (t5.policy == "trust gate")]
    for _, r in gg.iterrows():
        key = abs(r.noise_sd - best[det]) < 1e-9
        ax.scatter(r.false_alarms_per_hour_mean, r.event_recall_mean * 100, s=40 if key else 18, c=BLUE,
                   alpha=1 if key else 0.45, zorder=3, label="Trust gate" if key else None)
        if key:
            ax.annotate(f"σ = {r.noise_sd:g}: {r.false_alarms_per_hour_mean:.3f}/h, {r.median_delay_s_mean:.1f} s",
                        (r.false_alarms_per_hour_mean, r.event_recall_mean * 100), textcoords="offset points",
                        xytext=(6, 7) if det == "PCA" else (6, -16), ha="left", fontsize=7.5, weight="bold")
    ax.set_title(det, loc="left", fontsize=9, weight="bold"); ax.set_xlabel("False alarms per hour"); ax.set_ylim(28, 103)
    ax.grid(axis="y", color="#eeeeee")
axs[0].set_ylabel("Event recall (%)"); axs[0].legend(frameon=False, fontsize=7, loc="lower right")
save(fig, "Fig7_trust_gate")
print("ok")
