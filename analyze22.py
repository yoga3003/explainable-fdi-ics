"""Summaries, tests and figures for HAI 22.04 (run after hai22_pipeline.py)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results22"
F = f"{R}/figures"; os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, LIGHT = "#2a6fd6", "#bdbcae", "#e3e2dd"
DETS = ["PCA", "MLP-AE", "IForest", "Z-max", "USAD"]

audit = json.load(open(f"{R}/audit22.json"))
ev = pd.read_csv(f"{R}/events22.csv")
d = pd.read_csv(f"{R}/H1_detection.csv")
DETS = [x for x in DETS if x in set(d.detector)]
g = d.groupby("detector", sort=False)
cols = ["precision", "recall", "f1", "auc_pr", "event_recall", "median_delay_s", "false_alarms_per_hour"]
t10 = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd")).reset_index()
t10.to_csv(f"{R}/T10_detection22.csv", index=False)

pe = pd.read_csv(f"{R}/H2_localization_per_event.csv").merge(
    ev[["id", "single_loop", "single_proc", "ambiguous", "stealth_tier"]], on="id")
rows = []
for lvl, col in [("process", "single_proc"), ("loop", "single_loop")]:
    sub = pe[(pe.level == lvl) & (pe[col] == 1) & (pe.ambiguous == 0)]
    per_seed = sub.groupby(["detector", "method", "seed"]).agg(hit=("hit", "mean"), oracle=("hit_oracle", "mean"),
                                                               detected=("detected", "sum")).reset_index()
    s = per_seed.groupby(["detector", "method"]).agg(hit_mean=("hit", "mean"), hit_sd=("hit", "std"),
                                                     oracle_mean=("oracle", "mean"), oracle_sd=("oracle", "std"),
                                                     detected=("detected", "mean")).reset_index()
    s["level"] = lvl; rows.append(s)
t11 = pd.concat(rows); t11.to_csv(f"{R}/T11_localization22.csv", index=False)

pt = pe[pe.level == "point-oracle"].groupby(["detector", "method", "seed"])[["hit", "hit3", "hit5"]].mean().reset_index()
t12 = pt.groupby(["detector", "method"])[["hit", "hit3", "hit5"]].mean().reset_index()
t12["events"] = pe[pe.level == "point-oracle"].groupby(["detector", "method", "seed"]).id.count().groupby(["detector", "method"]).mean().values
t12.to_csv(f"{R}/T12_points22.csv", index=False)

# exact McNemar on oracle hits (every attack scored), majority over seeds
tests = []
for lvl, col in [("process", "single_proc"), ("loop", "single_loop")]:
    for det in DETS:
        sub = pe[(pe.level == lvl) & (pe[col] == 1) & (pe.ambiguous == 0) & (pe.detector == det)]
        piv = sub.pivot_table(index="id", columns="method", values="hit_oracle", aggfunc="mean")
        for other in ["Residual", "Z-mean", "Occlusion"]:
            if other in piv and "GroupShapley" in piv:
                a_ = piv.GroupShapley >= 0.5; b_ = piv[other] >= 0.5
                n01, n10 = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
                p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                tests.append(dict(level=lvl, detector=det, comparison=f"GroupShapley vs {other}", events=len(piv),
                                  gs_only=n01, other_only=n10, p=p))
if os.path.exists(f"{R}/H11_kernelshap_per_event.csv"):
    ks = pd.read_csv(f"{R}/H11_kernelshap_per_event.csv").merge(ev[["id", "single_loop", "single_proc", "ambiguous"]], on="id")
    rows_ = []
    for lvl, col in [("process", "single_proc"), ("loop", "single_loop")]:
        sub = ks[(ks.level == lvl) & (ks[col] == 1) & (ks.ambiguous == 0)]
        ps = sub.groupby(["detector", "method", "seed"]).hit_oracle.mean().reset_index()
        r_ = ps.groupby(["detector", "method"]).hit_oracle.agg(["mean", "std"]).reset_index(); r_["level"] = lvl; rows_.append(r_)
        for det in DETS:
            piv = sub[sub.detector == det].pivot_table(index="id", columns="method", values="hit_oracle", aggfunc="mean")
            if piv.empty:
                continue
            a_ = piv["GroupShapley (same seconds)"] >= 0.5; b_ = piv["KernelSHAP"] >= 0.5
            n01, n10 = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            tests.append(dict(level=lvl, detector=det, comparison="GroupShapley vs KernelSHAP (same seconds)", events=len(piv),
                              gs_only=n01, other_only=n10, p=p))
    pd.concat(rows_).to_csv(f"{R}/T11b_kernelshap22.csv", index=False)
pd.DataFrame(tests).to_csv(f"{R}/T13_mcnemar22.csv", index=False)

# Figure 8: localization at process and loop level, oracle (all attack seconds), with majority baselines
fig, axs = plt.subplots(1, 2, figsize=(7.09, 3.4), sharey=True)
meths = [("Residual", LIGHT), ("Z-mean", GREY), ("Occlusion", "#8fb3e8"), ("GroupShapley", BLUE)]
for ax, lvl, base_, lab in [(axs[0], "process", audit["majority_process_rate"], "Process (4 groups)"),
                            (axs[1], "loop", audit["majority_loop_rate"], "Control loop (11 groups)")]:
    sub = t11[t11.level == lvl]; ms = [m for m in meths if m[0] in sub.method.unique()]; w = 0.8 / len(ms)
    for j, (m, c) in enumerate(ms):
        for i, det in enumerate(DETS):
            r = sub[(sub.detector == det) & (sub.method == m)]
            if len(r):
                v = r.oracle_mean.iloc[0] * 100; x = i + (j - (len(ms) - 1) / 2) * w
                ax.bar(x, v, w * 0.92, color=c, label=("Group Shapley" if m == "GroupShapley" else m) if i == 1 else None, zorder=3)
    ax.axhline(base_ * 100, ls="--", c="#555555", lw=1.0, zorder=4, label=f"Majority baseline ({base_*100:.1f}%)")
    ax.set_xticks(range(len(DETS))); ax.set_xticklabels(DETS, fontsize=8); ax.set_title(lab, loc="left", fontsize=9, weight="bold")
    ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.grid(axis="y", color="#eeeeee", zorder=0)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.07))
axs[0].set_ylabel("Correct target, % of single-target attacks\n(every attack second explained)", fontsize=7.5)
fig.savefig(f"{F}/Fig11_hai22_localization.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{F}/Fig11_hai22_localization.tif", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print("ok")
