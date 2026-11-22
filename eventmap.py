"""Figure 9: per-event localization map for every detector-attribution combination (incl. USAD).
Usage: python eventmap.py results"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
R = sys.argv[1] if len(sys.argv) > 1 else "results"; F = f"{R}/figures"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, DARK = "#2a6fd6", "#c0504d", "#555555"
pe = pd.read_csv(f"{R}/E2_localization_per_event.csv"); ev = pd.read_csv(f"{R}/events.csv")
dets = [d for d in ["PCA", "MLP-AE", "IForest", "Z-max", "USAD"] if d in set(pe.detector)]
combos = [(d, m) for d in dets for m in ["Residual", "Z-mean", "GroupShapley"] if not (d == "IForest" and m == "Residual")]
order = ev.sort_values("magnitude_z").event.tolist()
S = np.zeros((len(order), len(combos)))
for j, (d, m) in enumerate(combos):
    sub = pe[(pe.detector == d) & (pe.method == m)]
    for i, e in enumerate(order):
        h = sub[sub.event == e].hit.to_numpy()
        S[i, j] = 0 if np.mean(~np.isnan(h)) < 0.5 else (2 if np.nanmean(h) >= 0.5 else 1)
pd.DataFrame(S, index=order, columns=[f"{d}|{m}" for d, m in combos]).to_csv(f"{R}/E6_event_map.csv")
fig, ax = plt.subplots(figsize=(7.09, 6.4))
ax.imshow(S, cmap=ListedColormap(["#f2f2f2", RED, BLUE]), aspect="auto", vmin=0, vmax=2)
ax.set_xticks(range(len(combos)))
ax.set_xticklabels([f"{d}\n{'G. Shap.' if m == 'GroupShapley' else ('Resid.' if m == 'Residual' else m)}" for d, m in combos], fontsize=5.8)
lab = ev.set_index("event").loc[order]
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{t:<6} {z:6.1f}" for t, z in zip(lab.targets, lab.magnitude_z)], fontsize=5.5, family="monospace")
for b in [17, 33]:
    ax.axhline(b - 0.5, color=DARK, lw=0.8)
x = -0.5
for d in dets[:-1]:
    x += 2 if d == "IForest" else 3
    ax.axvline(x, color="white", lw=2)
for y_, t in [(8, "high\nstealth"), (25, "medium"), (41.5, "low\nstealth")]:
    ax.text(len(combos) - 0.4, y_, t, fontsize=7, va="center")
ax.set_ylabel("Attack event: target process and deviation (z), sorted by deviation", fontsize=7.5)
ax.legend(handles=[Patch(color=BLUE, label="Detected, correct process"), Patch(color=RED, label="Detected, wrong process"),
                   Patch(color="#f2f2f2", label="Not detected")], frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
for ext in ("png", "tif"):
    fig.savefig(f"{F}/Fig9_event_map.{ext}", dpi=300, bbox_inches="tight", **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
print("event map", S.shape)
