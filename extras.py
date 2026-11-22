"""
Additional analyses for the paper (run after fdi_pipeline.py and analyze.py):
  E5  ablation: smoothing window x threshold quantile (PCA)            -> Figure 5, Table 8
  E5b ablation: Shapley reference value, calibration size, occlusion   -> Table 8
  E6  per-event localization map (all detectors/methods, from E2 CSV)  -> Figure 6
  E7  case studies: score, threshold and attribution over time (PCA)   -> Figure 7
  E8  computational cost per detector and attribution                  -> Table 9
  Figure 9: deployment architecture

Usage:
    python extras.py --data hai/hai-21.03 --results results
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fdi_pipeline as fp

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, LIGHT, DARK, RED = "#2a6fd6", "#bdbcae", "#e3e2dd", "#555555", "#c0504d"
PROC_COL = {"P1": "#2a6fd6", "P2": "#e8a33d", "P3": "#5aa469", "P4": "#9a8fbf"}
W_IN = 7.09


def save(fig, F, name):
    fig.savefig(f"{F}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{F}/{name}.tif", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def attr_occlusion(det, X, gidx, base):
    """Cheaper alternative: score drop when one process is restored to the reference (4 evaluations)."""
    s = det.score(X)
    out = []
    for g in fp.GROUPS:
        Xr = X.copy(); Xr[:, gidx[g]] = base[gidx[g]]
        out.append(s - det.score(Xr))
    return np.stack(out, axis=1)


def event_hits(A_rows, rows_idx, alarm, ev, single_only=True):
    """A_rows: attributions for rows_idx (attack rows). Returns hit array over events (nan = missed)."""
    pos = {r: i for i, r in enumerate(rows_idx)}
    hits = []
    for _, e in ev.iterrows():
        if single_only and not e.single:
            continue
        idx = np.arange(e.start, e.end + 1)
        idx = idx[alarm[idx]]
        if len(idx) == 0:
            hits.append(np.nan); continue
        A = A_rows[[pos[i] for i in idx]].mean(axis=0)
        hits.append(float(fp.GROUPS[int(np.argmax(A))] in e.targets.split("+")))
    return np.array(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = a.results; F = f"{R}/figures"; os.makedirs(F, exist_ok=True)
    t0 = time.time()

    tr, te = fp.load(a.data)
    feats, gidx, Xfit, Xcal, Xte, base = fp.prepare(tr, te)
    ev = fp.events(te)
    evs = pd.read_csv(f"{R}/events.csv")
    ev["magnitude_z"] = evs.magnitude_z.values; ev["stealth_tier"] = evs.stealth_tier.values
    y = te["attack"].to_numpy().astype(int); files = te["file"].to_numpy()
    att_rows = np.flatnonzero(y == 1)

    # ------------------------------------------------ E5: PCA ablations
    det = fp.PCARecon(42).fit(Xfit)
    raw_te, raw_cal = det.score(Xte), det.score(Xcal)
    cal_sd = np.std(det.resid(Xcal), axis=0) + 1e-6
    Xa = Xte[att_rows]
    refs = {"training median (default)": base, "training mean": np.zeros_like(base),
            "calibration median": np.median(Xcal, axis=0).astype(np.float32)}
    A_gs = {k: fp.attr_group_shapley(det, Xa, gidx, b) for k, b in refs.items()}
    A_res = fp.attr_residual(det, Xa, gidx, cal_sd)
    A_occ = attr_occlusion(det, Xa, gidx, base)
    A_z = fp.attr_z(Xa, gidx)

    grid = []
    for sm in [10, 30, 60, 120]:
        fp.SMOOTH = sm
        s_te = fp.smooth(raw_te, files); s_cal = fp.smooth(raw_cal, np.zeros(len(raw_cal)))
        for q in [0.99, 0.995, 0.999, 0.9995]:
            alarm = s_te > np.quantile(s_cal, q)
            m, _ = fp.evaluate(alarm, y, files, ev)
            h_gs = event_hits(A_gs["training median (default)"], att_rows, alarm, ev)
            h_res = event_hits(A_res, att_rows, alarm, ev)
            grid.append(dict(smooth_s=sm, q=q, event_recall=m["event_recall"], f1=m["f1"],
                             false_alarms_per_hour=m["false_alarms_per_hour"], median_delay_s=m["median_delay_s"],
                             detected_single=int((~np.isnan(h_gs)).sum()),
                             gs_hit=float(np.nanmean(h_gs)), residual_hit=float(np.nanmean(h_res))))
    fp.SMOOTH = 60
    grid = pd.DataFrame(grid); grid.to_csv(f"{R}/E5_ablation_grid.csv", index=False)

    s_te = fp.smooth(raw_te, files); s_cal = fp.smooth(raw_cal, np.zeros(len(raw_cal)))
    alarm = s_te > np.quantile(s_cal, fp.THR_Q)
    rows = []
    for k, A in A_gs.items():
        h = event_hits(A, att_rows, alarm, ev)
        rows.append(dict(factor="Shapley reference value", setting=k, hit=float(np.nanmean(h)), detected=int((~np.isnan(h)).sum())))
    for k, A in [("Group Shapley, 16 evaluations (default)", A_gs["training median (default)"]),
                 ("Occlusion, 4 evaluations", A_occ), ("Residual", A_res), ("Z-mean", A_z)]:
        h = event_hits(A, att_rows, alarm, ev)
        rows.append(dict(factor="Attribution method", setting=k, hit=float(np.nanmean(h)), detected=int((~np.isnan(h)).sum())))
    for cf in [0.1, 0.2, 0.3]:
        fp.CAL_FRAC = cf
        _, gidx2, Xf2, Xc2, Xt2, base2 = fp.prepare(tr, te)
        d2 = fp.PCARecon(42).fit(Xf2)
        al2 = fp.smooth(d2.score(Xt2), files) > np.quantile(fp.smooth(d2.score(Xc2), np.zeros(len(Xc2))), fp.THR_Q)
        m2, _ = fp.evaluate(al2, y, files, ev)
        h2 = event_hits(fp.attr_group_shapley(d2, Xt2[att_rows], gidx2, base2), att_rows, al2, ev)
        rows.append(dict(factor="Calibration tail of last training file", setting=f"{int(cf*100)}%{' (default)' if cf == 0.2 else ''}",
                         hit=float(np.nanmean(h2)), detected=int((~np.isnan(h2)).sum()),
                         event_recall=m2["event_recall"], false_alarms_per_hour=m2["false_alarms_per_hour"]))
    fp.CAL_FRAC = 0.2
    abl = pd.DataFrame(rows); abl.to_csv(f"{R}/E5_ablation_factors.csv", index=False)

    # Figure 5: heatmaps over smoothing x threshold
    fig, axs = plt.subplots(1, 3, figsize=(W_IN, 2.6))
    sms, qs = [10, 30, 60, 120], [0.99, 0.995, 0.999, 0.9995]
    for ax, (col, title, fmt, cmap) in zip(axs, [("event_recall", "Event recall (%)", "{:.0f}", "Greys"),
                                                  ("false_alarms_per_hour", "False alarms per hour", "{:.2f}", "Greys"),
                                                  ("gs_hit", "Group Shapley localization (%)", "{:.0f}", "Blues")]):
        M = grid.pivot(index="smooth_s", columns="q", values=col).loc[sms, qs].to_numpy()
        V = M * 100 if col != "false_alarms_per_hour" else M
        ax.imshow(V, cmap=cmap, aspect="auto", vmin={"event_recall": 60, "gs_hit": 80}.get(col, 0),
                  vmax=(V.max() * 1.3 if col == "false_alarms_per_hour" else 100))
        for i in range(len(sms)):
            for j in range(len(qs)):
                v = V[i, j]; dark = (v > (V.max() * 0.75 if col == "false_alarms_per_hour" else (90 if col == "event_recall" else 92)))
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5, color="white" if dark else "black",
                        weight="bold" if (sms[i] == 60 and qs[j] == 0.999) else "normal")
        ax.set_xticks(range(4)); ax.set_xticklabels([str(q) for q in qs], fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels([f"{s} s" for s in sms], fontsize=7)
        ax.set_title(title, fontsize=8.5, loc="left"); ax.set_xlabel("Threshold quantile q", fontsize=7.5)
        ax.add_patch(plt.Rectangle((qs.index(0.999) - 0.5, sms.index(60) - 0.5), 1, 1, fill=False, ec=RED, lw=1.4))
    axs[0].set_ylabel("Smoothing window")
    save(fig, F, "Fig8_ablation")

    # ------------------------------------------------ E6: per-event map (from pipeline CSV)
    pe = pd.read_csv(f"{R}/E2_localization_per_event.csv")
    combos = [(d, m) for d in ["PCA", "MLP-AE", "IForest", "Z-max"] for m in ["Residual", "Z-mean", "GroupShapley"]
              if not (d == "IForest" and m == "Residual")]
    order = ev.sort_values("magnitude_z").event.tolist()
    S = np.zeros((len(order), len(combos)))
    for j, (d, m) in enumerate(combos):
        sub = pe[(pe.detector == d) & (pe.method == m)]
        for i, e in enumerate(order):
            h = sub[sub.event == e].hit.to_numpy()
            det_frac = np.mean(~np.isnan(h))
            S[i, j] = 0 if det_frac < 0.5 else (2 if np.nanmean(h) >= 0.5 else 1)
    pd.DataFrame(S, index=order, columns=[f"{d}|{m}" for d, m in combos]).to_csv(f"{R}/E6_event_map.csv")
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(W_IN, 6.2))
    ax.imshow(S, cmap=ListedColormap(["#f2f2f2", RED, BLUE]), aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([f"{d}\n{'G. Shapley' if m == 'GroupShapley' else m}" for d, m in combos], fontsize=6.5)
    lab = ev.set_index("event").loc[order]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{t:<6} {z:6.1f}" for t, z in zip(lab.targets, lab.magnitude_z)], fontsize=5.5, family="monospace")
    for b in [17, 33]:
        ax.axhline(b - 0.5, color=DARK, lw=0.8)
    for j in [2.5, 5.5, 7.5]:
        ax.axvline(j, color="white", lw=2)
    ax.text(len(combos) - 0.4, 8, "high\nstealth", fontsize=7, va="center"); ax.text(len(combos) - 0.4, 25, "medium", fontsize=7, va="center")
    ax.text(len(combos) - 0.4, 41.5, "low\nstealth", fontsize=7, va="center")
    ax.set_ylabel("Attack event: target process and deviation (z), sorted by deviation", fontsize=7.5)
    ax.legend(handles=[Patch(color=BLUE, label="Detected, correct process"), Patch(color=RED, label="Detected, wrong process"),
                       Patch(color="#f2f2f2", label="Not detected")], frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    save(fig, F, "Fig9_event_map")

    # ------------------------------------------------ E7: case studies (PCA)
    pca_pe = pe[(pe.detector == "PCA")].pivot_table(index="event", columns="method", values="hit").reindex(ev.event.values)
    cand = pca_pe[(pca_pe.GroupShapley == 1) & (pca_pe.Residual == 0)].index.tolist()
    hs = [e for e in ev[(ev.stealth_tier == "high stealth") & ev.single].event if pca_pe.loc[e, "GroupShapley"] == 1] if len(pca_pe) else []
    picks = []
    for e in cand:
        t = ev.loc[e, "targets"]
        if t not in [ev.loc[p, "targets"] for p in picks]:
            picks.append(e)
        if len(picks) == 2:
            break
    if hs:
        picks.append(hs[0])
    picks = picks[:3]
    thr = float(np.quantile(s_cal, fp.THR_Q))
    fig, axs = plt.subplots(3, len(picks), figsize=(W_IN, 5.4), sharex="col", squeeze=False,
                            gridspec_kw={"height_ratios": [1.2, 1, 1]})
    for jj in range(1, len(picks)):
        axs[0, jj].sharey(axs[0, 0]); axs[0, jj].tick_params(labelleft=False)
    case_rows = []
    for j, e in enumerate(picks):
        E = ev.loc[e]; lo = max(int(E.start) - 300, 0); hi = min(int(E.end) + 300, len(y) - 1)
        lo = max(lo, int(np.flatnonzero(files == files[E.start])[0])); hi = min(hi, int(np.flatnonzero(files == files[E.start])[-1]))
        idx = np.arange(lo, hi + 1, 5); t = (idx - E.start)
        ax = axs[0, j]
        ax.plot(t, s_te[idx], color=DARK, lw=0.9); ax.axhline(thr, color=RED, lw=0.8, ls="--")
        ax.axvspan(0, E.end - E.start, color=PROC_COL.get(E.targets.split("+")[0], GREY), alpha=0.15)
        ax.set_yscale("log"); ax.set_title(f"Event {e}: {E.targets} attacked, {E.magnitude_z:.1f} z", fontsize=7.5, loc="left")
        for r, (name, fn) in enumerate([("Group Shapley", lambda X: fp.attr_group_shapley(det, X, gidx, base)),
                                         ("Residual", lambda X: fp.attr_residual(det, X, gidx, cal_sd))]):
            A = np.clip(fn(Xte[idx]), 0, None); share = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
            share = pd.DataFrame(share).rolling(6, min_periods=1).mean().to_numpy()
            ax2 = axs[r + 1, j]
            ax2.stackplot(t, share.T, colors=[PROC_COL[g] for g in fp.GROUPS], alpha=0.9, labels=fp.GROUPS)
            ax2.axvline(0, color="black", lw=0.6); ax2.axvline(E.end - E.start, color="black", lw=0.6)
            ax2.set_ylim(0, 1); ax2.set_yticks([0, 0.5, 1])
            if j == 0:
                ax2.set_ylabel(f"{name}\nattribution share", fontsize=7.5)
            ins = (t >= 0) & (t <= E.end - E.start)
            case_rows.append(dict(event=int(e), targets=E.targets, method=name,
                                  **{f"share_{g}": float(share[ins, k].mean()) for k, g in enumerate(fp.GROUPS)}))
        axs[2, j].set_xlabel("Seconds from attack start", fontsize=7.5)
    axs[0, 0].set_ylabel("PCA anomaly score\n(log, 60-s mean)", fontsize=7.5)
    h_, l_ = axs[1, 0].get_legend_handles_labels()
    fig.legend(h_, [f"{g} " + {"P1": "boiler", "P2": "turbine", "P3": "water treatment", "P4": "HIL"}[g] for g in l_],
               loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.12)
    pd.DataFrame(case_rows).to_csv(f"{R}/E7_case_studies.csv", index=False)
    save(fig, F, "Fig10_case_studies")

    # ------------------------------------------------ E8: computational cost
    cost = []
    Xs = Xte[:20000]; Xa1 = Xa[:1000]
    for D in [fp.PCARecon, fp.MLPAE, fp.IForest, fp.ZMax]:
        t1 = time.perf_counter(); d = D(42).fit(Xfit); fit_t = time.perf_counter() - t1
        t1 = time.perf_counter(); d.score(Xs); sc = (time.perf_counter() - t1) / len(Xs) * 1e3
        rowd = dict(detector=D.name, fit_s=fit_t, score_ms_per_1k=sc * 1000)  # ms per 1,000 samples
        t1 = time.perf_counter(); fp.attr_group_shapley(d, Xa1, gidx, base); rowd["group_shapley_ms_per_1k"] = (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter(); attr_occlusion(d, Xa1, gidx, base); rowd["occlusion_ms_per_1k"] = (time.perf_counter() - t1) * 1000
        if d.resid(Xa1[:2]) is not None:
            sd_ = np.std(d.resid(Xcal[:5000]), axis=0) + 1e-6
            t1 = time.perf_counter(); fp.attr_residual(d, Xa1, gidx, sd_); rowd["residual_ms_per_1k"] = (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter(); fp.attr_z(Xa1, gidx); rowd["zmean_ms_per_1k"] = (time.perf_counter() - t1) * 1000
        cost.append(rowd)
    pd.DataFrame(cost).to_csv(f"{R}/E8_cost.csv", index=False)

    # ------------------------------------------------ Figure 9: deployment architecture
    fig, ax = plt.subplots(figsize=(W_IN, 3.7)); ax.set_xlim(0, 760); ax.set_ylim(0, 370); ax.axis("off")
    def box(x, y, w, h, title, lines, fc="white", ec=DARK, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, 370 - y - h), w, h, boxstyle="round,pad=0,rounding_size=7", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, 370 - y - 16, title, ha="center", va="center", fontsize=7.5, weight="bold")
        for k, l in enumerate(lines):
            ax.text(x + w / 2, 370 - y - 31 - 12 * k, l, ha="center", va="center", fontsize=6.3, color="#333333")
    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, 370 - y2), xytext=(x1, 370 - y1), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.9))
    ax.add_patch(FancyBboxPatch((8, 370 - 26 - 104), 744, 104, boxstyle="round,pad=0,rounding_size=8", fc="#f5f5f2", ec="none"))
    ax.text(16, 370 - 38, "OFFLINE  (normal operation only)", fontsize=6.8, color="#666666", weight="bold")
    W4, G4 = 168, 18
    xs4 = [22 + k * (W4 + G4) for k in range(4)]
    box(xs4[0], 48, W4, 66, "Historian data", ["1-Hz sensor and actuator", "tags (e.g. OPC, Modbus)"])
    box(xs4[1], 48, W4, 66, "Fit detector", ["standardize; any model:", "PCA, AE, IForest, Z-max"])
    box(xs4[2], 48, W4, 66, "Calibrate", ["threshold q on a held-out", "normal tail"])
    box(xs4[3], 48, W4, 66, "Process map", ["sensor -> process group", "(P1 ... P4)"])
    for k in range(3):
        arr(xs4[k] + W4, 81, xs4[k + 1], 81)
    ax.text(16, 370 - 152, "ONLINE  (every second)", fontsize=6.8, color="#666666", weight="bold")
    W5, G5 = 132, 15
    xs5 = [22 + k * (W5 + G5) for k in range(5)]
    box(xs5[0], 165, W5, 70, "Live tags", ["buffered and", "standardized"])
    box(xs5[1], 165, W5, 70, "Detector score", ["causal 60-s mean,", "compared with q"])
    box(xs5[2], 165, W5, 70, "Group Shapley", ["16 evaluations;", "top = location"], fc="#e6eefb", ec=BLUE, lw=1.5)
    box(xs5[3], 165, W5, 70, "Trust gate", ["alert after 10", "stable seconds"], fc="#e6eefb", ec=BLUE, lw=1.5)
    box(xs5[4], 165, W5, 70, "Operator alert", ["process, attribution", "share, stability"])
    for k in range(4):
        arr(xs5[k] + W5, 200, xs5[k + 1], 200)
    ax.text(xs5[1] + W5 / 2, 370 - 250, "below threshold: no alert", fontsize=6.2, color="#666666", ha="center")
    ax.text(xs5[3] + W5 / 2, 370 - 250, "unstable: hold and re-check", fontsize=6.2, color="#666666", ha="center")
    arr(xs4[1] + W4 / 2, 114, xs5[1] + W5 / 2, 165); arr(xs4[2] + W4 / 2, 114, xs5[1] + W5 * 0.8, 165)
    arr(xs4[3] + W4 / 2, 114, xs5[2] + W5 / 2, 165)
    box(190, 278, 380, 62, "Response (outside this study)", ["inspect or isolate the named process;", "log the alert, attribution and stability"])
    arr(xs5[4] + W5 / 2, 235, 570, 300)
    save(fig, F, "Fig12_architecture")

    json.dump(dict(runtime_s=time.time() - t0, case_events=[int(p) for p in picks]), open(f"{R}/extras_config.json", "w"), indent=2)
    print("extras ok", round(time.time() - t0), "s; case events", picks)


if __name__ == "__main__":
    main()
