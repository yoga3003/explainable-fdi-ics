"""
HAI 22.04: detection and multi-level localization (process -> control loop -> data point)
against the attack schedule of the HAI technical details document (hai22_ground_truth.csv).

Usage:
    python hai22_pipeline.py --data /path/to/hai-22.04 --gt hai22_ground_truth.csv --out results22
"""
import argparse, glob, json, os, sys, time
from math import factorial
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fdi_pipeline as fp          # detectors, smoothing, evaluation, bootstrap (same code as HAI 21.03)

SEEDS = [42, 123, 2026, 7, 99]
STRIDE = 3                          # attribute every 3rd attack second (attributions are averaged per event)
PROCS = ["P1", "P2", "P3", "P4"]
# control loops documented in the HAI technical details (boiler, turbine, water treatment); rest grouped
LOOPS = {
    "P1-PC": ["P1_B2016", "P1_PIT01", "P1_PIT01_HH", "P1_PIT02", "P1_PCV01D", "P1_PCV01Z", "P1_PCV02D", "P1_PCV02Z"],
    "P1-LC": ["P1_B3004", "P1_LIT01", "P1_LCV01D", "P1_LCV01Z"],
    "P1-FC": ["P1_B3005", "P1_FT03", "P1_FT03Z", "P1_FCV03D", "P1_FCV03Z"],
    "P1-TC": ["P1_B4022", "P1_B4002", "P1_TIT01", "P1_FCV01D", "P1_FCV01Z", "P1_FCV02D", "P1_FCV02Z", "P1_FT02", "P1_FT02Z"],
    "P1-CC": ["P1_PP04", "P1_PP04SP", "P1_TIT03"],
    "P2-SC": ["P2_AutoSD", "P2_ManualSD", "P2_SCO", "P2_SIT01", "P2_SCST", "P2_AutoGO", "P2_ManualGO"],
    "P2-TR": ["P2_RTR", "P2_VTR01", "P2_VTR02", "P2_VTR03", "P2_VTR04", "P2_VIBTR01", "P2_VIBTR02",
              "P2_VIBTR03", "P2_VIBTR04", "P2_Emerg", "P2_TripEx", "P2_VT01"],
}
LOOP_ORDER = ["P1-PC", "P1-LC", "P1-FC", "P1-TC", "P1-CC", "P1-other", "P2-SC", "P2-TR", "P2-other", "P3-LC", "P4-HIL"]


def load(d):
    tr = [pd.read_csv(f) for f in sorted(glob.glob(f"{d}/train*.csv"))]
    te = []
    for i, f in enumerate(sorted(glob.glob(f"{d}/test*.csv"))):
        x = pd.read_csv(f); x["file"] = i; x["fname"] = os.path.basename(f)[:-4]; te.append(x)
    return tr, pd.concat(te, ignore_index=True)


def prepare(tr, te, cal_frac=0.2):
    sens = [c for c in tr[0].columns if c[:2] in PROCS]
    trall = pd.concat(tr, ignore_index=True)
    feats = [c for c in sens if trall[c].nunique(dropna=True) > 1]
    n = len(tr[-1]); cut = int(n * (1 - cal_frac))
    fit_df = pd.concat(tr[:-1] + [tr[-1].iloc[:cut]], ignore_index=True); cal_df = tr[-1].iloc[cut:]
    mu, sig = fit_df[feats].mean(), fit_df[feats].std(); sig = sig.where(sig > 1e-8, 1.0)
    z = lambda d: ((d[feats] - mu) / sig).to_numpy(np.float32)
    Xfit = z(fit_df)
    return feats, Xfit, z(cal_df), z(te), np.median(Xfit, axis=0).astype(np.float32), len(trall), len(fit_df), len(cal_df)


def groups_for(feats):
    pidx = {p: np.array([i for i, c in enumerate(feats) if c.startswith(p)]) for p in PROCS}
    loop_of = {}
    for L, cols in LOOPS.items():
        for c in cols:
            loop_of[c] = L
    lidx = {}
    for L in LOOP_ORDER:
        if L == "P1-other":
            m = [i for i, c in enumerate(feats) if c.startswith("P1") and c not in loop_of]
        elif L == "P2-other":
            m = [i for i, c in enumerate(feats) if c.startswith("P2") and c not in loop_of]
        elif L == "P3-LC":
            m = [i for i, c in enumerate(feats) if c.startswith("P3")]
        elif L == "P4-HIL":
            m = [i for i, c in enumerate(feats) if c.startswith("P4")]
        else:
            m = [i for i, c in enumerate(feats) if loop_of.get(c) == L]
        if m:
            lidx[L] = np.array(m)
    return pidx, lidx


def events(te, gt):
    a = te["Attack"].to_numpy()
    out = []
    for i, f in enumerate(sorted(te.fname.unique())):
        idx = np.flatnonzero(te.fname.to_numpy() == f)
        aa = a[idx]
        s = np.flatnonzero(np.diff(np.r_[0, aa]) == 1); e = np.flatnonzero(np.diff(np.r_[aa, 0]) == -1)
        g = gt[gt.file == f].sort_values("order")
        assert len(g) == len(s), f"{f}: {len(s)} events in CSV vs {len(g)} in schedule"
        for (si, ei), (_, r) in zip(zip(s, e), g.iterrows()):
            loops = r.loops.split(";")
            out.append(dict(id=r.id, start=int(idx[si]), end=int(idx[ei]), dur=int(ei - si + 1),
                            start_time=te.timestamp.iloc[idx[si]], loops=r.loops, points=r.points,
                            procs=";".join(sorted({l[:2] for l in loops})), ambiguous=int(r.ambiguous),
                            single_loop=int(len(loops) == 1), single_proc=int(len({l[:2] for l in loops}) == 1)))
    return pd.DataFrame(out)


def shapley(det, X, groups, base):
    """Exact Shapley values of det.score over a list of feature groups (2^G coalitions)."""
    G, n = len(groups), len(X)
    v = np.empty((2 ** G, n))
    for mask in range(2 ** G):
        Xs = np.repeat(base[None, :], n, axis=0)
        for g in range(G):
            if mask >> g & 1:
                Xs[:, groups[g]] = X[:, groups[g]]
        v[mask] = det.score(Xs)
    w = [factorial(k) * factorial(G - k - 1) / factorial(G) for k in range(G)]
    pc = np.array([bin(m).count("1") for m in range(2 ** G)])
    phi = np.zeros((n, G))
    for g in range(G):
        for mask in range(2 ** G):
            if not mask >> g & 1:
                phi[:, g] += w[pc[mask]] * (v[mask | (1 << g)] - v[mask])
    return phi


def occlusion(det, X, groups, base):
    s = det.score(X); out = []
    for gi in groups:
        Xr = X.copy(); Xr[:, gi] = base[gi]; out.append(s - det.score(Xr))
    return np.stack(out, axis=1)


def group_mean(M, groups):
    return np.stack([M[:, gi].mean(axis=1) for gi in groups], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="results22"); ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()

    gt = pd.read_csv(a.gt)
    tr, te = load(a.data)
    feats, Xfit, Xcal, Xte, base, n_train, n_fit, n_cal = prepare(tr, te)
    pidx, lidx = groups_for(feats)
    P_names, L_names = list(pidx), list(lidx)
    Pg, Lg = [pidx[p] for p in P_names], [lidx[L] for L in L_names]
    ev = events(te, gt)
    y = te["Attack"].to_numpy().astype(int); files = te["file"].to_numpy()

    # stealth magnitude: 95th pct of max |z| over the target loops' sensors
    mag = []
    for _, e in ev.iterrows():
        cols = np.concatenate([lidx[L] for L in e.loops.split(";") if L in lidx])
        mag.append(float(np.percentile(np.abs(Xte[e.start:e.end + 1][:, cols]).max(axis=1), 95)))
    ev["magnitude_z"] = mag
    ev["stealth_tier"] = pd.qcut(ev.magnitude_z, 3, labels=["high stealth", "medium", "low stealth"]).astype(str)
    ev.to_csv(f"{a.out}/events22.csv", index=False)

    valid_pts = lambda e: [feats.index(p) for p in e.points.split(";") if p in feats]
    sl = ev[(ev.single_loop == 1) & (ev.ambiguous == 0)]
    sp = ev[(ev.single_proc == 1)]
    audit = dict(features_used=len(feats), dropped_constant=86 - len(feats), train_rows=n_train, fit_rows=n_fit,
                 cal_rows=n_cal, test_rows=len(Xte), attack_rows=int(y.sum()), events=len(ev),
                 ambiguous=int(ev.ambiguous.sum()), single_loop_events=len(sl), single_process_events=len(sp),
                 loop_groups={L: int(len(lidx[L])) for L in L_names}, process_groups={p: int(len(pidx[p])) for p in P_names},
                 majority_loop=sl.loops.value_counts().index[0], majority_loop_rate=float(sl.loops.value_counts().iloc[0] / len(sl)),
                 majority_process=sp.procs.value_counts().index[0], majority_process_rate=float(sp.procs.value_counts().iloc[0] / len(sp)),
                 target_points_missing=sorted({p for e in ev.itertuples() for p in e.points.split(";") if p not in feats}))
    json.dump(audit, open(f"{a.out}/audit22.json", "w"), indent=2)
    print(json.dumps(audit, indent=1), flush=True)

    # attack seconds used for attribution (every STRIDE-th second of each event)
    rows = np.concatenate([np.arange(e.start, e.end + 1)[::STRIDE] for _, e in ev.iterrows()])
    Xa = Xte[rows]

    det_rows, per_ev = [], []
    for D in [fp.PCARecon, fp.MLPAE, fp.IForest, fp.ZMax]:
        seeds = SEEDS[:a.seeds] if D in (fp.MLPAE, fp.IForest) else SEEDS[:1]
        for seed in seeds:
            det = D(seed).fit(Xfit)
            s_cal = fp.smooth(det.score(Xcal), np.zeros(len(Xcal)))
            s = fp.smooth(det.score(Xte), files); alarm = s > np.quantile(s_cal, fp.THR_Q)
            evq = ev.rename(columns={})
            m, detected = fp.evaluate(alarm, y, files, ev.assign(targets=""), s)
            det_rows.append(dict(detector=D.name, seed=seed, **m))

            r = det.resid(Xcal); cal_sd = (np.std(r, axis=0) + 1e-6) if r is not None else None
            Rres = (det.resid(Xa) / cal_sd) ** 2 if cal_sd is not None else None
            Z = np.abs(Xa)
            A = {("process", "GroupShapley"): shapley(det, Xa, Pg, base),
                 ("process", "Z-mean"): group_mean(Z, Pg),
                 ("loop", "GroupShapley"): shapley(det, Xa, Lg, base),
                 ("loop", "Occlusion"): occlusion(det, Xa, Lg, base),
                 ("loop", "Z-mean"): group_mean(Z, Lg)}
            if Rres is not None:
                A[("process", "Residual")] = group_mean(Rres, Pg); A[("loop", "Residual")] = group_mean(Rres, Lg)
            for k, e in enumerate(ev.itertuples()):
                pos = np.flatnonzero((rows >= e.start) & (rows <= e.end))
                al = pos[alarm[rows[pos]]]
                for (lvl, meth), M in A.items():
                    names = P_names if lvl == "process" else L_names
                    targets = e.procs.split(";") if lvl == "process" else e.loops.split(";")
                    top_or = names[int(np.argmax(M[pos].mean(axis=0)))]
                    top = names[int(np.argmax(M[al].mean(axis=0)))] if len(al) else None
                    per_ev.append(dict(detector=D.name, seed=seed, level=lvl, method=meth, id=e.id,
                                       detected=int(len(al) > 0), top=top, hit=(float(top in targets) if top else np.nan),
                                       top_oracle=top_or, hit_oracle=float(top_or in targets)))
                # data-point level: does a target point appear in the top-k ranked points?
                vp = valid_pts(e)
                if vp and not e.ambiguous:
                    rank_src = {"Z-score": Z}
                    if Rres is not None:
                        rank_src["Residual"] = Rres
                    for meth, M in rank_src.items():
                        for src, P in [("alarm", al), ("oracle", pos)]:
                            if len(P) == 0:
                                continue
                            order = np.argsort(-M[P].mean(axis=0))
                            per_ev.append(dict(detector=D.name, seed=seed, level=f"point-{src}", method=meth, id=e.id,
                                               detected=int(len(al) > 0), top=feats[order[0]],
                                               hit=float(order[0] in vp), hit3=float(bool(set(order[:3]) & set(vp))),
                                               hit5=float(bool(set(order[:5]) & set(vp)))))
                    # hierarchical: loop by group Shapley, then point by z-score inside that loop
                    for src, P in [("alarm", al), ("oracle", pos)]:
                        if len(P) == 0:
                            continue
                        Lt = L_names[int(np.argmax(A[("loop", "GroupShapley")][P].mean(axis=0)))]
                        inside = lidx[Lt]; zz = Z[P][:, inside].mean(axis=0); order = inside[np.argsort(-zz)]
                        per_ev.append(dict(detector=D.name, seed=seed, level=f"point-{src}", method="Loop Shapley + Z-score",
                                           id=e.id, detected=int(len(al) > 0), top=feats[order[0]], hit=float(order[0] in vp),
                                           hit3=float(bool(set(order[:3]) & set(vp))), hit5=float(bool(set(order[:5]) & set(vp)))))
            print(f"[{time.time()-t0:6.0f}s] {D.name} seed {seed}: F1={m['f1']:.3f} event recall={m['event_recall']:.2f} "
                  f"FA/h={m['false_alarms_per_hour']:.3f}", flush=True)
    pd.DataFrame(det_rows).to_csv(f"{a.out}/H1_detection.csv", index=False)
    pd.DataFrame(per_ev).to_csv(f"{a.out}/H2_localization_per_event.csv", index=False)
    json.dump(dict(runtime_s=time.time() - t0, stride=STRIDE, seeds=SEEDS[:a.seeds], loops=L_names), open(f"{a.out}/run22.json", "w"), indent=2)


if __name__ == "__main__":
    main()
