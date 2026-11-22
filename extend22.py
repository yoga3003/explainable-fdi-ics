"""
HAI 22.04 versions of the two additions (run after hai22_pipeline.py and with extend.py next to it):
  H10  USAD through the same detection and multi-level localization protocol (rows detector = "USAD").
  H11  Per-sensor Kernel SHAP summed to processes and control loops vs exact group Shapley,
       on the same sampled attack seconds of every event (oracle setting), all detectors and seeds.

Usage:
    python extend22.py --data hai/hai-22.04 --gt hai22_ground_truth.csv --out results22
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fdi_pipeline as fp
import hai22_pipeline as hp
import extend as ex

KS_PER_EVENT = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="results22"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ks_only", action="store_true", help="skip H10; retrain USAD only for Kernel SHAP")
    ap.add_argument("--iforest_seeds", type=int, default=1, help="IForest seeds for Kernel SHAP (costly)")
    a = ap.parse_args(); t0 = time.time(); seeds = hp.SEEDS[:a.seeds]

    gt = pd.read_csv(a.gt)
    tr, te = hp.load(a.data)
    feats, Xfit, Xcal, Xte, base, n_train, n_fit, n_cal = hp.prepare(tr, te)
    lens = [len(t) for t in tr[:-1]]; fit_starts = list(np.cumsum([0] + lens))
    del tr
    pidx, lidx = hp.groups_for(feats)
    P_names, L_names = list(pidx), list(lidx)
    Pg, Lg = [pidx[p] for p in P_names], [lidx[L] for L in L_names]
    ev = hp.events(te, gt)
    y = te["Attack"].to_numpy().astype(int); files = te["file"].to_numpy()
    d = Xfit.shape[1]; W = ex.WIN
    rows = np.concatenate([np.arange(e.start, e.end + 1)[::hp.STRIDE] for _, e in ev.iterrows()])
    Xa = Xte[rows]; Z = np.abs(Xa)
    valid_pts = lambda e: [feats.index(p) for p in e.points.split(";") if p in feats]

    # ---------------------------------------------------------------- H10 USAD
    te_starts = [int(np.flatnonzero(files == f)[0]) for f in np.unique(files)]
    pool = np.sort(np.random.default_rng(0).choice(len(Xfit), size=200_000, replace=False))
    Lfit, Lcal, Lte = ex.lagged(Xfit, fit_starts, rows=pool), ex.lagged(Xcal, [0]), ex.lagged(Xte, te_starts)
    lift = lambda g: np.concatenate([g + k * d for k in range(W)])
    PgL, LgL = [lift(g) for g in Pg], [lift(g) for g in Lg]
    baseL = np.tile(base, W).astype(np.float32); La = Lte[rows]
    det_rows, per_ev, usad = [], [], {}
    for seed in seeds:
        det = ex.USAD(seed).fit(Lfit)
        if a.ks_only:
            usad[seed] = det; print(f"[{time.time()-t0:6.0f}s] USAD seed {seed} trained", flush=True); continue
        s_cal = fp.smooth(det.score(Lcal), np.zeros(len(Lcal)))
        s = fp.smooth(det.score(Lte), files); alarm = s > np.quantile(s_cal, fp.THR_Q)
        m, _ = fp.evaluate(alarm, y, files, ev.assign(targets=""), s)
        det_rows.append(dict(detector="USAD", seed=seed, **m)); usad[seed] = det
        cal_sd = np.std(det.resid(Lcal), axis=0) + 1e-6
        RresL = (det.resid(La) / cal_sd) ** 2
        Rres = RresL.reshape(len(La), W, d).mean(axis=1)          # per-sensor residual, averaged over the window
        A = {("process", "GroupShapley"): hp.shapley(det, La, PgL, baseL), ("process", "Z-mean"): hp.group_mean(Z, Pg),
             ("loop", "GroupShapley"): hp.shapley(det, La, LgL, baseL), ("loop", "Occlusion"): hp.occlusion(det, La, LgL, baseL),
             ("loop", "Z-mean"): hp.group_mean(Z, Lg),
             ("process", "Residual"): hp.group_mean(Rres, Pg), ("loop", "Residual"): hp.group_mean(Rres, Lg)}
        for e in ev.itertuples():
            pos = np.flatnonzero((rows >= e.start) & (rows <= e.end)); al = pos[alarm[rows[pos]]]
            for (lvl, meth), M in A.items():
                names = P_names if lvl == "process" else L_names
                targets = e.procs.split(";") if lvl == "process" else e.loops.split(";")
                top_or = names[int(np.argmax(M[pos].mean(axis=0)))]
                top = names[int(np.argmax(M[al].mean(axis=0)))] if len(al) else None
                per_ev.append(dict(detector="USAD", seed=seed, level=lvl, method=meth, id=e.id, detected=int(len(al) > 0),
                                   top=top, hit=(float(top in targets) if top else np.nan), top_oracle=top_or,
                                   hit_oracle=float(top_or in targets)))
            vp = valid_pts(e)
            if vp and not e.ambiguous:
                for meth, M in {"Z-score": Z, "Residual": Rres}.items():
                    for src, P in [("alarm", al), ("oracle", pos)]:
                        if len(P) == 0:
                            continue
                        order = np.argsort(-M[P].mean(axis=0))
                        per_ev.append(dict(detector="USAD", seed=seed, level=f"point-{src}", method=meth, id=e.id,
                                           detected=int(len(al) > 0), top=feats[order[0]], hit=float(order[0] in vp),
                                           hit3=float(bool(set(order[:3]) & set(vp))), hit5=float(bool(set(order[:5]) & set(vp)))))
                for src, P in [("alarm", al), ("oracle", pos)]:
                    if len(P) == 0:
                        continue
                    Lt = L_names[int(np.argmax(A[("loop", "GroupShapley")][P].mean(axis=0)))]
                    inside = lidx[Lt]; order = inside[np.argsort(-Z[P][:, inside].mean(axis=0))]
                    per_ev.append(dict(detector="USAD", seed=seed, level=f"point-{src}", method="Loop Shapley + Z-score",
                                       id=e.id, detected=int(len(al) > 0), top=feats[order[0]], hit=float(order[0] in vp),
                                       hit3=float(bool(set(order[:3]) & set(vp))), hit5=float(bool(set(order[:5]) & set(vp)))))
        print(f"[{time.time()-t0:6.0f}s] USAD seed {seed}: F1={m['f1']:.3f} event recall={m['event_recall']:.2f} "
              f"FA/h={m['false_alarms_per_hour']:.3f}", flush=True)
    if not a.ks_only:
        pd.DataFrame(det_rows).to_csv(f"{a.out}/H10_detection.csv", index=False)
        pd.DataFrame(per_ev).to_csv(f"{a.out}/H10_localization_per_event.csv", index=False)

    # ------------------------------------------------------------ H11 Kernel SHAP
    proc_of = np.array([P_names.index(next(p for p in P_names if j in pidx[p])) for j in range(d)])
    loop_of = np.full(d, -1)
    for li, g in enumerate(Lg):
        loop_of[g] = li
    out = []
    dets = [(fp.PCARecon, seeds[:1]), (fp.MLPAE, seeds), (fp.ZMax, seeds[:1]), ("USAD", seeds), (fp.IForest, seeds[:a.iforest_seeds])]
    for D, sl in dets:
        for seed in sl:
            if D == "USAD":
                det = usad[seed]; XT, B, P_, L_, players = Lte, baseL, PgL, LgL, [np.array([j + k * d for k in range(W)]) for j in range(d)]
                name = "USAD"
            else:
                det = D(seed).fit(Xfit); XT, B, P_, L_, players = Xte, base, Pg, Lg, [np.array([j]) for j in range(d)]
                name = D.name
            for e in ev.itertuples():
                sel = ex.sample_seconds(np.arange(e.start, e.end + 1), KS_PER_EVENT)
                phi = ex.kernel_shap(det.score, XT[sel], players, B, seed=seed)
                gsP = hp.shapley(det, XT[sel], P_, B); gsL = hp.shapley(det, XT[sel], L_, B)
                ksP = np.stack([phi[:, proc_of == i].sum(axis=1) for i in range(len(P_names))], axis=1)
                ksL = np.stack([phi[:, loop_of == i].sum(axis=1) for i in range(len(L_names))], axis=1)
                for lvl, names, targets, pairs in [("process", P_names, e.procs.split(";"), [("KernelSHAP", ksP), ("GroupShapley (same seconds)", gsP)]),
                                                   ("loop", L_names, e.loops.split(";"), [("KernelSHAP", ksL), ("GroupShapley (same seconds)", gsL)])]:
                    for meth, M in pairs:
                        top = names[int(np.argmax(M.mean(axis=0)))]
                        out.append(dict(detector=name, seed=seed, level=lvl, method=meth, id=e.id, top_oracle=top,
                                        hit_oracle=float(top in targets), seconds_explained=len(sel)))
            print(f"[{time.time()-t0:6.0f}s] Kernel SHAP 22.04 {name} seed {seed}", flush=True)
            pd.DataFrame(out).to_csv(f"{a.out}/H11_kernelshap_per_event.csv", index=False)
    pd.DataFrame(out).to_csv(f"{a.out}/H11_kernelshap_per_event.csv", index=False)
    json.dump(dict(runtime_s=time.time() - t0, kernel_shap_seconds_per_event=KS_PER_EVENT,
                   kernel_shap_coalitions=ex.KS_SAMPLES, usad_window_s=W), open(f"{a.out}/extend22.json", "w"), indent=2)
    print("extend22 ok", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
