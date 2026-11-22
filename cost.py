"""Table 9: computational cost of every detector and attribution, all measured on one machine.
Times are per 1,000 samples; Kernel SHAP uses 1,024 coalitions per explained sample.

Usage:
    python cost.py --data hai/hai-21.03 --results results
"""
import argparse, os, sys, time, platform
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fdi_pipeline as fp
import extend as ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--results", default="results")
    a = ap.parse_args()
    tr, te = fp.load(a.data)
    feats, gidx, Xfit, Xcal, Xte, base = fp.prepare(tr, te)
    n1, n2 = len(tr[0]), len(tr[1]); del tr
    y = te["attack"].to_numpy(); files = te["file"].to_numpy(); d = Xfit.shape[1]
    att = np.flatnonzero(y == 1)[::4][:1000]
    te_starts = [int(np.flatnonzero(files == f)[0]) for f in np.unique(files)]
    pool = np.sort(np.random.default_rng(0).choice(len(Xfit), size=200_000, replace=False))
    Lfit = ex.lagged(Xfit, [0, n1, n1 + n2], rows=pool); Lcal = ex.lagged(Xcal, [0])
    La = ex.lagged(Xte, te_starts, rows=att); Ls = ex.lagged(Xte, te_starts, rows=np.arange(20000))
    gidx_L = {g: np.concatenate([gidx[g] + k * d for k in range(ex.WIN)]) for g in fp.GROUPS}
    base_L = np.tile(base, ex.WIN).astype(np.float32)
    rows = []
    for D in [fp.PCARecon, fp.MLPAE, fp.IForest, fp.ZMax, ex.USAD]:
        lag = D is ex.USAD
        Xf, Xs, Xa, G, B = (Lfit, Ls, La, gidx_L, base_L) if lag else (Xfit, Xte[:20000], Xte[att], gidx, base)
        players = [np.array([j + k * d for k in range(ex.WIN)]) for j in range(d)] if lag else [np.array([j]) for j in range(d)]
        t = time.perf_counter(); det = D(42).fit(Xf); fit_s = time.perf_counter() - t
        t = time.perf_counter(); det.score(Xs); score = (time.perf_counter() - t) / len(Xs) * 1e6
        r = dict(detector=D.name, fit_s=fit_s, score_ms_per_1k=score)
        t = time.perf_counter(); fp.attr_group_shapley(det, Xa, G, B); r["group_shapley_ms_per_1k"] = (time.perf_counter() - t) * 1000
        occ = lambda X: np.stack([det.score(X) - det.score(np.where(np.isin(np.arange(X.shape[1]), G[g])[None, :], B[None, :], X))
                                  for g in fp.GROUPS], axis=1)
        t = time.perf_counter(); occ(Xa); r["occlusion_ms_per_1k"] = (time.perf_counter() - t) * 1000
        cal_r = det.resid(Xcal[:5000] if not lag else Lcal[:5000])
        if cal_r is not None:
            sd = np.std(cal_r, axis=0) + 1e-6
            t = time.perf_counter(); fp.attr_residual(det, Xa, G, sd); r["residual_ms_per_1k"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); fp.attr_z(Xa[:, :d], gidx); r["zmean_ms_per_1k"] = (time.perf_counter() - t) * 1000
        sub = Xa[:100]                                  # Kernel SHAP is timed on 100 samples and scaled to 1,000
        t = time.perf_counter(); ex.kernel_shap(det.score, sub, players, B); r["kernel_shap_ms_per_1k"] = (time.perf_counter() - t) * 1000 * 10
        rows.append(r); print(r, flush=True)
    out = pd.DataFrame(rows); out["cpu"] = platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": ")
    out.to_csv(f"{a.results}/E12_cost.csv", index=False)


if __name__ == "__main__":
    main()
