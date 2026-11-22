"""
Explainable FDI detection, process-level localization, stealth breakdown and
trust-gated alarms on HAI 21.03.

Usage:
    python fdi_pipeline.py --data /path/to/hai/hai-21.03 --out results

Everything reported in the paper is written by this script to --out
(CSV/JSON tables + PNG figures). Nothing is typed in by hand.
"""
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import average_precision_score

GROUPS = ["P1", "P2", "P3", "P4"]
SEEDS = [42, 123, 2026, 7, 99]
SMOOTH = 60            # trailing moving-average window for the anomaly score (s)
GRACE = 60             # seconds after an event end in which an alarm still counts
CAL_FRAC = 0.20        # chronological tail of train3 held out for threshold calibration
THR_Q = 0.999          # threshold = this quantile of calibration scores
GATE_K = 10            # seconds of evidence needed by duration filter / trust gate
NOISE_SD = 0.05        # perturbation (z-units) used for explanation-stability check
NOISE_REPS = 5
GATE_QS = [0.99, 0.995, 0.999]
GATE_SDS = [0.05, 0.2, 0.5]


# ----------------------------------------------------------------- data
def load(data_dir):
    tr = [pd.read_csv(f) for f in sorted(glob.glob(f"{data_dir}/train*.csv.gz"))]
    te_files = sorted(glob.glob(f"{data_dir}/test*.csv.gz"))
    te = []
    for i, f in enumerate(te_files):
        d = pd.read_csv(f)
        d["file"] = i
        te.append(d)
    return tr, pd.concat(te, ignore_index=True)


def prepare(tr, te):
    sens = [c for c in tr[0].columns if c[:2] in GROUPS]
    trall = pd.concat(tr, ignore_index=True)
    # drop sensors that never change in training (one distinct value). Checking
    # nunique instead of std == 0 avoids floating-point rounding differences
    # between pandas/NumPy versions (e.g. P2_MSD = 763.19324 everywhere).
    feats = [c for c in sens if trall[c].nunique(dropna=True) > 1]
    # chronological calibration tail of the last training file
    n3 = len(tr[-1]); cut = int(n3 * (1 - CAL_FRAC))
    fit_df = pd.concat(tr[:-1] + [tr[-1].iloc[:cut]], ignore_index=True)
    cal_df = tr[-1].iloc[cut:].reset_index(drop=True)
    mu, sig = fit_df[feats].mean(), fit_df[feats].std()
    sig = sig.where(sig > 1e-8, 1.0)                        # guard against near-zero spread
    z = lambda d: ((d[feats] - mu) / sig).to_numpy(np.float32)
    gidx = {g: np.array([i for i, c in enumerate(feats) if c.startswith(g)]) for g in GROUPS}
    base = np.median(z(fit_df), axis=0).astype(np.float32)
    return feats, gidx, z(fit_df), z(cal_df), z(te), base


def events(te):
    a = te["attack"].to_numpy()
    s = np.flatnonzero(np.diff(np.r_[0, a]) == 1)
    e = np.flatnonzero(np.diff(np.r_[a, 0]) == -1)
    out = []
    for k, (i, j) in enumerate(zip(s, e)):
        seg = te.iloc[i:j + 1]
        tg = [g for g in ["P1", "P2", "P3"] if seg[f"attack_{g}"].max() > 0]
        out.append(dict(event=k, start=int(i), end=int(j), dur=int(j - i + 1),
                        file=int(seg["file"].iloc[0]), targets="+".join(tg),
                        single=len(tg) == 1))
    return pd.DataFrame(out)


# ------------------------------------------------------------ detectors
class Detector:
    """score(X) -> per-row anomaly score; resid(X) -> per-feature residual (or None)."""
    name = ""
    def fit(self, X): ...
    def score(self, X): ...
    def resid(self, X): return None


class PCARecon(Detector):
    name = "PCA"
    def __init__(self, seed=0): self.seed = seed
    def fit(self, X):
        self.m = PCA(n_components=0.99, random_state=self.seed).fit(X); return self
    def resid(self, X): return X - self.m.inverse_transform(self.m.transform(X))
    def score(self, X): return np.mean(self.resid(X) ** 2, axis=1)


class MLPAE(Detector):
    name = "MLP-AE"
    def __init__(self, seed=0): self.seed = seed
    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(120_000, len(X)), replace=False)
        self.m = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu",
                              solver="adam", batch_size=512, max_iter=80,
                              early_stopping=True, n_iter_no_change=5,
                              random_state=self.seed).fit(X[idx], X[idx])
        return self
    def resid(self, X): return X - self.m.predict(X)
    def score(self, X): return np.mean(self.resid(X) ** 2, axis=1)


class IForest(Detector):
    name = "IForest"
    def __init__(self, seed=0): self.seed = seed
    def fit(self, X):
        self.m = IsolationForest(n_estimators=200, max_samples=4096,
                                 random_state=self.seed, n_jobs=-1).fit(X[::5]); return self
    def score(self, X): return -self.m.score_samples(X)


class ZMax(Detector):
    """Model-free baseline: largest absolute z-score of any sensor."""
    name = "Z-max"
    def __init__(self, seed=0): pass
    def fit(self, X): return self
    def resid(self, X): return X
    def score(self, X): return np.max(np.abs(X), axis=1)


def smooth(s, files):
    out = np.empty_like(s, dtype=np.float64)
    for f in np.unique(files):
        m = files == f
        out[m] = pd.Series(s[m]).rolling(SMOOTH, min_periods=1).mean().to_numpy()
    return out


# ---------------------------------------------------------- attribution
def attr_residual(det, X, gidx, cal_resid_sd):
    """Process attribution from per-feature residuals, normalised by their normal spread."""
    r = det.resid(X)
    if r is None:
        return None
    r = (r / cal_resid_sd) ** 2
    return np.stack([r[:, gidx[g]].mean(axis=1) for g in GROUPS], axis=1)


def attr_group_shapley(det, X, gidx, base):
    """Exact Shapley values of the anomaly score over the 4 process groups.
    Absent groups are replaced by the training median (16 coalitions)."""
    n, G = len(X), len(GROUPS)
    from itertools import combinations
    from math import factorial
    val = {}
    for k in range(G + 1):
        for S in combinations(range(G), k):
            Xs = np.repeat(base[None, :], n, axis=0)
            for g in S:
                Xs[:, gidx[GROUPS[g]]] = X[:, gidx[GROUPS[g]]]
            val[S] = det.score(Xs)
    phi = np.zeros((n, G))
    for g in range(G):
        others = [h for h in range(G) if h != g]
        for k in range(G):
            w = factorial(k) * factorial(G - k - 1) / factorial(G)
            for S in combinations(others, k):
                phi[:, g] += w * (val[tuple(sorted(S + (g,)))] - val[S])
    return phi


def attr_z(X, gidx):
    """Model-free attribution: mean |z| per process."""
    return np.stack([np.abs(X[:, gidx[g]]).mean(axis=1) for g in GROUPS], axis=1)


# --------------------------------------------------------------- metrics
def alarm_segments(alarm, files):
    segs = []
    for f in np.unique(files):
        idx = np.flatnonzero(files == f)
        a = alarm[idx]
        s = np.flatnonzero(np.diff(np.r_[0, a.astype(int)]) == 1)
        e = np.flatnonzero(np.diff(np.r_[a.astype(int), 0]) == -1)
        segs += [(idx[i], idx[j]) for i, j in zip(s, e)]
    return segs


def evaluate(alarm, y, files, ev, score_s=None):
    tp = int((alarm & (y == 1)).sum()); fp = int((alarm & (y == 0)).sum())
    fn = int((~alarm & (y == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-12)
    near = np.zeros_like(y, bool)
    det, delay = [], []
    for _, e in ev.iterrows():
        lo, hi = e.start, min(e.end + GRACE, len(y) - 1)
        near[lo:hi + 1] = True
        hit = np.flatnonzero(alarm[lo:hi + 1])
        det.append(len(hit) > 0); delay.append(hit[0] if len(hit) else np.nan)
    segs = alarm_segments(alarm, files)
    fa = sum(1 for s, t in segs if not near[s:t + 1].any())
    normal_hours = (~near).sum() / 3600
    return dict(precision=P, recall=R, f1=F1,
                auc_pr=average_precision_score(y, score_s) if score_s is not None else np.nan,
                event_recall=float(np.mean(det)),
                median_delay_s=float(np.nanmedian(delay)),
                false_alarms_per_hour=fa / normal_hours), np.array(det)


def duration_filter(alarm, files, k):
    out = np.zeros_like(alarm)
    for s, t in alarm_segments(alarm, files):
        if t - s + 1 >= k:
            out[s + k - 1:t + 1] = True      # raised once k seconds of evidence accumulate
    return out


def trust_gate(alarm, stable, files, k):
    """Raise an alarm segment only after k seconds whose explanation is stable."""
    out = np.zeros_like(alarm)
    for s, t in alarm_segments(alarm, files):
        c = np.cumsum(stable[s:t + 1])
        hit = np.flatnonzero(c >= k)
        if len(hit):
            out[s + hit[0]:t + 1] = True
    return out


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    tr, te = load(a.data)
    feats, gidx, Xfit, Xcal, Xte, base = prepare(tr, te)
    ev = events(te)
    y = te["attack"].to_numpy().astype(int); files = te["file"].to_numpy()
    in_attack = y == 1

    # stealth magnitude of each real event: 95th pct of max |z| over the target process(es)
    mag = []
    for _, e in ev.iterrows():
        cols = np.concatenate([gidx[g] for g in e.targets.split("+")])
        mag.append(float(np.percentile(np.abs(Xte[e.start:e.end + 1][:, cols]).max(axis=1), 95)))
    ev["magnitude_z"] = mag
    ev["stealth_tier"] = pd.qcut(ev["magnitude_z"], 3, labels=["high stealth", "medium", "low stealth"])
    ev.to_csv(f"{a.out}/events.csv", index=False)

    audit = dict(features_used=len(feats), dropped_constant=79 - len(feats),
                 group_sizes={g: int(len(gidx[g])) for g in GROUPS},
                 fit_rows=len(Xfit), cal_rows=len(Xcal), test_rows=len(Xte),
                 attack_rows=int(y.sum()), events=len(ev),
                 single_target_events=int(ev.single.sum()),
                 single_target_counts=ev[ev.single].targets.value_counts().to_dict(),
                 majority_class_hit_rate=float(ev[ev.single].targets.value_counts().iloc[0] / ev.single.sum()))
    json.dump(audit, open(f"{a.out}/dataset_audit.json", "w"), indent=2)

    det_rows, loc_rows, tier_rows, gate_rows, perev_rows, curve_rows = [], [], [], [], [], []
    for D in [PCARecon, MLPAE, IForest, ZMax]:
        seeds = SEEDS[:a.seeds] if D in (MLPAE, IForest) else SEEDS[:1]
        for seed in seeds:
            det = D(seed).fit(Xfit)
            cal_s = smooth(det.score(Xcal), np.zeros(len(Xcal)))
            thr = float(np.quantile(cal_s, THR_Q))
            raw = det.score(Xte); s = smooth(raw, files)
            alarm = s > thr
            m, detected = evaluate(alarm, y, files, ev, s)
            det_rows.append(dict(detector=D.name, seed=seed, threshold=thr, **m))

            # ---- localization on alarm points inside detected events
            cal_r = det.resid(Xcal)
            cal_sd = (np.std(cal_r, axis=0) + 1e-6) if cal_r is not None else None
            methods = {"Z-mean": lambda X: attr_z(X, gidx),
                       "GroupShapley": lambda X: attr_group_shapley(det, X, gidx, base)}
            if cal_sd is not None:
                methods["Residual"] = lambda X: attr_residual(det, X, gidx, cal_sd)
            for mname, fn in methods.items():
                hits = []
                for k, e in ev.iterrows():
                    allidx = np.arange(e.start, e.end + 1)
                    top_or = GROUPS[int(np.argmax(fn(Xte[allidx]).mean(axis=0)))]
                    hit_or = float(top_or in e.targets.split("+"))
                    idx = allidx[alarm[allidx]]
                    if len(idx) == 0:
                        hits.append(np.nan); top = None; hit = np.nan
                    else:
                        A = fn(Xte[idx]).mean(axis=0)
                        top = GROUPS[int(np.argmax(A))]
                        hit = float(top in e.targets.split("+"))
                        hits.append(hit)
                    perev_rows.append(dict(detector=D.name, seed=seed, method=mname, event=e.event,
                                           targets=e.targets, single=e.single, top=top, hit=hit,
                                           top_oracle=top_or, hit_oracle=hit_or,
                                           magnitude_z=e.magnitude_z, stealth_tier=str(e.stealth_tier)))
                hits = np.array(hits)
                sing = ev.single.to_numpy()
                lo, hi = boot_ci(hits[sing])
                loc_rows.append(dict(detector=D.name, seed=seed, method=mname,
                                     localized_events=int(np.nansum(hits[sing])),
                                     detected_single=int((~np.isnan(hits[sing])).sum()),
                                     hit_rate_single=float(np.nanmean(hits[sing])),
                                     ci_low=lo, ci_high=hi,
                                     hit_rate_multi=float(np.nanmean(hits[~sing])) if (~np.isnan(hits[~sing])).any() else np.nan))
                for tier in ["high stealth", "medium", "low stealth"]:
                    mk = (ev.stealth_tier == tier).to_numpy()
                    tier_rows.append(dict(detector=D.name, seed=seed, method=mname, tier=tier,
                                          events=int(mk.sum()),
                                          detection_rate=float(detected[mk].mean()),
                                          localization_rate_all=float(np.nan_to_num(hits[mk]).mean()),
                                          localization_rate_detected=float(np.nanmean(hits[mk])) if (~np.isnan(hits[mk])).any() else np.nan))

            # ---- trust gating (residual detectors only; residual attribution is cheap)
            # swept over pre-specified threshold quantiles and noise levels; all reported
            if cal_sd is not None:
                for q in GATE_QS:
                    thr_q = float(np.quantile(cal_s, q)); al_q = s > thr_q
                    aidx = np.flatnonzero(al_q)
                    top0 = np.argmax(attr_residual(det, Xte[aidx], gidx, cal_sd), axis=1)
                    for sd_ in GATE_SDS:
                        rng = np.random.default_rng(seed)
                        same = np.zeros(len(aidx))
                        for _ in range(NOISE_REPS):
                            Xn = Xte[aidx] + rng.normal(0, sd_, size=Xte[aidx].shape).astype(np.float32)
                            same += np.argmax(attr_residual(det, Xn, gidx, cal_sd), axis=1) == top0
                        stable = np.zeros(len(y), bool)
                        stable[aidx] = same >= NOISE_REPS - 1
                        pols = [("trust gate", trust_gate(al_q, stable, files, GATE_K))]
                        if sd_ == GATE_SDS[0]:
                            pols = [("no gate", al_q), ("duration filter", duration_filter(al_q, files, GATE_K))] + pols
                        for policy, al in pols:
                            mm, _ = evaluate(al, y, files, ev)
                            gate_rows.append(dict(detector=D.name, seed=seed, thr_quantile=q,
                                                  noise_sd=sd_ if policy == "trust gate" else np.nan,
                                                  policy=policy,
                                                  stable_share_attack=float(stable[aidx][in_attack[aidx]].mean()),
                                                  stable_share_normal=float(stable[aidx][~in_attack[aidx]].mean()) if (~in_attack[aidx]).any() else np.nan,
                                                  **mm))
                    for k in [1, 5, 10, 20, 30, 60, 90, 120]:
                        mk_, _ = evaluate(duration_filter(al_q, files, k), y, files, ev)
                        curve_rows.append(dict(detector=D.name, seed=seed, thr_quantile=q, k=k,
                                               event_recall=mk_["event_recall"],
                                               false_alarms_per_hour=mk_["false_alarms_per_hour"],
                                               median_delay_s=mk_["median_delay_s"]))
            print(f"[{time.time()-t0:7.0f}s] {D.name} seed {seed}: F1={m['f1']:.3f} "
                  f"event recall={m['event_recall']:.2f} FA/h={m['false_alarms_per_hour']:.2f}", flush=True)

    pd.DataFrame(det_rows).to_csv(f"{a.out}/E1_detection.csv", index=False)
    pd.DataFrame(loc_rows).to_csv(f"{a.out}/E2_localization.csv", index=False)
    pd.DataFrame(perev_rows).to_csv(f"{a.out}/E2_localization_per_event.csv", index=False)
    pd.DataFrame(tier_rows).to_csv(f"{a.out}/E3_stealth_tiers.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(f"{a.out}/E4_trust_gate.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(f"{a.out}/E4_duration_curve.csv", index=False)
    json.dump(dict(runtime_s=time.time() - t0, smooth_s=SMOOTH, grace_s=GRACE, cal_frac=CAL_FRAC,
                   threshold_quantile=THR_Q, gate_k=GATE_K, gate_threshold_quantiles=GATE_QS, gate_noise_sds=GATE_SDS, noise_reps=NOISE_REPS,
                   seeds=SEEDS[:a.seeds]), open(f"{a.out}/run_config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
