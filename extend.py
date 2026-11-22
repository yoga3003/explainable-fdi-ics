"""
Two additions requested by the planned reviewer checklist (run after fdi_pipeline.py):

  E10  USAD (Audibert et al., 2020), a deep adversarially trained autoencoder on
       10-second windows, run through exactly the same detection, localization,
       stealth and trust-gate protocol as the four detectors in fdi_pipeline.py.
       Rows are written with detector = "USAD" in E10_*.csv (same schema as E1-E4).
  E11  Per-sensor Kernel SHAP (Lundberg and Lee, 2017), summed to processes, compared
       with exact group Shapley on the SAME sampled alarm seconds of every detected
       event, for every detector and seed. Also times both attributions.

Usage:
    python extend.py --data hai/hai-21.03 --results results [--seeds 5]
"""
import argparse, json, os, sys, time
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fdi_pipeline as fp

WIN = 10              # USAD window length (s)
KS_SAMPLES = 1024     # Kernel SHAP coalitions per explained second
KS_PER_EVENT = 20     # alarm seconds explained per detected event (evenly spaced)


# ======================================================================= USAD
def lagged(X, starts, win=WIN, rows=None):
    """Row t -> concatenation of rows t, t-1, ..., t-win+1 (lag-major), padded at segment starts.
    `starts` = first row index of every contiguous segment (file); `rows` = optional subset of t."""
    n, d = X.shape
    seg_start = np.zeros(n, dtype=np.int64)
    bounds = list(starts) + [n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg_start[a:b] = a
    rows = np.arange(n) if rows is None else np.asarray(rows)
    out = np.empty((len(rows), d * win), dtype=np.float32)
    for k in range(win):
        idx = np.maximum(rows - k, seg_start[rows])
        out[:, k * d:(k + 1) * d] = X[idx]
    return out


class USAD(fp.Detector):
    """USAD with one encoder and two decoders, trained with the two-phase adversarial
    objective of Audibert et al. (2020). Input = 10-s window (lag features), min-max scaled
    to [0, 1] on the training windows; decoders end in a sigmoid (reference implementation).
    score = mean over window features of 0.5*(x-AE1)^2 + 0.5*(x-AE2(AE1))^2, so that
    score = mean(resid^2) as for the other reconstruction detectors."""
    name = "USAD"

    def __init__(self, seed=0, latent=40, epochs=20, n_train=60_000):
        self.seed, self.latent, self.epochs, self.n_train = seed, latent, epochs, n_train

    def fit(self, X):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed); np.random.seed(self.seed); torch.set_num_threads(os.cpu_count() or 2)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(self.n_train, len(X)), replace=False)
        # min-max scaling to [0, 1] on the training windows, with sigmoid decoder outputs,
        # as in the reference USAD implementation
        self.lo = X[idx].min(axis=0); self.rng = np.maximum(X[idx].max(axis=0) - self.lo, 1e-6)
        Xt = torch.tensor(self._mm(X[idx]))
        d = X.shape[1]; h1, h2 = d // 2, d // 4
        enc = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, self.latent), nn.ReLU())
        dec = lambda: nn.Sequential(nn.Linear(self.latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, d), nn.Sigmoid())
        self.enc, self.d1, self.d2 = enc, dec(), dec()
        o1 = torch.optim.Adam(list(enc.parameters()) + list(self.d1.parameters()), lr=1e-3)
        o2 = torch.optim.Adam(list(enc.parameters()) + list(self.d2.parameters()), lr=1e-3)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(1, self.epochs + 1):
            perm = torch.randperm(len(Xt), generator=g)
            for b in range(0, len(Xt), 512):
                w = Xt[perm[b:b + 512]]
                z = enc(w); w1 = self.d1(z); w2 = self.d2(z); w3 = self.d2(enc(w1))
                l1 = (1 / ep) * ((w - w1) ** 2).mean() + (1 - 1 / ep) * ((w - w3) ** 2).mean()
                o1.zero_grad(); l1.backward(); o1.step()
                z = enc(w); w1 = self.d1(z); w2 = self.d2(z); w3 = self.d2(enc(w1))
                l2 = (1 / ep) * ((w - w2) ** 2).mean() - (1 - 1 / ep) * ((w - w3) ** 2).mean()
                o2.zero_grad(); l2.backward(); o2.step()
        for m in (self.enc, self.d1, self.d2):
            m.eval()
        return self

    def _mm(self, X):
        return ((X - self.lo) / self.rng).astype(np.float32)

    def resid(self, X):
        import torch
        out = np.empty_like(X, dtype=np.float32)
        with torch.no_grad():
            for b in range(0, len(X), 8192):
                w = torch.tensor(self._mm(X[b:b + 8192])); w1 = self.d1(self.enc(w)); w3 = self.d2(self.enc(w1))
                out[b:b + 8192] = torch.sqrt(0.5 * (w - w1) ** 2 + 0.5 * (w - w3) ** 2).numpy()
        return out

    def score(self, X):
        return np.mean(self.resid(X) ** 2, axis=1)


# ================================================================ Kernel SHAP
def kernel_shap(score, X, players, base, n_samples=KS_SAMPLES, seed=0):
    """Kernel SHAP (Lundberg and Lee, 2017) with a single reference point `base`.
    players: list of column-index arrays (one per player). Coalition sizes are sampled
    in proportion to the Shapley kernel, in complementary pairs, and the efficiency
    constraint sum(phi) = f(x) - f(base) is imposed exactly. Returns (n, M)."""
    rng = np.random.default_rng(seed)
    M = len(players)
    sizes = np.arange(1, M)
    w_size = (M - 1) / (sizes * (M - sizes)); w_size /= w_size.sum()
    half = n_samples // 2
    Z = np.zeros((2 * half, M), dtype=bool)
    s = rng.choice(sizes, size=half, p=w_size)
    for i, k in enumerate(s):
        on = rng.choice(M, size=k, replace=False)
        Z[2 * i, on] = True; Z[2 * i + 1] = ~Z[2 * i]
    col_of = np.zeros(sum(len(p) for p in players), dtype=np.int64)
    for j, p in enumerate(players):
        col_of[p] = j
    f0 = score(base[None, :])[0]
    fx = score(X)
    out = np.zeros((len(X), M))
    Zf = Z.astype(np.float64)
    A = Zf[:, :-1] - Zf[:, [-1]]
    for i, x in enumerate(X):
        Xs = np.where(Z[:, col_of], x[None, :], base[None, :]).astype(np.float32)
        y = score(Xs) - f0 - Zf[:, -1] * (fx[i] - f0)
        phi, *_ = np.linalg.lstsq(A, y, rcond=None)      # uniform weights: sizes already sampled by kernel
        out[i, :-1] = phi; out[i, -1] = (fx[i] - f0) - phi.sum()
    return out


def sample_seconds(idx, k=KS_PER_EVENT):
    if len(idx) <= k:
        return idx
    return idx[np.linspace(0, len(idx) - 1, k).round().astype(int)]


def validate_kernel_shap(det, X, gidx, base):
    """Kernel SHAP with the 4 processes as players must reproduce exact group Shapley."""
    players = [gidx[g] for g in fp.GROUPS]
    ks = kernel_shap(det.score, X, players, base, n_samples=4096)
    ex = fp.attr_group_shapley(det, X, gidx, base)
    return float(np.max(np.abs(ks - ex)) / (np.max(np.abs(ex)) + 1e-12))


# ======================================================== shared protocol run
def run_protocol(D, seed, Xfit, Xcal, Xte, base, gidx, y, files, ev, gate=True):
    """Same steps as fdi_pipeline.main for one detector and seed (detection,
    localization, stealth tiers, trust gate). Returns row lists and the fitted detector."""
    in_attack = y == 1
    det_rows, loc_rows, tier_rows, gate_rows, perev_rows, curve_rows = [], [], [], [], [], []
    t1 = time.perf_counter(); det = D(seed).fit(Xfit); fit_s = time.perf_counter() - t1
    cal_s = fp.smooth(det.score(Xcal), np.zeros(len(Xcal)))
    thr = float(np.quantile(cal_s, fp.THR_Q))
    s = fp.smooth(det.score(Xte), files); alarm = s > thr
    m, detected = fp.evaluate(alarm, y, files, ev, s)
    det_rows.append(dict(detector=D.name, seed=seed, threshold=thr, fit_s=fit_s, **m))
    cal_r = det.resid(Xcal)
    cal_sd = (np.std(cal_r, axis=0) + 1e-6) if cal_r is not None else None
    methods = {"Z-mean": lambda X: fp.attr_z(X[:, :len(base_cur)], gidx_cur),
               "GroupShapley": lambda X: fp.attr_group_shapley(det, X, gidx, base)}
    if cal_sd is not None:
        methods["Residual"] = lambda X: fp.attr_residual(det, X, gidx, cal_sd)
    for mname, fn in methods.items():
        hits = []
        for _, e in ev.iterrows():
            allidx = np.arange(e.start, e.end + 1)
            top_or = fp.GROUPS[int(np.argmax(fn(Xte[allidx]).mean(axis=0)))]
            hit_or = float(top_or in e.targets.split("+"))
            idx = allidx[alarm[allidx]]
            if len(idx) == 0:
                hits.append(np.nan); top = None; hit = np.nan
            else:
                top = fp.GROUPS[int(np.argmax(fn(Xte[idx]).mean(axis=0)))]
                hit = float(top in e.targets.split("+")); hits.append(hit)
            perev_rows.append(dict(detector=D.name, seed=seed, method=mname, event=e.event, targets=e.targets,
                                   single=e.single, top=top, hit=hit, top_oracle=top_or, hit_oracle=hit_or,
                                   magnitude_z=e.magnitude_z, stealth_tier=str(e.stealth_tier)))
        hits = np.array(hits); sing = ev.single.to_numpy(); lo, hi = fp.boot_ci(hits[sing])
        loc_rows.append(dict(detector=D.name, seed=seed, method=mname, localized_events=int(np.nansum(hits[sing])),
                             detected_single=int((~np.isnan(hits[sing])).sum()), hit_rate_single=float(np.nanmean(hits[sing])),
                             ci_low=lo, ci_high=hi,
                             hit_rate_multi=float(np.nanmean(hits[~sing])) if (~np.isnan(hits[~sing])).any() else np.nan))
        for tier in ["high stealth", "medium", "low stealth"]:
            mk = (ev.stealth_tier == tier).to_numpy()
            tier_rows.append(dict(detector=D.name, seed=seed, method=mname, tier=tier, events=int(mk.sum()),
                                  detection_rate=float(detected[mk].mean()),
                                  localization_rate_all=float(np.nan_to_num(hits[mk]).mean()),
                                  localization_rate_detected=float(np.nanmean(hits[mk])) if (~np.isnan(hits[mk])).any() else np.nan))
    if gate and cal_sd is not None:
        for q in fp.GATE_QS:
            al_q = s > float(np.quantile(cal_s, q)); aidx = np.flatnonzero(al_q)
            top0 = np.argmax(fp.attr_residual(det, Xte[aidx], gidx, cal_sd), axis=1)
            for sd_ in fp.GATE_SDS:
                rng = np.random.default_rng(seed); same = np.zeros(len(aidx))
                for _ in range(fp.NOISE_REPS):
                    Xn = Xte[aidx] + rng.normal(0, sd_, size=Xte[aidx].shape).astype(np.float32)
                    same += np.argmax(fp.attr_residual(det, Xn, gidx, cal_sd), axis=1) == top0
                stable = np.zeros(len(y), bool); stable[aidx] = same >= fp.NOISE_REPS - 1
                pols = [("trust gate", fp.trust_gate(al_q, stable, files, fp.GATE_K))]
                if sd_ == fp.GATE_SDS[0]:
                    pols = [("no gate", al_q), ("duration filter", fp.duration_filter(al_q, files, fp.GATE_K))] + pols
                for policy, al in pols:
                    mm, _ = fp.evaluate(al, y, files, ev)
                    gate_rows.append(dict(detector=D.name, seed=seed, thr_quantile=q,
                                          noise_sd=sd_ if policy == "trust gate" else np.nan, policy=policy,
                                          stable_share_attack=float(stable[aidx][in_attack[aidx]].mean()),
                                          stable_share_normal=float(stable[aidx][~in_attack[aidx]].mean()) if (~in_attack[aidx]).any() else np.nan,
                                          **mm))
            for k in [1, 5, 10, 20, 30, 60, 90, 120]:
                mk_, _ = fp.evaluate(fp.duration_filter(al_q, files, k), y, files, ev)
                curve_rows.append(dict(detector=D.name, seed=seed, thr_quantile=q, k=k, event_recall=mk_["event_recall"],
                                       false_alarms_per_hour=mk_["false_alarms_per_hour"], median_delay_s=mk_["median_delay_s"]))
    return det, alarm, (det_rows, loc_rows, tier_rows, gate_rows, perev_rows, curve_rows)


# globals used by the Z-mean lambda (current-row z-scores, also for lagged USAD input)
base_cur, gidx_cur = None, None


def main():
    global base_cur, gidx_cur
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--seeds", type=int, default=len(fp.SEEDS))
    ap.add_argument("--skip_usad", action="store_true")
    a = ap.parse_args(); R = a.results; t0 = time.time()

    tr, te = fp.load(a.data)
    feats, gidx, Xfit, Xcal, Xte, base = fp.prepare(tr, te)
    base_cur, gidx_cur = base, gidx
    ev = fp.events(te); evs = pd.read_csv(f"{R}/events.csv")
    ev["magnitude_z"] = evs.magnitude_z.values; ev["stealth_tier"] = evs.stealth_tier.values
    y = te["attack"].to_numpy().astype(int); files = te["file"].to_numpy()
    d = Xfit.shape[1]; seeds = fp.SEEDS[:a.seeds]

    # ---------------------------------------------------------------- E10 USAD
    n1, n2 = len(tr[0]), len(tr[1])
    fit_starts = [0, n1, n1 + n2]
    del tr
    import gc; gc.collect()
    te_starts = [int(np.flatnonzero(files == f)[0]) for f in np.unique(files)]
    # USAD trains on 60,000 windows per seed; build windows only for a fixed pool of 200,000 fitting rows
    pool = np.sort(np.random.default_rng(0).choice(len(Xfit), size=200_000, replace=False))
    Lfit, Lcal, Lte = lagged(Xfit, fit_starts, rows=pool), lagged(Xcal, [0]), lagged(Xte, te_starts)
    gidx_L = {g: np.concatenate([gidx[g] + k * d for k in range(WIN)]) for g in fp.GROUPS}
    base_L = np.tile(base, WIN).astype(np.float32)
    usad = {}
    if not a.skip_usad:
        rows = [[] for _ in range(6)]
        for seed in seeds:
            det, alarm, out = run_protocol(USAD, seed, Lfit, Lcal, Lte, base_L, gidx_L, y, files, ev)
            usad[seed] = (det, alarm)
            for r, o in zip(rows, out):
                r += o
            print(f"[{time.time()-t0:6.0f}s] USAD seed {seed}: F1={out[0][0]['f1']:.3f} "
                  f"event recall={out[0][0]['event_recall']:.2f} FA/h={out[0][0]['false_alarms_per_hour']:.3f}", flush=True)
        for name, r in zip(["E1_detection", "E2_localization", "E3_stealth_tiers", "E4_trust_gate",
                            "E2_localization_per_event", "E4_duration_curve"], rows):
            pd.DataFrame(r).to_csv(f"{R}/E10_{name}.csv", index=False)

    # ---------------------------------------------------------- E11 Kernel SHAP
    # validation on the PCA detector: 4-player Kernel SHAP vs exact group Shapley
    pca = fp.PCARecon(42).fit(Xfit)
    att = np.flatnonzero(y == 1)
    val_err = validate_kernel_shap(pca, Xte[att[::200][:40]], gidx, base)
    print(f"Kernel SHAP vs exact (4 players), max relative error = {val_err:.4f}", flush=True)

    sensors = [np.array([j]) for j in range(d)]
    sensors_L = [np.array([j + k * d for k in range(WIN)]) for j in range(d)]
    proc_of = np.array([next(g for g in fp.GROUPS if j in gidx[g]) for j in range(d)])
    rows, cost = [], []
    dets = [(fp.PCARecon, seeds[:1]), (fp.MLPAE, seeds), (fp.IForest, seeds), (fp.ZMax, seeds[:1])]
    if usad:
        dets.append((USAD, seeds))
    for D, sd_list in dets:
        for seed in sd_list:
            if D is USAD:
                det, alarm = usad[seed]; XT, B, G, P = Lte, base_L, gidx_L, sensors_L
                cal_s = None
            else:
                det = D(seed).fit(Xfit)
                cal_s = fp.smooth(det.score(Xcal), np.zeros(len(Xcal)))
                alarm = fp.smooth(det.score(Xte), files) > float(np.quantile(cal_s, fp.THR_Q))
                XT, B, G, P = Xte, base, gidx, sensors
            t_ks = t_gs = 0.0; n_expl = 0
            for _, e in ev.iterrows():
                idx = np.arange(e.start, e.end + 1); idx = idx[alarm[idx]]
                if len(idx) == 0:
                    continue
                sel = sample_seconds(idx)
                t1 = time.perf_counter(); phi = kernel_shap(det.score, XT[sel], P, B, seed=seed); t_ks += time.perf_counter() - t1
                t1 = time.perf_counter(); gs = fp.attr_group_shapley(det, XT[sel], G, B); t_gs += time.perf_counter() - t1
                n_expl += len(sel)
                ks_proc = np.stack([phi[:, proc_of == g].sum(axis=1) for g in fp.GROUPS], axis=1)
                tg = e.targets.split("+")
                for mname, A in [("KernelSHAP", ks_proc), ("GroupShapley (same seconds)", gs)]:
                    top = fp.GROUPS[int(np.argmax(A.mean(axis=0)))]
                    rows.append(dict(detector=D.name, seed=seed, method=mname, event=e.event, targets=e.targets,
                                     single=e.single, seconds_explained=len(sel), top=top, hit=float(top in tg)))
            cost.append(dict(detector=D.name, seed=seed, seconds_explained=n_expl,
                             kernel_shap_ms_per_second=1000 * t_ks / max(n_expl, 1),
                             group_shapley_ms_per_second=1000 * t_gs / max(n_expl, 1),
                             kernel_shap_evaluations=KS_SAMPLES + 2, group_shapley_evaluations=16))
            print(f"[{time.time()-t0:6.0f}s] Kernel SHAP {D.name} seed {seed}: {n_expl} seconds, "
                  f"{1000*t_ks/max(n_expl,1):.1f} vs {1000*t_gs/max(n_expl,1):.2f} ms/s", flush=True)
    pd.DataFrame(rows).to_csv(f"{R}/E11_kernelshap_per_event.csv", index=False)
    pd.DataFrame(cost).to_csv(f"{R}/E11_kernelshap_cost.csv", index=False)
    json.dump(dict(usad_window_s=WIN, usad_latent=40, usad_epochs=20, usad_train_windows=60_000, usad_window_pool=200_000,
                   kernel_shap_coalitions=KS_SAMPLES, kernel_shap_seconds_per_event=KS_PER_EVENT,
                   kernel_shap_validation_max_rel_error=val_err, runtime_s=time.time() - t0),
              open(f"{R}/extend_config.json", "w"), indent=2)
    print("extend ok", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
