"""
Vectorized re-implementation of the cross-KPI predictive-maintenance benchmark described in the paper,
extended with CUSUM/EWMA baselines, paired and common-subset analyses, sensor-noise / missing-telemetry
stress, and an unseen-mode (open-set) probe. ALL DATA ARE SYNTHETIC.

The generator follows the degradation functions of the original harness (synthetic_data.py in the
earlier release): same baselines, ramps, onset ranges, and 17/17/16 mode split.
"""
import os, sys, json, zlib
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); os.makedirs(OUT, exist_ok=True)
DAYS, HOURS = 90, 90 * 24
KP = ["temperature", "voltage", "fan_speed", "fan_current", "cpu_util", "packet_error_rate"]
BASE = {"temperature": (45.0, 2.0), "voltage": (12.0, 0.15), "fan_speed": (3000.0, 100.0),
        "fan_current": (0.5, 0.03), "cpu_util": (35.0, 8.0), "packet_error_rate": (0.01, 0.005)}
MODES = ["fan_bearing", "power_supply", "thermal_interface"]
FEAT = KP + ["voltage_std"]            # 7 daily features
FI = {k: i for i, k in enumerate(FEAT)}
TARGET_FA = 0.02

def gen_node(rng, mode):
    h = np.arange(HOURS)
    X = np.empty((HOURS, 6))
    for j, k in enumerate(KP):
        m, s = BASE[k]
        X[:, j] = m + 0.5 * s * np.sin(2 * np.pi * (h % 24) / 24) + rng.normal(0, s, HOURS)
    if mode is None:
        return X, None
    ff = int(rng.integers(int(HOURS * 0.6), HOURS - 24))
    htf = ff - h
    def ramp(start):
        m = (htf > 0) & (htf <= start)
        r = np.zeros(HOURS); r[m] = np.clip((start - htf[m]) / start, 0, 1); return r, m
    c = {k: KP.index(k) for k in KP}
    if mode == "fan_bearing":
        r, _ = ramp(rng.integers(5 * 24, 9 * 24)); X[:, c["fan_current"]] += r * 0.35
        r, _ = ramp(rng.integers(2 * 24, 4 * 24)); X[:, c["temperature"]] += r * 12.0
        r, _ = ramp(rng.integers(6, 18)); X[:, c["fan_speed"]] -= r * 900.0
    elif mode == "power_supply":
        r, m = ramp(rng.integers(6 * 24, 10 * 24)); X[:, c["voltage"]] += rng.normal(0, 1, HOURS) * r * 0.5
        r, _ = ramp(rng.integers(4, 10)); X[:, c["voltage"]] -= r * 2.5
    elif mode == "thermal_interface":
        r, _ = ramp(rng.integers(7 * 24, 12 * 24)); X[:, c["temperature"]] += r * 15.0
        r, _ = ramp(rng.integers(3 * 24, 5 * 24)); X[:, c["cpu_util"]] += r * 25.0
    elif mode in ("ovA", "ovB", "ovC"):   # overlapping single-KPI signatures: each KPI appears in two modes
        use = {"ovA": ["temperature", "cpu_util"], "ovB": ["temperature", "fan_speed"], "ovC": ["cpu_util", "fan_speed"]}[mode]
        amp = {"temperature": (12.0, (5, 9)), "cpu_util": (25.0, (4, 7)), "fan_speed": (-900.0, (5, 9))}
        for k in use:
            a_, (lo, hi) = amp[k]; r, _ = ramp(rng.integers(lo * 24, hi * 24)); X[:, c[k]] += r * a_
    elif mode == "storage_wear":       # unseen mode for the open-set probe
        r, _ = ramp(rng.integers(5 * 24, 9 * 24)); X[:, c["packet_error_rate"]] += r * 0.03
        r, _ = ramp(rng.integers(3 * 24, 6 * 24)); X[:, c["cpu_util"]] += r * 10.0
    return X, ff

def daily(X, drop=None, noise=0.0, rng=None):
    """X: (HOURS,6) -> (DAYS,7). Optional hourly sample drop probability and extra noise multiplier."""
    X = X.copy()
    if noise > 0:
        sd = np.array([BASE[k][1] for k in KP])
        X = X + rng.normal(0, 1, X.shape) * sd * noise
    mask = np.ones(HOURS, bool)
    if drop:
        mask = rng.random(HOURS) > drop
    Xd = X.reshape(DAYS, 24, 6); md = mask.reshape(DAYS, 24)
    cnt = np.maximum(md.sum(1), 1)[:, None]
    mean = (Xd * md[:, :, None]).sum(1) / cnt
    v = Xd[:, :, 1]; vm = (v * md).sum(1) / cnt[:, 0]
    vs = np.sqrt(((v - vm[:, None]) ** 2 * md).sum(1) / np.maximum(cnt[:, 0] - 1, 1))
    return np.column_stack([mean, vs])

# ---------------- detectors ----------------
def first_persist(alarm, limit=None):
    """alarm (N,D) bool. Return first index d with alarm[d]&alarm[d+1] and (limit is None or d+1 < limit[i]), else -1."""
    p = alarm[:, :-1] & alarm[:, 1:]
    if limit is not None:
        idx = np.arange(p.shape[1])[None, :]
        p = p & ((idx + 1) < limit[:, None])
    has = p.any(1)
    return np.where(has, p.argmax(1), -1)

def calibrate(alarm_fn, params, calib_Z):
    best, gap = None, 9
    for pr in params:
        fa = (first_persist(alarm_fn(calib_Z, pr)) >= 0).mean()
        g = abs(fa - TARGET_FA)
        if g < gap - 1e-12 or (abs(g - gap) < 1e-12 and best is not None and pr > best[0]):
            best, gap, bfa = (pr, fa), g, fa
    return best  # (param, calib_fa)

def thr_alarm(feats):
    idx = [FI[k] for k in feats]
    return lambda Z, c: (np.abs(Z[:, :, idx]) > c).any(2)

def cusum_alarm(feats, k=0.5):
    idx = [FI[x] for x in feats]
    def f(Z, h):
        N, D, _ = Z.shape; sp = np.zeros((N, len(idx))); sn = np.zeros((N, len(idx))); A = np.zeros((N, D), bool)
        for t in range(D):
            z = Z[:, t, idx]
            sp = np.maximum(0, sp + z - k); sn = np.maximum(0, sn - z - k)
            A[:, t] = ((sp > h) | (sn > h)).any(1)
        return A
    return f

def ewma_alarm(feats, lam=0.3):
    idx = [FI[x] for x in feats]; sig = np.sqrt(lam / (2 - lam))
    def f(Z, L):
        N, D, _ = Z.shape; e = np.zeros((N, len(idx))); A = np.zeros((N, D), bool)
        for t in range(D):
            e = lam * Z[:, t, idx] + (1 - lam) * e
            A[:, t] = (np.abs(e) > L * sig).any(1)
        return A
    return f

def dist_series(Z, ref, kidx, W):
    """Z (N,D,7); ref (len(kidx), >=W) template by days-before-failure; returns (N, D-W+1) distances for t=W-1..D-1."""
    N, D, _ = Z.shape; out = np.zeros((N, D - W + 1))
    for tau in range(W):
        seg = Z[:, W - 1 - tau: D - tau, :][:, :, kidx]
        out += ((seg - ref[:, tau][None, None, :]) ** 2).sum(2)
    return out / (len(kidx) * W)

class Traj:
    def __init__(self, name, spec, W):
        self.name, self.spec, self.W = name, spec, W
    def fit(self, train, mu, sd):
        self.ref = {}
        for m in MODES:
            kidx = [FI[k] for k in self.spec[m]]
            acc = np.zeros((len(kidx), 12)); n = np.zeros(12)
            for Zr, fd in train[m]:
                Z = (Zr - mu) / sd
                for tau in range(12):
                    d = fd - tau
                    if 0 <= d < DAYS: acc[:, tau] += Z[d, kidx]; n[tau] += 1
            self.ref[m] = acc / np.maximum(n, 1)
        return self
    def dists(self, Z):
        D = Z.shape[1]
        out = np.full((Z.shape[0], D, len(MODES)), np.inf)
        for mi, m in enumerate(MODES):
            kidx = [FI[k] for k in self.spec[m]]
            out[:, self.W - 1:, mi] = dist_series(Z, self.ref[m], kidx, self.W)
        return out
    def calibrate(self, calib_Z):
        d = self.dists(calib_Z)
        self.base = np.array([np.median(d[:, self.W - 1:, mi]) for mi in range(len(MODES))])
        grid = np.arange(0.3, 4.01, 0.05)
        def fn(Z, s): return (self.dists(Z) < (self.base * s)[None, None, :]).any(2)
        # speed: reuse calibration distances
        best, gap = None, 9
        for s in grid:
            fa = (first_persist((d < (self.base * s)[None, None, :]).any(2)) >= 0).mean()
            g = abs(fa - TARGET_FA)
            if g < gap - 1e-12 or (abs(g - gap) < 1e-12 and s > best[0]): best, gap = (s, fa), g
        self.scalar, self.calib_fa = best
        self.thr = self.base * self.scalar
        return self
    def run(self, Z, limit=None):
        d = self.dists(Z); thr = self.thr[None, None, :]
        fire = d < thr
        A = fire.any(2)
        p = A[:, :-1] & A[:, 1:]
        if limit is not None:
            p = p & ((np.arange(p.shape[1])[None, :] + 1) < limit[:, None])
        has = p.any(1); first = np.where(has, p.argmax(1), -1)
        margin = (thr - d) / thr
        attr = np.full(Z.shape[0], -1)
        for i in np.nonzero(has)[0]:
            f = first[i]; mm = margin[i, f:f + 2, :].mean(0); attr[i] = int(np.argmax(mm))
        return first, attr, A

SPEC_CROSS = {"fan_bearing": ["fan_current", "temperature", "fan_speed"],
              "power_supply": ["voltage", "voltage_std"],
              "thermal_interface": ["temperature", "cpu_util"]}
SPEC_MEAN = {"fan_bearing": ["fan_current", "temperature", "fan_speed"],
             "power_supply": ["voltage"], "thermal_interface": ["temperature", "cpu_util"]}
SPEC_UNI = {"fan_bearing": ["fan_current"], "power_supply": ["voltage_std"], "thermal_interface": ["temperature"]}

def run_seed(seed, extras=True):
    rng = np.random.default_rng(np.random.SeedSequence([20261018, seed]))
    healthy = [gen_node(rng, None)[0] for _ in range(250)]
    deg = []
    for i in range(50):
        m = MODES[i % 3]; X, ff = gen_node(rng, m); deg.append((m, X, ff))
    cal_h, ev_h = healthy[:100], healthy[100:]
    by = {m: [d for d in deg if d[0] == m] for m in MODES}
    train = {m: [(daily(X), ff // 24) for _, X, ff in by[m][:3]] for m in MODES}
    ev_d = [d for m in MODES for d in by[m][3:]]
    Dc = np.stack([daily(X) for X in cal_h]); mu = Dc.reshape(-1, 7).mean(0); sd = Dc.reshape(-1, 7).std(0)
    Zc = (Dc - mu) / sd
    Zh = np.stack([(daily(X) - mu) / sd for X in ev_h])
    Zd = np.stack([(daily(X) - mu) / sd for _, X, _ in ev_d])
    fdays = np.array([ff // 24 for _, _, ff in ev_d]); tmode = np.array([MODES.index(m) for m, _, _ in ev_d])
    res = {"seed": seed}; nodes = []
    detectors = {}
    def add_alarm(name, fn, grid):
        pr, cfa = calibrate(fn, grid, Zc); detectors[name] = ("alarm", fn, pr, cfa)
    add_alarm("Threshold-Mean", thr_alarm(KP), np.arange(1.5, 6.01, 0.05))
    add_alarm("Threshold+Var", thr_alarm(FEAT), np.arange(1.5, 6.01, 0.05))
    if extras:
        add_alarm("CUSUM+Var", cusum_alarm(FEAT), np.arange(1.0, 12.01, 0.1))
        add_alarm("EWMA+Var", ewma_alarm(FEAT), np.arange(1.5, 6.01, 0.05))
    trajs = {"Mean Cross W12": Traj("mean", SPEC_MEAN, 12), "Uni-Traj W5": Traj("uni", SPEC_UNI, 5),
             "Cross-Traj W5": Traj("cross", SPEC_CROSS, 5)}
    if extras:
        for W in [3, 7, 9, 12]:
            trajs[f"Cross-Traj W{W}"] = Traj("cross", SPEC_CROSS, W)
    for nm, t in trajs.items():
        t.fit(train, mu, sd).calibrate(Zc); detectors[nm] = ("traj", t, t.scalar, t.calib_fa)

    def evaluate(Zh_, Zd_, tag=""):
        out = {}
        for nm, (kind, obj, pr, cfa) in detectors.items():
            if kind == "alarm":
                fa = (first_persist(obj(Zh_, pr)) >= 0).mean()
                f = first_persist(obj(Zd_, pr), fdays); attr = None
            else:
                fa = (obj.run(Zh_)[0] >= 0).mean()
                f, attr, _ = obj.run(Zd_, fdays)
            det = f >= 0; lead = np.where(det, fdays - f, np.nan)
            row = dict(miss=1 - det.mean(), fa=fa, lead_med=np.nanmedian(lead) if det.any() else np.nan,
                       calib_fa=cfa)
            if attr is not None:
                row["attr"] = (attr[det] == tmode[det]).mean() if det.any() else np.nan
            out[nm] = row
            if tag == "":
                for i in range(len(f)):
                    nodes.append(dict(seed=seed, det=nm, node=i, mode=MODES[tmode[i]], lead=lead[i],
                                      detected=bool(det[i]), correct=(None if attr is None else bool(attr[i] == tmode[i]))))
        return out
    res["main"] = evaluate(Zh, Zd)
    if extras:
        # stress: extra sensor noise and missing hourly telemetry on evaluation nodes (thresholds frozen)
        st = {}
        for label, kw in [("noise x1.1", dict(noise=0.46)), ("noise x1.25", dict(noise=0.75)),
                          ("10% samples missing", dict(drop=0.10)), ("30% samples missing", dict(drop=0.30))]:
            r2 = np.random.default_rng(np.random.SeedSequence([777, seed, zlib.crc32(label.encode())]))
            Zh2 = np.stack([(daily(X, rng=r2, **kw) - mu) / sd for X in ev_h])
            Zd2 = np.stack([(daily(X, rng=r2, **kw) - mu) / sd for _, X, _ in ev_d])
            st[label] = evaluate(Zh2, Zd2, tag=label)
        res["stress"] = st
        # open-set: 14 unseen-mode nodes
        un = [gen_node(rng, "storage_wear") for _ in range(14)]
        Zu = np.stack([(daily(X) - mu) / sd for X, _ in un]); fu = np.array([ff // 24 for _, ff in un])
        osr = {}
        for nm in ["Threshold+Var", "Uni-Traj W5", "Cross-Traj W5"]:
            kind, obj, pr, _ = detectors[nm]
            if kind == "alarm":
                f = first_persist(obj(Zu, pr), fu); osr[nm] = dict(detected=(f >= 0).mean())
            else:
                f, attr, _ = obj.run(Zu, fu); det = f >= 0
                osr[nm] = dict(detected=det.mean(), forced_label=[(attr[det] == m).mean() if det.any() else np.nan for m in range(3)])
        res["open_set"] = osr
    return res, nodes

def main(n_seeds=30):
    allres, allnodes = [], []
    for s in range(n_seeds):
        r, nd = run_seed(s); allres.append(r); allnodes += nd
        if s % 5 == 0: print("seed", s, "done", flush=True)
    json.dump(allres, open(os.path.join(OUT, "per_seed.json"), "w"), default=float)
    pd.DataFrame(allnodes).to_csv(os.path.join(OUT, "node_level.csv"), index=False)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n)
