"""Phase 0d: IncrementalFixedLagSmoother + the GATE.

Random-walk bias (0c decision). Feeds slow_racing into a gtsam
IncrementalFixedLagSmoother (lag = WINDOW) stride-by-stride; the lag marginalizes
knots older than t_now - lag.  Gate questions (the genuinely NEW iSAM2 risks):

  1. CONDITIONING: does marginalization run stably at our cond(H) ~ 5.5e10 (what
     killed BandedSchur)?  QR vs Cholesky.
  2. BOUNDED: do per-update time / clique count stay flat once the lag is full
     (the real-time premise)?
  3. CONSISTENCY (NEES-like, FEJ probe): vs a full-smoothing ISAM2 (no marg, the
     exact reference).  Does the live edge drift, and is the smoother's marginal
     covariance calibrated (e^T Sigma^-1 e ~ dof)?  Relinearize-threshold acts as
     the FEJ on/off proxy (gtsam has no direct FEJ flag).

Full-rate accuracy + mocap NEES is deferred to the C++ port (Phase 3); the Python
FD spike cannot run the full trajectory at 1 kHz.  0b already showed gtsam
reproduces the Ceres optimum, so accuracy parity is expected once full-rate.
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gtsam
import gtsam_unstable as gu
from gtsam.symbol_shorthand import P, R, B
import spline_factors as sf
from scipy.spatial.transform import Rotation

NPZ = os.path.join(HERE, '_cache', 'slow_racing_best_velocity_batch.npz')
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
LAG = 1.5
STRIDE = 0.3
IMU_HZ = 250.0


class StrideFeeder:
    """Builds per-stride (graph, values, timestamps) with random-walk bias,
    init at the C++ solution. Stateful across strides."""

    def __init__(self, prob):
        self.prob = prob
        self.iq = prob.d['cpp_ori_knots']; self.ip = prob.d['cpp_pos_cps']
        self.ib = prob.d['cpp_biases']
        self.added_ori, self.added_pos = set(), set()
        self.added_snap, self.added_aacc = set(), set()
        self.bias_k = 0
        cfg = prob.cfg
        imu = prob.d['imu']
        step = max(1, int(round((1.0 / np.median(np.diff(imu[:, 0]))) / IMU_HZ)))
        self.imu = imu[::step]
        self.rts = prob.d['radar_ts']; self.rs = prob.d['radar_split']
        self.rp = prob.d['radar_pos']; self.rv = prob.d['radar_vel']
        self.n_acc = sf._iso(3, cfg['lambda_accel']); self.n_gyr = sf._iso(3, cfg['lambda_gyro'])
        self.n_rad = sf._iso(1, 1.0); self.n_snap = sf._iso(3, cfg['lambda_snap_pos'])
        self.n_aacc = sf._iso(3, cfg['lambda_ori_accel'])
        self.rw = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        self.cfg = cfg

    def _ensure(self, v, ts, t_now, ori_idx, pos_idx):
        for j in ori_idx:
            if 0 <= j < self.prob.n_ori:
                if j not in self.added_ori:
                    q = self.iq[j]
                    v.insert(R(j), gtsam.Rot3(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
                    self.added_ori.add(j)
                ts[R(j)] = t_now
        for i in pos_idx:
            if 0 <= i < self.prob.n_pos:
                if i not in self.added_pos:
                    v.insert(P(i), np.asarray(self.ip[i], float))
                    self.added_pos.add(i)
                ts[P(i)] = t_now

    def stride(self, ta, tb, first):
        prob = self.prob
        g = gtsam.NonlinearFactorGraph(); v = gtsam.Values(); ts = {}
        t_now = tb
        # bias variable (random walk)
        if first:
            v.insert(B(0), np.asarray(self.ib, float))
            g.add(sf.make_bias_prior(prob, prob.init_biases,
                                     sf._iso(6, self.cfg['lambda_bias_prior_accel']), bias_key=B(0)))
        else:
            self.bias_k += 1
            v.insert(B(self.bias_k), np.asarray(self.ib, float))
            g.add(sf.make_bias_between(B(self.bias_k - 1), B(self.bias_k), self.rw))
        bkey = B(self.bias_k)
        ts[bkey] = t_now
        if not first:
            ts[B(self.bias_k - 1)] = t_now   # keep predecessor alive one more lag

        imu = self.imu
        rows = [s for s in imu[(imu[:, 0] > ta) & (imu[:, 0] <= tb)] if prob.in_domain(s[0])]
        rad = [fi for fi in range(len(self.rts))
               if ta < self.rts[fi] <= tb and prob.in_domain(self.rts[fi])]
        for s in rows:
            t = float(s[0]); ko, _ = prob.ori_active(t); kp, _, _, _ = prob.pos_active(t)
            self._ensure(v, ts, t_now, range(ko - 3, ko + 1), range(kp - 5, kp + 1))
        for fi in rad:
            t = float(self.rts[fi]); ko, _ = prob.ori_active(t); kp, _, _, _ = prob.pos_active(t)
            self._ensure(v, ts, t_now, range(ko - 3, ko + 1), range(kp - 5, kp + 1))

        if first:   # anchor the gauge
            for j in sorted(self.added_ori)[:4]:
                Rj = Rotation.from_quat(self.iq[j]).as_matrix()
                g.add(sf.make_rot_prior(R(j), Rj, gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)))
            for i in sorted(self.added_pos)[:6]:
                g.add(sf.make_vec_prior(P(i), self.ip[i], gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)))

        for s in rows:
            t = float(s[0])
            g.add(sf.make_accel_factor(prob, t, s[1:4], self.n_acc, bias_key=bkey))
            g.add(sf.make_gyro_factor(prob, t, s[4:7], self.n_gyr, bias_key=bkey))
        for fi in rad:
            t = float(self.rts[fi]); pts = self.rp[self.rs[fi]:self.rs[fi + 1]]
            vels = self.rv[self.rs[fi]:self.rs[fi + 1]]
            for j in range(len(pts)):
                u = pts[j] / max(np.linalg.norm(pts[j]), 1e-9)
                g.add(sf.make_radar_factor(prob, t, u, vels[j], self.n_rad))
        for seg in range(min(self.added_pos), max(self.added_pos) - (sf.N_POS - 1)):
            if seg not in self.added_snap and all((seg + l) in self.added_pos for l in range(sf.N_POS)):
                g.add(sf.make_minsnap_factor(prob, seg, self.n_snap)); self.added_snap.add(seg)
        for i in range(min(self.added_ori), max(self.added_ori) - 1):
            if i not in self.added_aacc and all((i + l) in self.added_ori for l in range(3)):
                g.add(sf.make_angaccel_factor(prob, i, self.n_aacc)); self.added_aacc.add(i)
        return g, v, ts, bkey


def make_tsm(ts):
    m = gu.FixedLagSmootherKeyTimestampMap()
    for k, t in ts.items():
        m.insert((k, float(t)))
    return m


def live_edge_ori(prob, est, added_ori):
    """Most recent fully-supported ori knot index present in est."""
    js = sorted(added_ori)
    for j in reversed(js[:-1]):
        if est.exists(R(j)):
            return j
    return js[len(js) // 2]


def run(prob, factorization='QR', relin=0.01):
    p = gtsam.ISAM2Params(); p.setFactorization(factorization)
    p.setRelinearizeThreshold(relin); p.relinearizeSkip = 1
    smoother = gu.IncrementalFixedLagSmoother(LAG, p)
    feeder = StrideFeeder(prob)
    t0 = prob.t_ref + 5.0
    n = int(SECONDS / STRIDE)
    print(f"\n=== IFLS  factorization={factorization}  lag={LAG}s  relin={relin} ===")
    print(f"{'k':>3} {'t_rel':>6} {'+fac':>5} {'actVar':>6} {'iters':>5} {'upd_s':>6}  note")
    times, nvars = [], []
    fail = None
    for k in range(n):
        ta = t0 + k * STRIDE; tb = ta + STRIDE
        g, v, ts, bkey = feeder.stride(ta, tb, first=(k == 0))
        try:
            t = time.time()
            res = smoother.update(g, v, make_tsm(ts))
            dt = time.time() - t
        except Exception as e:
            fail = f"stride {k}: {type(e).__name__}: {str(e)[:80]}"
            print(f"{k:>3} {ta - prob.t_ref:>6.2f}  -> FAILURE: {fail}")
            break
        na = smoother.timestamps().size()    # active variables kept by the lag
        times.append(dt); nvars.append(na)
        note = "lag full" if (tb - t0) > LAG else ""
        print(f"{k:>3} {ta - prob.t_ref:>6.2f} {g.size():>5} {na:>6} {res.getIterations():>5} {dt:>6.2f}  {note}")
    est = smoother.calculateEstimate()
    return smoother, feeder, est, times, fail, nvars


def main():
    prob = sf.Problem(NPZ)
    print(f"Phase 0d gate: lag={LAG}s stride={STRIDE}s over {SECONDS}s, IMU~{IMU_HZ}Hz, random-walk bias")

    # 1) CONDITIONING: QR (primary) + Cholesky (does it fail?)
    sm, feeder, est, times, fail, nvars = run(prob, 'QR', 0.01)
    if fail:
        print(f"\nGATE conditioning(QR): FAIL -> {fail}")
    else:
        half = len(times) // 2
        full = times[half:]; nvf = nvars[half:]
        print(f"\n[1] CONDITIONING(QR): no failure over {len(times)} strides.")
        print(f"    active vars once lag full: {nvf}  (plateau => marginalization caps the problem)")
        print(f"    update time once lag full: mean={np.mean(full):.2f}s max={np.max(full):.2f}s")

    print("\n[chol] Cholesky conditioning probe:")
    _, _, _, _, fail_c, _ = run(prob, 'CHOLESKY', 0.01)
    print(f"  Cholesky: {'FAILED -> '+fail_c if fail_c else 'survived'}  "
          f"(QR is the safe default either way)")

    # 2) CONSISTENCY vs full-smoother (no marginalization)
    print("\n[2] CONSISTENCY vs full-smoothing ISAM2 (exact reference):")
    pf = gtsam.ISAM2Params(); pf.setFactorization('QR'); pf.setRelinearizeThreshold(0.01)
    isam_full = gtsam.ISAM2(pf)
    feeder2 = StrideFeeder(prob)
    t0 = prob.t_ref + 5.0; n = int(SECONDS / STRIDE)
    drifts = []
    for k in range(n):
        ta = t0 + k * STRIDE; tb = ta + STRIDE
        g, v, ts, bkey = feeder2.stride(ta, tb, first=(k == 0))
        isam_full.update(g, v)
    est_full = isam_full.calculateEstimate()
    # compare live-edge ori knots common to both
    common = sorted(set(feeder.added_ori) & set(feeder2.added_ori))
    for j in common:
        if est.exists(R(j)) and est_full.exists(R(j)):
            Ra = est.atRot3(R(j)).matrix(); Rb = est_full.atRot3(R(j)).matrix()
            drifts.append(np.degrees(np.linalg.norm(sf.so3_log(Ra.T @ Rb))))
    if drifts:
        print(f"  IFLS(marg) vs full-smoother, ori over {len(drifts)} live knots: "
              f"max={np.max(drifts):.4f} mean={np.mean(drifts):.4f} deg")
    # NEES-like at live edge
    j = live_edge_ori(prob, est, feeder.added_ori)
    try:
        cov = sm.getISAM2().marginalCovariance(R(j))
        Ra = est.atRot3(R(j)).matrix(); Rb = est_full.atRot3(R(j)).matrix()
        e = sf.so3_log(Ra.T @ Rb)
        nees = float(e @ np.linalg.solve(cov, e))
        print(f"  live-edge ori NEES = {nees:.2f} (dof 3; ~[0.2,9.3] is 95% chi2)")
    except Exception as ex:
        print(f"  marginalCovariance probe skipped: {type(ex).__name__}")

    print("\n=== GATE SUMMARY (structural de-risk; full-rate accuracy -> Phase 3) ===")
    print(f"  conditioning(QR): {'PASS' if not fail else 'FAIL'}")
    print(f"  bounded cost: see [1]   consistency vs exact smoother: see [2]")


if __name__ == '__main__':
    main()
