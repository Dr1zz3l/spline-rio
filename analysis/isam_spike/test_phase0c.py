"""Phase 0c: ISAM2 incremental smoothing + the bias-model decision.

Feeds slow_racing into gtsam.ISAM2 stride-by-stride (full smoothing, NO
marginalization yet) for BOTH bias models, and measures the Bayes-tree fill-in
that the report flags for the shared global bias:
  - constant bias : one B(0) shared by every IMU factor
  - random-walk   : per-stride B(k) linked by a between-factor

Primary metric: variablesReeliminated per update (how much of the tree each
incremental step must refactor) and total clique count vs trajectory length.
Decisive question: does the affected set stay BOUNDED (random-walk) or grow /
stay large (constant bias = root-coupled)?

IMU is decimated to ~250 Hz here: this preserves the connectivity that drives
the fill-in (every ori knot still has a gyro factor -> bias couples to all
knots) while keeping the Python FD solve tractable.  Structure depends on
connectivity, not factor count.
"""
import os
import re
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gtsam
from gtsam.symbol_shorthand import P, R, B
import spline_factors as sf
from scipy.spatial.transform import Rotation

NPZ = os.path.join(HERE, '_cache', 'slow_racing_best_velocity_batch.npz')
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
STRIDE = 0.3
IMU_HZ = 250.0


def max_clique_size(isam):
    """Parse the Bayes-tree dot() for the largest clique (frontals+separator)."""
    try:
        dot = isam.dot()
    except Exception:
        return -1
    best = 0
    for label in re.findall(r'label="([^"]*)"', dot):
        # clique nodes list variable keys; count tokens that look like keys
        n = len(re.findall(r'[a-zA-Z]\d+', label))
        best = max(best, n)
    return best


class Incremental:
    def __init__(self, prob, bias_mode, init='cpp'):
        self.prob = prob
        self.mode = bias_mode
        # init source: 'cpp' = near-optimal (isolates STRUCTURAL fill-in from the
        # convergence transient); 'p1p3' = raw init (mixes in relinearization).
        self.iq = prob.d['cpp_ori_knots'] if init == 'cpp' else prob.init_ori_quats
        self.ip = prob.d['cpp_pos_cps'] if init == 'cpp' else prob.init_pos_cps
        self.ib = prob.d['cpp_biases'] if init == 'cpp' else prob.init_biases
        p = gtsam.ISAM2Params()
        p.setRelinearizeThreshold(0.01)
        p.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(p)
        self.added_ori = set()
        self.added_pos = set()
        self.added_minsnap = set()
        self.added_aacc = set()
        self.bias_k = 0
        self.cfg = prob.cfg
        self.imu = prob.d['imu']
        # decimate IMU
        keep = np.ones(len(self.imu), bool)
        step = max(1, int(round((1.0 / np.median(np.diff(self.imu[:, 0]))) / IMU_HZ)))
        self.imu = self.imu[::step]
        self.rts = prob.d['radar_ts']; self.rs = prob.d['radar_split']
        self.rp = prob.d['radar_pos']; self.rv = prob.d['radar_vel']
        self.head = prob.d['heading']
        self.noise_acc = sf._iso(3, self.cfg['lambda_accel'])
        self.noise_gyr = sf._iso(3, self.cfg['lambda_gyro'])
        self.noise_radar = sf._iso(1, 1.0)
        self.noise_snap = sf._iso(3, self.cfg['lambda_snap_pos'])
        self.noise_aacc = sf._iso(3, self.cfg['lambda_ori_accel'])
        self.rw_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)  # bias random-walk
        self.bias_key = lambda k: B(k)

    def _ensure_knot(self, g, v, ori_idx, pos_idx):
        for j in sorted(ori_idx):
            if j not in self.added_ori and 0 <= j < self.prob.n_ori:
                q = self.iq[j]
                v.insert(R(j), gtsam.Rot3(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
                self.added_ori.add(j)
        for i in sorted(pos_idx):
            if i not in self.added_pos and 0 <= i < self.prob.n_pos:
                v.insert(P(i), np.asarray(self.ip[i], float))
                self.added_pos.add(i)

    def add_stride(self, ta, tb, first):
        prob = self.prob
        g = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()

        # bias variable for this stride
        if first:
            v.insert(self.bias_key(0), np.asarray(self.ib, float))
            g.add(sf.make_bias_prior(prob, prob.init_biases,
                                     sf._iso(6, self.cfg['lambda_bias_prior_accel']),
                                     bias_key=self.bias_key(0)))
        if self.mode == 'rw' and not first:
            self.bias_k += 1
            v.insert(self.bias_key(self.bias_k), np.asarray(self.ib, float))
            g.add(sf.make_bias_between(self.bias_key(self.bias_k - 1),
                                       self.bias_key(self.bias_k), self.rw_noise))
        bkey = self.bias_key(self.bias_k if self.mode == 'rw' else 0)

        # collect new knots first (so factors find their vars), then factors
        imu = self.imu
        m = (imu[:, 0] > ta) & (imu[:, 0] <= tb)
        rows = [s for s in imu[m] if prob.in_domain(s[0])]
        for s in rows:
            t = float(s[0])
            ko, _ = prob.ori_active(t); kp, _, _, _ = prob.pos_active(t)
            self._ensure_knot(g, v, range(ko - 3, ko + 1), range(kp - 5, kp + 1))
        rad = [fi for fi in range(len(self.rts))
               if ta < self.rts[fi] <= tb and prob.in_domain(self.rts[fi])]
        for fi in rad:
            t = float(self.rts[fi])
            ko, _ = prob.ori_active(t); kp, _, _, _ = prob.pos_active(t)
            self._ensure_knot(g, v, range(ko - 3, ko + 1), range(kp - 5, kp + 1))

        # boundary pin on the very first knots (anchor the gauge)
        if first:
            for j in sorted(self.added_ori)[:4]:
                Rj = Rotation.from_quat(self.iq[j]).as_matrix()
                g.add(sf.make_rot_prior(R(j), Rj, gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)))
            for i in sorted(self.added_pos)[:6]:
                g.add(sf.make_vec_prior(P(i), self.ip[i],
                                        gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)))

        # sensor factors
        for s in rows:
            t = float(s[0])
            g.add(sf.make_accel_factor(prob, t, s[1:4], self.noise_acc, bias_key=bkey))
            g.add(sf.make_gyro_factor(prob, t, s[4:7], self.noise_gyr, bias_key=bkey))
        for fi in rad:
            t = float(self.rts[fi])
            pts = self.rp[self.rs[fi]:self.rs[fi + 1]]; vels = self.rv[self.rs[fi]:self.rs[fi + 1]]
            for j in range(len(pts)):
                u = pts[j] / max(np.linalg.norm(pts[j]), 1e-9)
                g.add(sf.make_radar_factor(prob, t, u, vels[j], self.noise_radar))
        # heading priors (yaw observability)
        for t, yaw in self.head:
            if ta < t <= tb and prob.in_domain(t):
                pass  # heading factor omitted in 0c (gauge pinned at start); structure-neutral

        # regularizers now fully supported
        for seg in range(max(0, min(self.added_pos)), max(self.added_pos) - (sf.N_POS - 1)):
            if seg not in self.added_minsnap and all((seg + l) in self.added_pos for l in range(sf.N_POS)):
                g.add(sf.make_minsnap_factor(prob, seg, self.noise_snap)); self.added_minsnap.add(seg)
        for i in range(min(self.added_ori), max(self.added_ori) - 1):
            if i not in self.added_aacc and all((i + l) in self.added_ori for l in range(3)):
                g.add(sf.make_angaccel_factor(prob, i, self.noise_aacc)); self.added_aacc.add(i)

        res = self.isam.update(g, v)
        return res, g.size()


def run_mode(prob, mode):
    inc = Incremental(prob, mode)
    t0 = prob.t_ref + 5.0
    n_str = int(SECONDS / STRIDE)
    print(f"\n=== bias mode: {mode} ===")
    print(f"{'stride':>6} {'t_rel':>6} {'+fac':>5} {'reElim':>7} {'reLin':>6} "
          f"{'cliques':>7} {'totVar':>7} {'upd_s':>6}")
    reelim_hist = []
    for k in range(n_str):
        ta = t0 + k * STRIDE; tb = ta + STRIDE
        t = time.time()
        res, nf = inc.add_stride(ta, tb, first=(k == 0))
        dt = time.time() - t
        re_n = res.getVariablesReeliminated(); rl_n = res.getVariablesRelinearized()
        nclq = res.getCliques()
        totv = len(inc.added_ori) + len(inc.added_pos) + (inc.bias_k + 1 if mode == 'rw' else 1)
        reelim_hist.append(re_n)
        print(f"{k:>6} {ta - prob.t_ref:>6.2f} {nf:>5} {re_n:>7} {rl_n:>6} "
              f"{nclq:>7} {totv:>7} {dt:>6.1f}")
    mcs = max_clique_size(inc.isam)
    # steady-state reElim (ignore first 3 warm-up strides)
    ss = reelim_hist[3:] if len(reelim_hist) > 3 else reelim_hist
    print(f"  -> max clique size = {mcs} vars;  steady-state reElim "
          f"mean={np.mean(ss):.0f} max={np.max(ss)} (vs total {totv})")
    return inc, reelim_hist, mcs


def main():
    prob = sf.Problem(NPZ)
    print(f"Phase 0c: ISAM2 incremental over {SECONDS}s, stride {STRIDE}s, IMU~{IMU_HZ}Hz")
    inc_c, rc, mcs_c = run_mode(prob, 'const')
    inc_r, rr, mcs_r = run_mode(prob, 'rw')
    print("\n=== VERDICT ===")
    print(f"constant bias: max clique {mcs_c} vars, steady reElim mean {np.mean(rc[3:]):.0f}")
    print(f"random-walk  : max clique {mcs_r} vars, steady reElim mean {np.mean(rr[3:]):.0f}")
    print("Bounded + small reElim/clique that does NOT grow with length => banded, iSAM2-friendly.")


if __name__ == '__main__':
    main()
