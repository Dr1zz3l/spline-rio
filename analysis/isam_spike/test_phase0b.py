"""Phase 0b: batch parity of the gtsam graph vs the Ceres optimum over a window.

Two checks on a ~1 s interior window of slow_racing:
  (A) STATIONARITY: initialize at the C++-solved knots, solve gtsam LM with the
      outer knots pinned to C++ values.  If the factors/weights/conventions match
      Ceres, the C++ optimum is also a gtsam optimum -> the solve barely moves.
  (B) CONVERGENCE: initialize the free interior at the P1-P3 init instead, solve,
      and compare the gtsam trajectory to the C++ trajectory over the window.

Python finite-difference makes a full-trajectory solve intractable; a sub-window
isolates factor/weight correctness (timing verdict is C++ Phase 3, per the plan).
"""
import os
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
WIN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0      # window length (s)


def sample_traj(prob, ori_xyzw, pos_cps, times):
    """R(t) (list of 3x3), p(t) (N,3) from absolute knot arrays."""
    R_knots = Rotation.from_quat(ori_xyzw).as_matrix()
    Rs, ps = [], []
    for t in times:
        ko, uo = prob.ori_active(t)
        R4 = [R_knots[ko - 3 + l] for l in range(4)]
        Rm, _ = sf.eval_ori_local(R4, uo, prob.inv_dt_ori)
        kp, N0, _, _ = prob.pos_active(t)
        Rs.append(Rm)
        ps.append(N0 @ pos_cps[kp - prob.pos_degree: kp + 1])
    return Rs, np.array(ps)


def build_graph(prob, ta, tb, use_cpp_boundary=True):
    """Graph + active index sets over [ta,tb].  Returns (graph, ori_idx, pos_idx)."""
    g = gtsam.NonlinearFactorGraph()
    cfg = prob.cfg
    n_radar = nimu = 0
    ori_used, pos_used = set(), set()

    # sensor factors
    imu = prob.d['imu']
    noise_acc = sf._iso(3, cfg['lambda_accel'])
    noise_gyr = sf._iso(3, cfg['lambda_gyro'])
    m = (imu[:, 0] >= ta) & (imu[:, 0] <= tb)
    for s in imu[m]:
        t = float(s[0])
        if not prob.in_domain(t):
            continue
        g.add(sf.make_accel_factor(prob, t, s[1:4], noise_acc))
        g.add(sf.make_gyro_factor(prob, t, s[4:7], noise_gyr))
        ko, _ = prob.ori_active(t); kp, _, _, _ = prob.pos_active(t)
        ori_used.update(range(ko - 3, ko + 1)); pos_used.update(range(kp - 5, kp + 1))
        nimu += 1

    rts = prob.d['radar_ts']; rs = prob.d['radar_split']
    rp = prob.d['radar_pos']; rv = prob.d['radar_vel']
    noise_radar = sf._iso(1, 1.0)
    for fi in range(len(rts)):
        t = float(rts[fi])
        if not (ta <= t <= tb and prob.in_domain(t)):
            continue
        pts = rp[rs[fi]:rs[fi + 1]]; vels = rv[rs[fi]:rs[fi + 1]]
        ko, _ = prob.ori_active(t); kp, _, _, _ = prob.pos_active(t)
        ori_used.update(range(ko - 3, ko + 1)); pos_used.update(range(kp - 5, kp + 1))
        for j in range(len(pts)):
            u = pts[j] / max(np.linalg.norm(pts[j]), 1e-9)
            g.add(sf.make_radar_factor(prob, t, u, vels[j], noise_radar))
            n_radar += 1

    # regularizers within the used range
    oi = sorted(ori_used); pi = sorted(pos_used)
    noise_snap = sf._iso(3, cfg['lambda_snap_pos'])
    noise_aacc = sf._iso(3, cfg['lambda_ori_accel'])
    for seg in range(pi[0], pi[-1] - (sf.N_POS - 1)):
        if all((seg + l) in pos_used for l in range(sf.N_POS)):
            g.add(sf.make_minsnap_factor(prob, seg, noise_snap))
    for i in range(oi[0], oi[-1] - 1):
        if all((i + l) in ori_used for l in range(3)):
            g.add(sf.make_angaccel_factor(prob, i, noise_aacc))

    # bias prior (10000)
    g.add(sf.make_bias_prior(prob, prob.init_biases,
                             sf._iso(6, cfg['lambda_bias_prior_accel'])))
    return g, oi, pi, n_radar, nimu


def main():
    prob = sf.Problem(NPZ)
    cpp_q = prob.d['cpp_ori_knots']; cpp_p = prob.d['cpp_pos_cps']
    t0 = prob.t_ref + 6.0
    ta, tb = t0, t0 + WIN
    print(f"window [{ta - prob.t_ref:.2f}, {tb - prob.t_ref:.2f}] s rel  (len {WIN}s)")

    g, oi, pi, n_radar, nimu = build_graph(prob, ta, tb)
    print(f"graph: {g.size()} factors  ({nimu} imu*2 + {n_radar} radar + reg)  "
          f"vars: {len(oi)} ori, {len(pi)} pos")

    # boundary: pin outer 4 ori + outer 6 pos at C++ values (strong)
    pin_ori = set(oi[:4]) | set(oi[-4:])
    pin_pos = set(pi[:6]) | set(pi[-6:])
    rot_pin = gtsam.noiseModel.Isotropic.Sigma(3, 1e-4)
    vec_pin = gtsam.noiseModel.Isotropic.Sigma(3, 1e-4)
    for j in pin_ori:
        Rj = Rotation.from_quat(cpp_q[j]).as_matrix()
        g.add(sf.make_rot_prior(R(j), Rj, rot_pin))
    for i in pin_pos:
        g.add(sf.make_vec_prior(P(i), cpp_p[i], vec_pin))

    # sample grid (interior, away from pinned edges)
    times = np.linspace(ta + 0.15, tb - 0.15, 60)
    Rc, pc = sample_traj(prob, cpp_q, cpp_p, times)

    def run(init_q, init_p, label):
        v = gtsam.Values()
        for j in oi:
            q = init_q[j]; v.insert(R(j), gtsam.Rot3(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
        for i in pi:
            v.insert(P(i), np.asarray(init_p[i], float))
        v.insert(B(0), np.asarray(prob.init_biases, float))
        e0 = g.error(v)
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(40)
        params.setlambdaInitial(1e-3)
        opt = gtsam.LevenbergMarquardtOptimizer(g, v, params)
        t = time.time()
        res = opt.optimize()
        dt = time.time() - t
        # extract solved knot arrays (start from C++, overwrite active)
        sq = cpp_q.copy(); sp = cpp_p.copy()
        for j in oi:
            R3 = res.atRot3(R(j)); q = R3.toQuaternion(); sq[j] = [q.x(), q.y(), q.z(), q.w()]
        for i in pi:
            sp[i] = res.atVector(P(i))
        Rs, ps = sample_traj(prob, sq, sp, times)
        d_ori = np.degrees(max(np.linalg.norm(sf.so3_log(Ra.T @ Rb)) for Ra, Rb in zip(Rs, Rc)))
        d_pos = float(np.max(np.linalg.norm(ps - pc, axis=1)))
        print(f"  [{label}] err {e0:.1f} -> {g.error(res):.1f}  ({opt.iterations()} it, {dt:.1f}s)  "
              f"vs C++:  max d_pos={d_pos * 1000:.2f} mm  max d_ori={d_ori:.4f} deg")
        return d_pos, d_ori

    print("(A) STATIONARITY  init = C++ solution:")
    run(cpp_q, cpp_p, "A")
    print("(B) CONVERGENCE   init = P1-P3 init:")
    run(prob.init_ori_quats, prob.init_pos_cps, "B")


if __name__ == '__main__':
    main()
