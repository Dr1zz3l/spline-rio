"""Phase 2 validation: drive the C++ rio_isam.IsamSolver over the captured
slow_racing problem (full IMU rate, fed in non-overlapping strides) and compare
the resulting trajectory to the Ceres BATCH solution.  Also reports real C++
per-update timing + active-variable count (a Phase-3 preview).

Usage (from analysis/):  ../.venv/bin/python3 isam_spike/validate_isam_cpp.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'rio_solver_cpp', 'build_release')))
import rio_isam
import spline_factors as sf
from scipy.spatial.transform import Rotation

NPZ = os.path.join(HERE, '_cache', 'slow_racing_best_velocity_batch.npz')
STRIDE = 0.3


def sample_traj(prob, ori_xyzw, pos_cps, times):
    R_knots = Rotation.from_quat(ori_xyzw).as_matrix()
    Rs, ps = [], []
    for t in times:
        ko, uo = prob.ori_active(t)
        R4 = [R_knots[ko - 3 + l] for l in range(4)]
        Rm, _ = sf.eval_ori_local(R4, uo, prob.inv_dt_ori)
        kp, N0, _, _ = prob.pos_active(t)
        Rs.append(Rm); ps.append(N0 @ pos_cps[kp - prob.pos_degree: kp + 1])
    return Rs, np.array(ps)


def main():
    prob = sf.Problem(NPZ)
    cfg = rio_isam.IsamConfig()
    c = prob.cfg
    cfg.dt_pos = prob.dt_pos; cfg.dt_ori = prob.dt_ori
    cfg.lambda_accel = c['lambda_accel']; cfg.lambda_gyro = c['lambda_gyro']
    cfg.huber_delta = c['huber_delta']; cfg.lambda_snap_pos = c['lambda_snap_pos']
    cfg.lambda_ori_accel = c['lambda_ori_accel']
    cfg.lambda_heading = c['lambda_heading']
    cfg.lambda_bias_prior = c['lambda_bias_prior_accel']
    cfg.lag = float(os.environ.get('LAG', '1.5'))
    if os.environ.get('RELIN'):
        cfg.relinearize_threshold = float(os.environ['RELIN'])
    if os.environ.get('BIAS_RW'):
        cfg.bias_rw_sigma = float(os.environ['BIAS_RW'])
    if os.environ.get('LH'):
        cfg.lambda_heading = float(os.environ['LH'])

    ext = rio_isam.ExtrinsicConfig()
    ext.roll_deg, ext.pitch_deg, ext.yaw_deg = [float(x) for x in prob.ext_euler_deg]
    ext.tx, ext.ty, ext.tz = [float(x) for x in prob.t_bs]

    solver = rio_isam.IsamSolver(cfg, ext)
    # INIT_CPP=1 diagnostic: seed knots from the Ceres solution (near-optimal) to
    # isolate factor/smoother correctness from warm-start quality.
    if os.environ.get('INIT_CPP'):
        ip = prob.d['cpp_pos_cps'].astype(float); iq = prob.d['cpp_ori_knots'].astype(float)
        print('[INIT_CPP] seeding from Ceres solution')
    else:
        ip = prob.init_pos_cps.astype(float); iq = prob.init_ori_quats.astype(float)
    solver.initialize(ip, iq, prob.init_biases.astype(float), float(prob.t_ref))

    imu = prob.d['imu']; rts = prob.d['radar_ts']; rs = prob.d['radar_split']
    rp = prob.d['radar_pos']; rv = prob.d['radar_vel']; ri = prob.d['radar_int']
    head = prob.d['heading']
    t_lo = max(imu[0, 0], rts[0]) + (sf.N_POS) * prob.dt_pos
    t_hi = min(imu[-1, 0], rts[-1])
    print(f"feeding {t_hi - t_lo:.1f}s, stride {STRIDE}s, full IMU "
          f"({np.median(1/np.diff(imu[:,0])):.0f} Hz)")

    times_upd, nactive = [], []
    t_prev = t_lo
    t_now = t_lo + STRIDE
    nstr = 0
    while t_now <= t_hi + 1e-6:
        m = (imu[:, 0] > t_prev) & (imu[:, 0] <= t_now)
        imu_blk = imu[m]
        radar_list = []
        for fi in range(len(rts)):
            if t_prev < rts[fi] <= t_now:
                a, b = rs[fi], rs[fi + 1]
                pts = np.column_stack([rp[a:b], rv[a:b], ri[a:b]])
                radar_list.append((float(rts[fi]), pts.astype(float)))
        hmask = (head[:, 0] > t_prev) & (head[:, 0] <= t_now)
        heading = [(float(t), float(y)) for t, y in head[hmask]]
        if len(imu_blk) == 0:
            t_prev, t_now = t_now, t_now + STRIDE; continue
        dt = solver.update(radar_list, imu_blk.astype(float), heading, float(t_now))
        times_upd.append(dt); nactive.append(solver.num_active())
        nstr += 1
        t_prev, t_now = t_now, t_now + STRIDE

    # assemble estimated knots (fallback to cpp where not estimated)
    ori_full = prob.d['cpp_ori_knots'].astype(float).copy()
    pos_full = prob.d['cpp_pos_cps'].astype(float).copy()
    got_o = solver.ori_knots(); got_p = solver.pos_cps()
    for idx, q in got_o: ori_full[idx] = q
    for idx, cc in got_p: pos_full[idx] = cc

    # compare to cpp batch over the covered interior
    idxs = sorted(i for i, _ in got_o)
    j0, j1 = idxs[3], idxs[-3]
    ta = prob.t_ref + j0 * prob.dt_ori + 0.1
    tb = prob.t_ref + j1 * prob.dt_ori - 0.1
    ts = np.linspace(ta, tb, 200)
    Re, pe = sample_traj(prob, ori_full, pos_full, ts)
    Rc, pc = sample_traj(prob, prob.d['cpp_ori_knots'], prob.d['cpp_pos_cps'], ts)
    d_ori = np.degrees([np.linalg.norm(sf.so3_log(a.T @ b)) for a, b in zip(Re, Rc)])
    d_pos = np.linalg.norm(pe - pc, axis=1)

    print(f"\nstrides={nstr}  knots estimated: {len(got_o)} ori, {len(got_p)} pos")
    half = len(times_upd) // 3
    full = times_upd[half:]; nvf = nactive[half:]
    print(f"per-update C++ time (lag full): mean={1000*np.mean(full):.1f}ms "
          f"max={1000*np.max(full):.1f}ms  (stride budget {1000*STRIDE:.0f}ms)")
    print(f"active vars (lag full): mean={np.mean(nvf):.0f} max={np.max(nvf)}  (plateau => bounded)")
    print(f"vs Ceres BATCH over covered interior:  "
          f"pos max={1000*d_pos.max():.1f}mm mean={1000*d_pos.mean():.1f}mm  | "
          f"ori max={d_ori.max():.3f} mean={d_ori.mean():.3f} deg")


if __name__ == '__main__':
    main()
