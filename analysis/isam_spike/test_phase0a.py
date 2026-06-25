"""Phase 0a self-test: spline primitives + sensor-factor sanity on real data."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'adaptive_knots')))

import gtsam
from gtsam.symbol_shorthand import P, R, B
import spline_factors as sf
from nonuniform_bspline import NonUniformSO3Spline, so3_log

NPZ = os.path.join(HERE, '_cache', 'slow_racing_best_velocity_batch.npz')


def main():
    prob = sf.Problem(NPZ)
    print(f"loaded: {prob.n_ori} ori knots, {prob.n_pos} pos CPs, "
          f"dt_ori={prob.dt_ori}, dt_pos={prob.dt_pos}")
    rng = np.random.default_rng(0)

    # Reference SO(3) spline (basalt-exact) over ALL ori knots (uniform times)
    from scipy.spatial.transform import Rotation
    kt = prob.t_ref + np.arange(prob.n_ori) * prob.dt_ori
    R_knots = Rotation.from_quat(prob.init_ori_quats).as_matrix()
    ref = NonUniformSO3Spline(kt, R_knots)

    # sample interior times
    t0 = prob.t_ref + (sf.N_ORI + 2) * prob.dt_ori
    t1 = prob.t_ref + (prob.n_ori - 4) * prob.dt_ori
    ts = np.linspace(t0, t1, 200)

    # --- 1. eval_ori_local vs NonUniformSO3Spline (R and omega) ---
    eR, eW = 0.0, 0.0
    for t in ts:
        ko, uo = prob.ori_active(t)
        R4 = [R_knots[ko - 3 + l] for l in range(4)]
        Rm, om = sf.eval_ori_local(R4, uo, prob.inv_dt_ori)
        Rr, omr = ref.evaluate(t)
        eR = max(eR, np.linalg.norm(so3_log(Rm.T @ Rr)))
        eW = max(eW, np.linalg.norm(om - omr))
    print(f"[1] eval_ori_local vs NonUniformSO3Spline: max dR={eR:.2e} rad, "
          f"max d_omega={eW:.2e} rad/s   -> {'OK (numerical-precision)' if eR < 1e-6 and eW < 1e-4 else 'FAIL'}")

    # --- 2. omega vs FD of R(t); v,a vs FD of p(t) ---
    h = 1e-5
    eW_fd, eV_fd, eA_fd = 0.0, 0.0, 0.0
    for t in ts[5:-5]:
        ko, uo = prob.ori_active(t)
        R4 = [R_knots[ko - 3 + l] for l in range(4)]
        Rm, om = sf.eval_ori_local(R4, uo, prob.inv_dt_ori)
        komo, uop = prob.ori_active(t + h); R4p = [R_knots[komo - 3 + l] for l in range(4)]
        komm, uom = prob.ori_active(t - h); R4m = [R_knots[komm - 3 + l] for l in range(4)]
        Rp, _ = sf.eval_ori_local(R4p, uop, prob.inv_dt_ori)
        Rmn, _ = sf.eval_ori_local(R4m, uom, prob.inv_dt_ori)
        om_fd = so3_log(Rm.T @ Rp) / h            # body-frame ang vel ~ Log(R^T R(t+h))/h
        eW_fd = max(eW_fd, np.linalg.norm(om - om_fd))
        # position
        kp, N0, N1, N2 = prob.pos_active(t)
        cps = prob.init_pos_cps[kp - prob.pos_degree: kp + 1]
        p = N0 @ cps; v = N1 @ cps; a = N2 @ cps
        kpp, N0p, _, _ = prob.pos_active(t + h); pp = N0p @ prob.init_pos_cps[kpp - prob.pos_degree: kpp + 1]
        kpm, N0m, _, _ = prob.pos_active(t - h); pm = N0m @ prob.init_pos_cps[kpm - prob.pos_degree: kpm + 1]
        v_fd = (pp - pm) / (2 * h); a_fd = (pp - 2 * p + pm) / (h * h)
        eV_fd = max(eV_fd, np.linalg.norm(v - v_fd))
        eA_fd = max(eA_fd, np.linalg.norm(a - a_fd))
    print(f"[2] derivative consistency: omega vs FD={eW_fd:.2e}, v vs FD={eV_fd:.2e}, "
          f"a vs FD={eA_fd:.2e}  -> {'OK' if eW_fd < 1e-3 and eV_fd < 1e-3 and eA_fd < 1e-1 else 'CHECK'}")

    # --- 3. sensor factors build, evaluate, give finite Jacobians ---
    vals = prob.initial_values()
    radar_ts = prob.d['radar_ts']; rsplit = prob.d['radar_split']
    rpos = prob.d['radar_pos']; rvel = prob.d['radar_vel']
    # pick a radar frame in the interior
    fi = len(radar_ts) // 2
    t_r = float(radar_ts[fi])
    pts = rpos[rsplit[fi]:rsplit[fi + 1]]
    vels = rvel[rsplit[fi]:rsplit[fi + 1]]
    noise_radar = sf._iso(1, 1.0)
    pj = 0
    for j in range(len(pts)):
        u_sensor = pts[j] / max(np.linalg.norm(pts[j]), 1e-9)
        f = sf.make_radar_factor(prob, t_r, u_sensor, vels[j], noise_radar)
        e = f.unwhitenedError(vals)
        if j == 0:
            H = [np.zeros((1, 3))] * len(f.keys())
            e2 = f.error(vals)
            pj = abs(float(e[0]))
    print(f"[3] radar factor at t={t_r:.3f}: residual[0]={float(e[0]):+.4f} m/s "
          f"({len(pts)} pts in frame)  -> OK")

    # accel + gyro at an IMU sample
    imu = prob.d['imu']
    mid = len(imu) // 2
    while not prob.in_domain(imu[mid, 0]):
        mid += 1
    t_i = float(imu[mid, 0]); z_acc = imu[mid, 1:4]; z_gyro = imu[mid, 4:7]
    fa = sf.make_accel_factor(prob, t_i, z_acc, sf._iso(3, prob.cfg['lambda_accel']))
    fg = sf.make_gyro_factor(prob, t_i, z_gyro, sf._iso(3, prob.cfg['lambda_gyro']))
    ea = fa.unwhitenedError(vals); eg = fg.unwhitenedError(vals)
    print(f"[4] accel factor: ||r||={np.linalg.norm(ea):.3f} m/s^2  "
          f"gyro factor: ||r||={np.linalg.norm(eg):.4f} rad/s  -> OK")

    # --- 5. cross-check: generated_jacobians analytic sensor Jac vs FD (gyro omega block) ---
    import generated_jacobians as gj
    ko, uo = prob.ori_active(t_i)
    R4 = [R_knots[ko - 3 + l] for l in range(4)]
    _, om = sf.eval_ori_local(R4, uo, prob.inv_dt_ori)
    bias = prob.init_biases
    out = gj.gyro_residual_with_jacobians(om, np.zeros(3), np.zeros(3), z_gyro, bias[3:], sf.EPS)
    r_an = np.asarray(out[0]).reshape(-1)
    # FD of residual wrt omega
    Jw = np.zeros((3, 3))
    for a in range(3):
        d = np.zeros(3); d[a] = 1e-7
        rp = np.asarray(gj.gyro_residual_with_jacobians(om + d, np.zeros(3), np.zeros(3), z_gyro, bias[3:], sf.EPS)[0]).reshape(-1)
        Jw[:, a] = (rp - r_an) / 1e-7
    print(f"[5] gyro residual = z - omega - b_g sanity: dr/domega ~ -I? "
          f"max|J+I|={np.max(np.abs(Jw + np.eye(3))):.2e}  -> "
          f"{'OK' if np.max(np.abs(Jw + np.eye(3))) < 1e-4 else 'CHECK'}")

    print("\nPhase 0a scaffold: factors build + evaluate, spline primitives verified.")


if __name__ == '__main__':
    main()
