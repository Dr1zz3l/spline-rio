#!/usr/bin/env python3
"""rio_spline_node: live ROS1 wrapper around the deployed spline sliding-window
RIO solver (rio_solver.SlidingWindowSolver; 2026-10-13).

Same inputs/outputs and runtime as rio_gp_node (rio_runtime.RioNodeBase):
  publishes   rio_spline/odom       live edge (window end), once per stride
              rio_spline/odom_fast  IMU-propagated from the live edge, high rate
              rio_spline/status     JSON health / watchdog / timing
  (no covariance yet: the spline's live-edge covariance is an offline NEES
   tool, ceres::Covariance, too slow per stride)

Pipeline per stride (stride 0.3 s, window 3.0 s, the deployed profile
config/rio_spline_profile.yaml EXPORTED from the harness):
  1. shared causal front end (rio_frontend): IMU-anchored Doppler pre-unwrap
     + aliased-bearing repair + per-return weights with the SPLINE's law
     (0.0572, 0.02328), then per-frame RANSAC (no live re-unwrap: the spline
     harness has none)
  2. per-frame frame-shared sigma sigma_c(v) = sqrt(a^2 + (b |v_wls|)^2)
  3. warm start: the knots entering the window are IMU dead-reckoned from the
     previous solve's state at its window end, placed at their Greville times
     (ori knot j <-> (j+2) dt_ori, pos CP j <-> (j+3) dt_pos) relative to the
     solved seam knot, so the C++ warm-start alignment offset is zero
     (SlidingWindowSolver.set_init_tail)
  4. 10 Hz accelerometer increments in C++ (rio_solver.preintegrate_grid, the
     harness builder ported, identical to 1e-15), on a grid fixed at arm time
  5. solve_window(t_start, t_end, stride): the window GROWS from the arm time
     until it reaches window_duration (stride 0: no prior, the first-window
     pins hold the stationary arm state), then slides by one stride per solve
     (the C++ marginalizes exactly one stride ahead)
  6. live edge = the spline at the window end; trim_before() keeps memory
     bounded for long flights
Real time: the solve runs in the runtime's worker thread with a wall-clock
LM cap (max_solver_time_s, default 0.2 s < the 0.3 s stride).
"""
import math

import numpy as np
import rospy

from rio_frontend import (quat_to_R, R_to_quat, so3_exp, G_W, causal_reunwrap_stride,
                          set_bearing_law)
from rio_runtime import RioNodeBase, run_node


class RioSplineNode(RioNodeBase):
    NODE = 'rio_spline'

    def __init__(self):
        gp = lambda k, d: rospy.get_param('~' + k, d)
        set_bearing_law(*gp('bearing_law', [0.0572, 0.02328]))
        self.window = float(gp('window_duration', 3.0))
        self.sc_a = float(gp('radar_frame_sc_a', 0.0))
        self.sc_b = float(gp('radar_frame_sc_b', 0.0))
        self.rearm_boundary_scale = float(gp('rearm_boundary_scale', 1e-3))
        self.causal_unwrap = bool(gp('causal_unwrap', True))
        self.causal_omega = float(gp('causal_omega', 10.0))
        import rio_solver
        self._rs = rio_solver
        prof = dict(gp('solver_config', {}))
        if not prof:
            raise RuntimeError('rio_spline_node: no ~solver_config (load config/rio_spline_profile.yaml)')
        self._prof = prof
        self._max_solve = float(gp('max_solver_time_s', 0.2))
        self.cfg = self._make_cfg(1.0)
        ext = rio_solver.ExtrinsicConfig()
        _rot = gp('rotation_euler_deg', [180.0, 27.5, 0.0])
        _tbs = gp('translation_body_m', [0.08, 0.02, -0.01])
        ext.roll_deg, ext.pitch_deg, ext.yaw_deg = [float(x) for x in _rot]
        ext.tx, ext.ty, ext.tz = [float(x) for x in _tbs]
        self.ext = ext
        self.solver = None
        super().__init__()

    def _make_cfg(self, boundary_scale):
        c = self._rs.SolverConfig()
        for k, v in self._prof.items():
            if not hasattr(c, k):
                rospy.logwarn('rio_spline: unknown solver field %s (profile newer than the module?)', k)
                continue
            cur = getattr(c, k)
            setattr(c, k, type(cur)(v) if isinstance(cur, (bool, int, float)) else v)
        c.max_solver_time_s = self._max_solve
        if boundary_scale != 1.0:
            # in-air re-arm: the first-window pins hold a PROPAGATED state, not a
            # measured stationary one -> relax velocity / orientation (position
            # is the gauge either way)
            c.lambda_boundary_vel *= boundary_scale
            c.lambda_boundary_ori *= boundary_scale
        return c

    # ---------------------------------------------------------------- arm --
    def _backend_arm(self, p0, R0, biases, t0, anchor=None, v0=None):
        cfg = self._make_cfg(self.rearm_boundary_scale if anchor is not None else 1.0)
        self.cfg_live = cfg
        self.solver = self._rs.SlidingWindowSolver(cfg, self.ext)
        self.dtp, self.dto = float(cfg.dt_pos), float(cfg.dt_ori)
        # t_ref 0.4 s before the arm (a whole number of the 80 ms common
        # period): the spline is valid from t_ref + 5 dt_pos < t0
        t_ref = float(t0) - 0.4
        self.t_arm = float(t0)
        self.t_grid0 = t_ref                      # increment grid origin (fixed)
        p0 = np.asarray(p0, float); v0 = np.zeros(3) if v0 is None else np.asarray(v0, float)
        t_cover = t0 + self.stride + 0.5
        n_p = int(math.ceil((t_cover - t_ref) / self.dtp)) + 6
        n_o = int(math.ceil((t_cover - t_ref) / self.dto)) + 4
        tp = t_ref + (np.arange(n_p) + 3.0) * self.dtp
        P = p0[None, :] + v0[None, :] * (tp - t0)[:, None]
        Q = np.tile(R_to_quat(np.asarray(R0, float)), (n_o, 1))
        self.solver.initialize(P, Q, np.asarray(biases, float), t_ref)
        self.frames = []          # (t, RadarFrame)
        self.imu_win = np.zeros((0, 7))
        self.head_win = []
        self.t_end_prev = None
        self.p_arm, self.v_arm, self.R_arm = p0, v0, np.asarray(R0, float)
        self.strides_since_init = 0

    def _backend_biases(self):
        return (np.asarray(self.solver.biases, float) if self.solver is not None
                else np.zeros(6))

    def _rearm_velocity(self, v0):
        return v0

    # ---------------------------------------------------------- warm start --
    def _dead_reckon(self, t_e, times):
        """Spline state at t_e, IMU-propagated to each time (>= t_e)."""
        q, p, v, w = self.solver.sample([float(t_e)])
        R, p, v, om = quat_to_R(q[0]), np.array(p[0]), np.array(v[0]), np.array(w[0])
        b = np.asarray(self.solver.biases, float)
        imu = self.imu_win[self.imu_win[:, 0] > t_e]
        out_p, out_R = [], []
        k, tc = 0, float(t_e)
        for tau in times:
            while k < len(imu) and imu[k, 0] <= tau:
                h = imu[k, 0] - tc
                if 0.0 < h < 0.05:
                    a_w = R @ (imu[k, 1:4] - b[:3]) + G_W
                    p = p + v * h + 0.5 * a_w * h * h
                    v = v + a_w * h
                    om = imu[k, 4:7] - b[3:]
                    R = R @ so3_exp(om * h)
                tc = imu[k, 0]
                k += 1
            h = tau - tc                    # beyond the IMU: constant velocity / rate
            out_p.append(p + v * max(h, 0.0))
            out_R.append(R @ so3_exp(om * max(h, 0.0)))
        return out_p, out_R

    def _extend_init(self, t_end):
        s = self.solver
        t_ref = s.t_ref
        pi1 = int(round((t_end - t_ref) / self.dtp)) + 5
        oi1 = int(round((t_end - t_ref) / self.dto)) + 3
        if self.t_end_prev is None:
            # first solve after the arm: hold the arm state out to the window end
            n_p, n_o = s.n_pos_cps, s.n_ori_knots
            if pi1 >= n_p:
                tp = t_ref + (np.arange(n_p, pi1 + 1) + 3.0) * self.dtp
                P = self.p_arm[None, :] + self.v_arm[None, :] * (tp - self.t_arm)[:, None]
                Q = np.tile(R_to_quat(self.R_arm), (0, 1))
                s.set_init_tail(n_p, P, n_o, Q)
            if oi1 >= s.n_ori_knots:
                n_o = s.n_ori_knots
                s.set_init_tail(s.n_pos_cps, np.zeros((0, 3)), n_o,
                                np.tile(R_to_quat(self.R_arm), (oi1 + 1 - n_o, 1)))
            return
        seam_p = int(round((self.t_end_prev - t_ref) / self.dtp))
        seam_o = int(round((self.t_end_prev - t_ref) / self.dto))
        seam_p = min(seam_p, s.n_pos_cps - 1)
        seam_o = min(seam_o, s.n_ori_knots - 1)
        tau_p = t_ref + (np.arange(seam_p, pi1 + 1) + 3.0) * self.dtp
        tau_o = t_ref + (np.arange(seam_o, oi1 + 1) + 2.0) * self.dto
        tau_p = np.maximum(tau_p, self.t_end_prev)
        tau_o = np.maximum(tau_o, self.t_end_prev)
        # one IMU pass for both grids (sorted union of the sample times)
        allt = np.concatenate([tau_p, tau_o]); order = np.argsort(allt, kind='stable')
        dp, dR = self._dead_reckon(self.t_end_prev, allt[order])
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        cp = [dp[inv[i]] for i in range(len(tau_p))]
        cR = [dR[inv[len(tau_p) + j]] for j in range(len(tau_o))]
        P_seam = np.asarray(s.pos_cps[seam_p], float)
        R_seam = quat_to_R(np.asarray(s.ori_knots[seam_o], float))
        P = np.array([P_seam + (c - cp[0]) for c in cp])
        R_off = R_seam @ cR[0].T
        Q = np.array([R_to_quat(R_off @ Rj) for Rj in cR])
        s.set_init_tail(seam_p, P, seam_o, Q)

    # ------------------------------------------------------------- stride --
    def _backend_stride(self, imu_blk, unwrapped, heading_blk, t_now):
        rs = self._rs
        speeds = {t: spd for (t, _p, spd) in unwrapped}
        lst = [(t, pts) for (t, pts, _spd) in unwrapped]
        # live-window re-unwrap against the spline's OWN state at the previous
        # window end, IMU-propagated to each frame (the GP-22 machinery with the
        # 2026-09-27 propagation; measured on the spline harness 2026-09-13, par
        # to better).  Without it a whole frame mis-unwrapped by the per-frame
        # pre-unwrap reaches the live edge as a 1-4 m/s velocity spike (slow
        # replay).  Gated like the GP: warm window, |omega| < causal_omega.
        wmax = float(np.linalg.norm(imu_blk[:, 4:7], axis=1).max())
        pred, reuw = None, (self.causal_unwrap and self.t_end_prev is not None
                            and self.strides_since_init >= 5 and wmax < self.causal_omega)
        if reuw:
            try:
                q, p, v, w = self.solver.sample([float(self.t_end_prev)])
                b = np.asarray(self.solver.biases, float)
                hist = np.vstack([self.imu_win, imu_blk]) if len(self.imu_win) else imu_blk
                hist = hist[(hist[:, 0] > self.t_end_prev) & (hist[:, 0] <= t_now)]
                pred = (np.asarray(v[0], float), np.asarray(w[0], float), quat_to_R(q[0]),
                        float(self.t_end_prev), hist.reshape(-1, 7), b[:3], b[3:])
                if not all(np.all(np.isfinite(x)) for x in pred[:3]):
                    pred, reuw = None, False
            except Exception:
                pred, reuw = None, False
        n_re = 0
        if lst:
            lst, n_re = causal_reunwrap_stride(lst, pred, self.R_bs, self.t_bs, self.v_max,
                                               self.ransac_thr, self.rng, reunwrap=reuw,
                                               min_range=self.min_range)
        n_pts = 0
        for (t, pts) in lst:
            if len(pts) == 0:
                continue
            sc = (math.sqrt(self.sc_a ** 2 + (self.sc_b * speeds.get(t, 0.0)) ** 2)
                  if (self.sc_a > 0.0 or self.sc_b > 0.0) else -1.0)
            self.frames.append((float(t), rs.make_radar_frame(float(t), np.ascontiguousarray(pts[:, :6]), sc)))
            n_pts += len(pts)
        self.starve_run = 0 if n_pts else self.starve_run + 1
        self.imu_win = np.vstack([self.imu_win, imu_blk]) if len(self.imu_win) else imu_blk.copy()
        self.head_win.extend(heading_blk)
        # window bounds: grow from the arm time, then slide one stride per solve
        t_end = float(t_now)
        W, S = self.window, self.stride
        full = (t_end - self.t_arm) > W - 1e-6
        t_start = t_end - W if full else self.t_arm
        nxt_full = (t_end + S - self.t_arm) > W - 1e-6
        t_start_next = t_end + S - W if nxt_full else self.t_arm
        stride_param = max(0.0, t_start_next - t_start)
        # warm start, data slices, increments
        self._extend_init(t_end)
        m = (self.imu_win[:, 0] >= t_start) & (self.imu_win[:, 0] <= t_end)
        imu_w = np.ascontiguousarray(self.imu_win[m])
        radar_w = [rf for (t, rf) in self.frames if t_start <= t <= t_end]
        head_w = [(float(t), float(y)) for (t, y) in self.head_win if t_start <= t <= t_end]
        b = np.asarray(self.solver.biases, float)
        preint = []
        if self.cfg_live.accel_increment_hz > 0.0 and len(imu_w):
            preint = rs.preintegrate_grid(imu_w, b[:3], b[3:], t_start, t_end,
                                          1.0 / self.cfg_live.accel_increment_hz, self.t_grid0, True)
        flags, sample, hard = [], None, False
        if not len(imu_w):
            return {'sample': None, 't_pub': t_end, 'cov9': None, 'bias': b, 'dt_s': 0.0,
                    'iters': 0, 'flags': ['NO_IMU'], 'hard_warm': False, 'diag': {}}
        res = self.solver.solve_window(radar_w, rs.make_imu_samples(imu_w), preint, head_w,
                                       t_start, t_end, stride_param)
        self.t_end_prev = t_end
        self.strides_since_init += 1
        if res.num_iterations >= self.cfg_live.max_iterations:
            flags.append('LM_AT_CAP')
        try:
            q, p, v, w = self.solver.sample([t_end])
            sample = (np.asarray(q[0], float), np.asarray(p[0], float),
                      np.asarray(v[0], float), np.asarray(w[0], float))
            if not all(np.all(np.isfinite(x)) for x in sample):
                sample = None
        except Exception:
            sample = None
        bias = np.asarray(self.solver.biases, float)
        hard = sample is None or not np.all(np.isfinite(bias))
        # bounded memory: drop knots / data no later window can touch
        t_keep = t_start_next - 0.5
        self.solver.trim_before(t_keep)
        self.frames = [(t, rf) for (t, rf) in self.frames if t >= t_keep]
        self.imu_win = self.imu_win[self.imu_win[:, 0] >= t_keep - 0.1]
        self.head_win = [(t, y) for (t, y) in self.head_win if t >= t_keep]
        ch = list(res.cost_history)
        diag = {'frames': len(radar_w), 'pts_new': n_pts, 'iters': res.num_iterations,
                'n_re': int(n_re), 'reuw': bool(reuw),
                'cost0': float(ch[0]) if ch else -1.0, 'cost': float(ch[-1]) if ch else -1.0,
                'solve_ms': round(res.solve_time_s * 1000.0, 1), 'prior': bool(res.marg_prior_valid),
                'win_s': round(t_end - t_start, 2), 'n_pos': self.solver.n_pos_cps}
        return {'sample': None if hard else sample, 't_pub': t_end, 'cov9': None,
                'bias': bias, 'dt_s': res.solve_time_s, 'iters': res.num_iterations,
                'flags': flags, 'hard_warm': bool(hard and self.strides_since_init >= 3),
                'diag': diag}


def main():
    run_node('rio_spline_node', RioSplineNode, 'rio_solver_path')


if __name__ == '__main__':
    main()
