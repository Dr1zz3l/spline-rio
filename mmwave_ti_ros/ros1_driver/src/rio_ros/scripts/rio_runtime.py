#!/usr/bin/env python3
"""rio_runtime: the ROS1 runtime shared by the live RIO nodes (2026-10-13).

Moved out of rio_gp_node.py and made real-time safe; the solver backends
(rio_gp_node.RioGpNode, rio_spline_node.RioSplineNode) subclass RioNodeBase
and implement four hooks:

    _backend_arm(p0, R0, biases, t0, anchor)   build a fresh solver
    _backend_stride(imu_blk, unwrapped, heading_blk, t_now)
        -> dict(sample=(q,p,v,om) | None, t_pub, cov9 | None, bias(6),
                dt_s, iters, flags[list], hard_warm[bool], diag[dict])
    _backend_biases() -> (6,) current [ba, bg]

Data flow (all stamps converted to IMU time on ingestion):
  callbacks  -> buffers only (IMU, radar, heading) + the high-rate IMU
                propagation of the last live edge (odom_fast)
  stride     -> closed when the IMU watermark passes t_prev + stride +
                margin + radar offset (+ accel lag): per-frame causal
                pre-unwrap (shared), then the backend solve
  threaded   -> (default, live) a worker thread closes strides; callbacks
                never wait for a solve.  threaded=false (replay) closes them
                inside the IMU callback, deterministic.
  backlog    -> if the worker falls more than max_backlog_sec behind the
                IMU stream, the stale strides are dropped and the backend is
                re-armed from the propagated state (flag BACKLOG_REARM).
"""
import json
import threading
import time as _time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Float64MultiArray, String

from rio_frontend import (so3_exp, quat_to_R, R_to_quat, euler_deg_to_R_bs, G_W,
                          PreUnwrapper)

WAIT_DATA, STATIONARY, RUNNING = 'WAIT_DATA', 'STATIONARY_INIT', 'RUNNING'


class RioNodeBase:
    NODE = 'rio'                 # topic prefix / log tag, set by the subclass

    def __init__(self):
        gp = lambda k, d: rospy.get_param('~' + k, d)
        self._gp = gp
        # --- topics ---
        self.topic_radar = gp('topic_radar', '/mmWaveDataHdl/RScanVelocity')
        self.topic_imu = gp('topic_imu', '/angrybird2/imu')
        self.topic_pose = gp('topic_pose', '/mocap/angrybird2/pose')
        # --- extrinsics / timing (per-day calibration!) ---
        self.rot_euler = gp('rotation_euler_deg', [180.0, 27.5, 0.0])
        self.t_bs = np.array(gp('translation_body_m', [0.08, 0.02, -0.01]))
        self.radar_imu_offset = float(gp('radar_imu_offset_sec', 0.135))
        self.imu_mocap_offset = float(gp('imu_mocap_offset_sec', 0.015))
        # --- radar config ---
        self.v_max = float(gp('doppler_v_max', 3.136))
        # measured constants of the Mar-2026 rig (2026-09-16/17): gyro per-axis
        # scale omega = (1 + s) z (worklog/2026-09-16_gyro-scale.md, y +1.44 %)
        # and the aliased-bearing repair; both OFF unless the profile sets them
        self.gyro_scale = np.asarray(gp('gyro_scale', [0.0, 0.0, 0.0]), float)
        # IMU-to-body misalignment (2026-09-27, worklog/2026-09-27_circle-vertical-
        # imu-misalignment.md): gyro AND accel rotated z_body = Exp(c) z_imu
        # before the scale; OFF ([0,0,0]) unless the profile sets it
        _c = np.radians(np.asarray(gp('imu_misalign_deg', [0.0, 0.0, 0.0]), float))
        _th = float(np.linalg.norm(_c))
        if _th > 0.0:
            _k = _c / _th
            _K = np.array([[0.0, -_k[2], _k[1]], [_k[2], 0.0, -_k[0]], [-_k[1], _k[0], 0.0]])
            self.R_mis = np.eye(3) + np.sin(_th) * _K + (1.0 - np.cos(_th)) * (_K @ _K)
        else:
            self.R_mis = None
        # accelerometer-only lag (2026-09-30, worklog/2026-09-29_accel-lag.md):
        # acc(t) <- acc(t + L) when a stride block is cut (the watermark waits
        # L longer).  The spline profile sets 0.020; the GP keeps 0.
        self.accel_lag = float(gp('accel_lag_sec', 0.0))
        self.alias_repair = bool(gp('alias_repair', False))
        self.ransac_thr = float(gp('radar_ransac_threshold', 0.15))
        self.min_range = float(gp('min_range', 0.2))
        # --- stride machinery ---
        self.stride = float(gp('stride_sec', 0.3))
        self.watermark = float(gp('imu_watermark_margin_sec', 0.02))
        self.threaded = bool(gp('threaded', True))
        self.max_backlog = float(gp('max_backlog_sec', 1.5))
        # --- bootstrap ---
        self.stat_sec = float(gp('min_stationary_sec', 1.0))
        self.stat_gyro = float(gp('stationary_gyro_max', 0.03))
        self.stat_acc_std = float(gp('stationary_accel_std_max', 0.12))
        # --- watchdog ---
        self.wd_iter_strides = int(gp('watchdog_iter_cap_strides', 5))
        self.wd_ms_budget = float(gp('watchdog_ms_budget', 250.0))
        self.wd_pos_sigma = float(gp('watchdog_pos_sigma_max', 3.0))
        self.wd_starve_strides = int(gp('watchdog_radar_starve_strides', 10))
        self.wd_rearm_wait = float(gp('watchdog_rearm_wait_sec', 3.0))
        self.rearm_anchor = float(gp('rearm_anchor_pose_sigma', 0.05))
        self.rearm_quiet_gyro = float(gp('rearm_quiet_gyro', 2.0))
        # --- output ---
        self.fast_rate = float(gp('fast_odom_rate_hz', 100.0))
        self.frame_id = gp('frame_id', 'odom')
        self.child_id = gp('child_frame_id', 'base_link')
        self.publish_cov = bool(gp('publish_covariance', True))
        self.R_bs = euler_deg_to_R_bs(self.rot_euler)

        # runtime state
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.state = WAIT_DATA
        self.imu_buf = []           # (t, ax, ay, az, gx, gy, gz), IMU time, raw accel
        self.radar_buf = []         # (t, pts(n,5)) raw Doppler, IMU time
        self.heading_buf = []       # (t, yaw)
        self.last_pose = None       # (t, p(3), q(4)) latest relay pose
        self.preunwrap = None
        self.rng = np.random.default_rng(0)
        self.t_prev = None          # last closed stride boundary
        self.last_radar_t = -1e18
        self.stat_t0 = None
        self.n_strides = 0
        self.n_rearms = 0
        self.iter_cap_run = 0
        self.starve_run = 0
        self.wd_flags = []
        self.max_lag_ms = 0.0
        self.prop = None            # dict(t, R, p, v, om) from the live edge
        self.last_fast_pub = 0.0
        self.last_cov = None

        P = self.NODE
        self.pub_odom = rospy.Publisher(P + '/odom', Odometry, queue_size=10)
        self.pub_fast = rospy.Publisher(P + '/odom_fast', Odometry, queue_size=50)
        self.pub_state = rospy.Publisher(P + '/state_cov', Float64MultiArray, queue_size=10)
        self.pub_status = rospy.Publisher(P + '/status', String, queue_size=10)
        rospy.Subscriber(self.topic_imu, Imu, self.cb_imu, queue_size=2000,
                         tcp_nodelay=True)
        rospy.Subscriber(self.topic_radar, PointCloud2, self.cb_radar,
                         queue_size=50)
        rospy.Subscriber(self.topic_pose, PoseStamped, self.cb_pose,
                         queue_size=200)
        if self.threaded:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
        rospy.loginfo('%s up: radar=%s imu=%s pose=%s v_max=%.3f threaded=%s',
                      P, self.topic_radar, self.topic_imu, self.topic_pose,
                      self.v_max, self.threaded)

    # ------------------------------------------------------------- inputs --
    def cb_imu(self, msg):
        t = msg.header.stamp.to_sec()
        a = msg.linear_acceleration
        g = msg.angular_velocity
        av = np.array([a.x, a.y, a.z]); gv = np.array([g.x, g.y, g.z])
        if self.R_mis is not None:
            av = self.R_mis @ av; gv = self.R_mis @ gv
        gx, gy, gz = (1.0 + self.gyro_scale) * gv
        ax, ay, az = av
        with self.lock:
            self.imu_buf.append((t, ax, ay, az, gx, gy, gz))
            if self.state == RUNNING and self.prop is not None:
                self._propagate_fast(t, np.array([ax, ay, az]),
                                     np.array([gx, gy, gz]))
        if self.state == WAIT_DATA:
            with self.lock:
                self.state = STATIONARY
                rospy.loginfo('%s: IMU streaming, waiting for %.1fs '
                              'stationary window', self.NODE, self.stat_sec)
        elif self.state == STATIONARY:
            # throttle the O(window) stationarity scan to ~10 Hz
            self._boot_ctr = getattr(self, '_boot_ctr', 0) + 1
            if self._boot_ctr % 100 == 0:
                self._try_bootstrap()
        elif self.state == RUNNING:
            if self.threaded:
                with self.cv:
                    self.cv.notify()
            else:
                self._try_stride()

    def cb_radar(self, msg):
        t = msg.header.stamp.to_sec() - self.radar_imu_offset
        pts = []
        for p in point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z', 'velocity', 'intensity'),
                skip_nans=True):
            pts.append(p)
        if not pts:
            return
        arr = np.asarray(pts, dtype=float).reshape(-1, 5)
        self.ingest_radar(t, arr)

    def ingest_radar(self, t, arr):
        with self.lock:
            # monotonic guard (stamp-frozen duplicate frames -> zero-dt knots)
            if t <= self.last_radar_t + 1e-6:
                return
            self.last_radar_t = t
            self.radar_buf.append((t, arr))

    def cb_pose(self, msg):
        t = msg.header.stamp.to_sec() - self.imu_mocap_offset
        q = msg.pose.orientation
        R = quat_to_R([q.x, q.y, q.z, q.w])
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        p = msg.pose.position
        with self.lock:
            self.heading_buf.append((t, yaw))
            self.last_pose = (t, np.array([p.x, p.y, p.z]),
                              np.array([q.x, q.y, q.z, q.w]))

    # ---------------------------------------------------------- bootstrap --
    def _try_bootstrap(self):
        with self.lock:
            if len(self.imu_buf) < 50:
                return
            arr = np.asarray(self.imu_buf[-4000:], float)
            t_hi = arr[-1, 0]
            win = arr[arr[:, 0] >= t_hi - self.stat_sec]
            if win.shape[0] < 20 or (win[-1, 0] - win[0, 0]) < \
                    self.stat_sec * 0.9:
                return
            gyr = win[:, 4:7]
            acc = win[:, 1:4]
            # motion check on the DEVIATION (bias-invariant) + a sane bias
            if (gyr - gyr.mean(axis=0)).std(axis=0).max() > self.stat_gyro \
                    or np.abs(gyr.mean(axis=0)).max() > 0.1 \
                    or acc.std(axis=0).max() > self.stat_acc_std:
                self.stat_t0 = None
                return
            # stationary confirmed: biases + gravity attitude
            bg = gyr.mean(axis=0)
            a_m = acc.mean(axis=0)
            a_n = float(np.linalg.norm(a_m))
            if abs(a_n - 9.81) > 1.5:
                rospy.logwarn_throttle(5.0, '%s: stationary but |a|=%.2f '
                                       '(not gravity?), waiting', self.NODE, a_n)
                return
            # gravity-aligned scalar accel-bias correction (sensor-only)
            ba = (a_n - 9.81) * (a_m / a_n)
            gb = a_m / a_n
            R0 = self._align_z(gb)
            # yaw from relay if present (else own frame, yaw=0)
            hp = [h for h in self.heading_buf if h[0] <= t_hi]
            if hp:
                dy = hp[-1][1] - np.arctan2(R0[1, 0], R0[0, 0])
                cz, sz = np.cos(dy), np.sin(dy)
                R0 = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]]) @ R0
            p0 = (self.last_pose[1].copy() if self.last_pose is not None
                  else np.zeros(3))
            self._arm(p0, R0, np.concatenate([ba, bg]), t_hi, anchor=None)
            self.preunwrap = PreUnwrapper(self.R_bs, self.v_max, ba, bg,
                                          R0, t_hi, self.min_range, alias_repair=self.alias_repair)
            self.t_prev = t_hi
            self.state = RUNNING
            rospy.loginfo('%s: ARMED at t=%.3f  bg=[%.4f %.4f %.4f] '
                          'ba~%.3f m/s^2  yaw=%s', self.NODE, t_hi, *bg,
                          np.linalg.norm(ba),
                          'relay' if hp else 'own-frame')

    @staticmethod
    def _align_z(gb):
        """Minimal rotation R with R @ gb = z (gravity attitude)."""
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(gb, z)
        c = float(np.dot(gb, z))
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        return so3_exp(v / np.linalg.norm(v)
                       * np.arctan2(np.linalg.norm(v), c))

    def _arm(self, p0, R0, biases, t_ref, anchor=None, v0=None):
        """Backend solver + the runtime bookkeeping of a (re-)arm.  Caller
        holds the lock."""
        self._backend_arm(p0, R0, biases, t_ref, anchor=anchor, v0=v0)
        # drop pre-init radar: the pop-based stride consumption would feed
        # every buffered pre-arm frame into the first stride (knots before
        # t_ref; polluted the slow-bag replay 0.19 -> 0.34)
        self.radar_buf = [f for f in self.radar_buf if f[0] > t_ref]
        self.prop = {'t': float(t_ref), 'R': np.asarray(R0, float).copy(),
                     'p': np.asarray(p0, float).copy(),
                     'v': (np.zeros(3) if v0 is None else np.asarray(v0, float).copy()),
                     'om': np.zeros(3)}

    # ------------------------------------------------------------- stride --
    def _worker_loop(self):
        while not rospy.is_shutdown():
            with self.cv:
                self.cv.wait(timeout=0.05)
            if self.state == RUNNING:
                try:
                    self._try_stride()
                except Exception as e:      # never let the worker die silently
                    rospy.logerr('%s: stride failed: %r', self.NODE, e)

    def _try_stride(self):
        # loop: several strides may be ready (backlog after a slow solve, or
        # faster-than-real-time bag replay)
        while self._close_one_stride():
            pass

    def _imu_block(self, t_lo, t_hi):
        """IMU rows in (t_lo, t_hi], accel re-sampled at t + accel_lag.
        Caller holds the lock."""
        blk = np.asarray([s for s in self.imu_buf if t_lo < s[0] <= t_hi], float)
        if blk.shape[0] and self.accel_lag != 0.0:
            buf = np.asarray(self.imu_buf, float)
            for k in range(3):
                blk[:, 1 + k] = np.interp(blk[:, 0] + self.accel_lag, buf[:, 0], buf[:, 1 + k])
        return blk

    def _close_one_stride(self):
        with self.lock:
            if self.t_prev is None or not self.imu_buf:
                return False
            t_now = self.t_prev + self.stride
            # watermark: radar stamps are shifted by -radar_imu_offset on
            # ingestion, so a frame's ARRIVAL trails its corrected stamp by
            # the offset -- the stride must wait that much longer or its
            # last frames are still in flight (found in bag replay: ~45%
            # of frames silently lost without this); + the accel lag
            wm = (t_now + self.watermark + max(0.0, self.radar_imu_offset)
                  + max(0.0, self.accel_lag))
            t_imu = self.imu_buf[-1][0]
            if t_imu < wm:
                return False
            # backlog guard (threaded live operation only): the solver chain
            # cannot catch up -> drop the stale strides, re-arm from the
            # propagated state at the newest closable boundary
            if self.threaded and t_imu - wm > self.max_backlog:
                n_skip = int((t_imu - wm) // self.stride)
                t_new = self.t_prev + n_skip * self.stride
                rospy.logwarn('%s: BACKLOG %.2fs > %.2fs: skipping %d strides, re-arm',
                              self.NODE, t_imu - wm, self.max_backlog, n_skip)
                self.radar_buf = [f for f in self.radar_buf if f[0] > t_new]
                self.t_prev = t_new
                self._rearm_locked(None, 0.0, t_new, t_new, reason='BACKLOG_REARM')
                return True
            imu_blk = self._imu_block(self.t_prev, t_now)
            # consume radar by POP (not interval filter): a straggler that
            # arrives after its stride closed is fed with the NEXT stride
            # (the solver accepts any frame newer than its last knot)
            radar_list = sorted([f for f in self.radar_buf if f[0] <= t_now],
                                key=lambda f: f[0])
            self.radar_buf = [f for f in self.radar_buf if f[0] > t_now]
            heading_blk = [(t, y) for (t, y) in self.heading_buf
                           if self.t_prev < t <= t_now]
            # GC old buffer entries (keep a lag of history for safety)
            keep = self.t_prev - 5.0
            self.imu_buf = [s for s in self.imu_buf if s[0] > keep]
            self.heading_buf = [h for h in self.heading_buf if h[0] > keep]
            t_prev = self.t_prev
            self.t_prev = t_now
            if imu_blk.shape[0] == 0:
                return True
            # 1. base pre-unwrap (IMU-anchored, per frame, causal) -- shared
            unwrapped = []
            for (t, pts) in radar_list:
                self.preunwrap.advance_imu(
                    imu_blk[(imu_blk[:, 0] > self.preunwrap.t_att)
                            & (imu_blk[:, 0] <= t)])
                out = self.preunwrap.unwrap_frame(pts, self.min_range)
                unwrapped.append((t, out, self.preunwrap.last_speed))
            self.preunwrap.advance_imu(
                imu_blk[imu_blk[:, 0] > self.preunwrap.t_att])
        # 2. backend solve OUTSIDE the lock (callbacks keep running)
        wall0 = _time.time()
        res = self._backend_stride(imu_blk, unwrapped, heading_blk, t_now)
        wall = _time.time() - wall0
        with self.lock:
            self.n_strides += 1
            self._watchdog_and_publish(res, imu_blk, t_now, wall, t_prev)
        return True

    # ----------------------------------------------------------- watchdog --
    def _watchdog_and_publish(self, res, imu_blk, t_now, wall, t_stride0):
        """Caller holds the lock."""
        flags = list(res.get('flags', []))
        wmax = float(np.linalg.norm(imu_blk[:, 4:7], axis=1).max())
        dt_s = float(res.get('dt_s', wall))
        if dt_s * 1000.0 > self.wd_ms_budget:
            flags.append('SLOW_SOLVE')
        if self.starve_run >= self.wd_starve_strides:
            flags.append('RADAR_STARVED')
        lag_ms = None
        try:
            now = rospy.get_time()
            if now > 1e9:            # real/sim clock running (not the replay stub)
                lag_ms = (now - float(res.get('t_pub', t_now))) * 1000.0
                self.max_lag_ms = max(self.max_lag_ms, lag_ms)
        except Exception:
            pass
        sample, cov9 = res.get('sample'), res.get('cov9')
        if res.get('hard_warm'):
            flags.append('STATE_NONFINITE')
            # re-init at the START of the failing stride (as the original node)
            self._rearm_locked(imu_blk, wmax, t_now, t_stride0, reason='STATE_NONFINITE')
        elif sample is None:
            flags.append('WARMUP')
        else:
            if cov9 is not None:
                if not np.all(np.isfinite(cov9)):
                    cov9, flags = None, flags + ['COV_NONFINITE']
                elif np.sqrt(max(0.0, np.trace(cov9[:3, :3]) / 3.0)) > self.wd_pos_sigma:
                    flags.append('POS_SIGMA_HIGH')
            self.last_cov = cov9
            self._publish(sample, res['bias'], cov9, float(res['t_pub']))
        self.wd_flags = flags
        st = {'t': t_now, 'state': self.state, 'strides': self.n_strides,
              'rearms': self.n_rearms, 'gn_iters': res.get('iters', 0),
              'ms': round(dt_s * 1000.0, 1), 'wall_ms': round(wall * 1000.0, 1),
              'wmax': round(wmax, 2), 'flags': flags}
        if lag_ms is not None:
            st['lag_ms'] = round(lag_ms, 1)
            st['max_lag_ms'] = round(self.max_lag_ms, 1)
        st.update(res.get('diag', {}))
        self.pub_status.publish(String(json.dumps(st)))

    def _rearm_locked(self, imu_blk, wmax, t_now, t_arm, reason=''):
        """In-air re-init (the validated gp_reset_t machinery). Prefers the
        carry attitude from the propagation state; relaxed anchor lets fresh
        gravity/heading evidence re-pull it (offline: transient ~5s
        mid-maneuver, near-baseline in quiet flight).  Caller holds the lock."""
        self.n_rearms += 1
        rospy.logwarn('%s: WATCHDOG re-arm #%d at t=%.2f (wmax=%.1f) %s',
                      self.NODE, self.n_rearms, t_now, wmax, reason)
        v0 = None
        if self.prop is not None and np.all(np.isfinite(self.prop['R'])) \
                and np.all(np.isfinite(self.prop['p'])):
            R0, p0 = self.prop['R'].copy(), self.prop['p'].copy()
            if np.all(np.isfinite(self.prop['v'])):
                v0 = self.prop['v'].copy()
        else:
            _ma = (imu_blk[:, 1:4].mean(axis=0) if imu_blk is not None and len(imu_blk)
                   else np.array([0.0, 0.0, 9.81]))
            _an = float(np.linalg.norm(_ma))
            R0 = self._align_z(_ma / _an) if abs(_an - 9.81) < 2.0 \
                else np.eye(3)
            hp = self.heading_buf[-1] if self.heading_buf else None
            if hp is not None:
                dy = hp[1] - np.arctan2(R0[1, 0], R0[0, 0])
                cz, sz = np.cos(dy), np.sin(dy)
                R0 = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]]) @ R0
            p0 = self.prop['p'] if self.prop is not None else np.zeros(3)
        bias = np.asarray(self._backend_biases(), float)
        if not np.all(np.isfinite(bias)):
            bias = np.zeros(6)
        self._arm(p0, R0, bias, float(t_arm), anchor=self.rearm_anchor,
                  v0=self._rearm_velocity(v0))

    def _rearm_velocity(self, v0):
        """GP re-arms at rest (its validated machinery); the spline carries the
        propagated velocity (its first-window pins hold it)."""
        return None

    # ---------------------------------------------------------- publishing --
    def _publish(self, sample, bias, cov9, t_pub):
        q, p, v, om = sample
        R = quat_to_R(q)
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(float(t_pub))
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_id
        msg.pose.pose.position.x, msg.pose.pose.position.y, \
            msg.pose.pose.position.z = p
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, \
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q
        v_body = R.T @ v
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, \
            msg.twist.twist.linear.z = v_body
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, \
            msg.twist.twist.angular.z = om
        if cov9 is not None:
            pc = np.zeros((6, 6))
            pc[:3, :3] = cov9[:3, :3]
            pc[3:, 3:] = cov9[3:6, 3:6]
            pc[:3, 3:] = cov9[:3, 3:6]
            pc[3:, :3] = cov9[3:6, :3]
            msg.pose.covariance = pc.flatten().tolist()
            tc = np.zeros((6, 6))
            tc[:3, :3] = R.T @ cov9[6:9, 6:9] @ R
            tc[3:, 3:] = np.eye(3) * self._twist_rate_var()
            msg.twist.covariance = tc.flatten().tolist()
        self.pub_odom.publish(msg)
        # reset the high-rate propagation from this live edge
        self.prop = {'t': float(t_pub), 'R': R, 'p': np.asarray(p, float).copy(),
                     'v': np.asarray(v, float).copy(), 'om': np.asarray(om, float).copy()}
        # full dump for rosbag analysis
        if cov9 is not None:
            arr = Float64MultiArray()
            arr.data = ([float(t_pub)] + list(q) + list(p) + list(v)
                        + list(om) + list(bias) + cov9.flatten().tolist())
            self.pub_state.publish(arr)

    def _twist_rate_var(self):
        return 4e-4

    def _propagate_fast(self, t, a, g):
        """IMU dead-reckon from the live edge (same math as the C++ knot
        seeding); publishes <node>/odom_fast decimated to fast_rate.  Uses the
        un-lagged accelerometer (the lag is a sub-stride correction for the
        solver; the fast output is a prediction)."""
        pr = self.prop
        dt = t - pr['t']
        if dt <= 0.0 or dt > 0.5:
            pr['t'] = t
            return
        bias = np.asarray(self._backend_biases(), float)
        om = g - bias[3:6]
        a_w = pr['R'] @ (a - bias[0:3]) + G_W
        pr['p'] = pr['p'] + pr['v'] * dt + 0.5 * a_w * dt * dt
        pr['v'] = pr['v'] + a_w * dt
        pr['R'] = pr['R'] @ so3_exp(om * dt)
        pr['om'] = om
        pr['t'] = t
        if t - self.last_fast_pub < 1.0 / max(self.fast_rate, 1e-3):
            return
        self.last_fast_pub = t
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(t)
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_id
        q = R_to_quat(pr['R'])
        msg.pose.pose.position.x, msg.pose.pose.position.y, \
            msg.pose.pose.position.z = pr['p']
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, \
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q
        vb = pr['R'].T @ pr['v']
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, \
            msg.twist.twist.linear.z = vb
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, \
            msg.twist.twist.angular.z = om
        if self.last_cov is not None:
            pc = np.zeros((6, 6))
            pc[:3, :3] = self.last_cov[:3, :3]
            pc[3:, 3:] = self.last_cov[3:6, 3:6]
            msg.pose.covariance = pc.flatten().tolist()
        self.pub_fast.publish(msg)

    # ------------------------------------------------------- backend hooks --
    def _backend_arm(self, p0, R0, biases, t0, anchor=None, v0=None):
        raise NotImplementedError

    def _backend_stride(self, imu_blk, unwrapped, heading_blk, t_now):
        raise NotImplementedError

    def _backend_biases(self):
        raise NotImplementedError


def run_node(name, cls, so_param, so_default=''):
    """main() for both nodes: optional import path for the container-built
    module, then spin."""
    import sys
    rospy.init_node(name)
    so_path = rospy.get_param('~' + so_param, so_default)
    if so_path:
        sys.path.insert(0, so_path)
    cls()
    rospy.spin()
