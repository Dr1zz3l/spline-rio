#!/usr/bin/env python3
"""Offline replay test for rio_gp_node: stub rospy, feed a real bag through
the node's callbacks in arrival order (raw stamps), score published odometry
against MoCap. Expected on slow_racing: pos RMSE ~0.19 m (harness parity)."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import json

import os
# RIO_REPO: the checkout holding the node + analysis libs (a worktree works);
# RIO_BUILD: the build dir with rio_gp.so; RIO_DATA: the checkout holding rosbags/
REPO = Path(os.environ.get('RIO_REPO', '/home/mouse/MyData/radar-iwr6843-driver'))
BUILD = Path(os.environ.get('RIO_BUILD', str(REPO / 'rio_solver_cpp' / 'build_release')))
DATA = Path(os.environ.get('RIO_DATA', str(REPO)))
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'analysis' / 'lib'))
sys.path.insert(0, str(BUILD))

BAG = sys.argv[1] if len(sys.argv) > 1 else 'slow_racing_best_velocity'
import yaml
bags_cfg = yaml.safe_load(open(REPO / 'analysis/config/bags.yaml'))
bag_path = DATA / bags_cfg['bags'][BAG]
win = bags_cfg['timing'][BAG]

# load the bag BEFORE stubbing ros modules (the real rosbag lib needs them)
from rosbag_loader.loader import load_bag_topics
bag = load_bag_topics(str(bag_path), verbose=False)

# ---------------------------------------------------------------- stubs ---
class _NS(types.SimpleNamespace):
    pass


def _vec3():
    return _NS(x=0.0, y=0.0, z=0.0)


def _quat():
    return _NS(x=0.0, y=0.0, z=0.0, w=1.0)


class _Stamp:
    def __init__(self, t):
        self._t = t

    def to_sec(self):
        return self._t


class _Odometry:
    def __init__(self):
        self.header = _NS(stamp=None, frame_id='')
        self.child_frame_id = ''
        self.pose = _NS(pose=_NS(position=_vec3(), orientation=_quat()),
                        covariance=[0.0] * 36)
        self.twist = _NS(twist=_NS(linear=_vec3(), angular=_vec3()),
                         covariance=[0.0] * 36)


class _F64MA:
    def __init__(self):
        self.data = []


class _String:
    def __init__(self, data=''):
        self.data = data


PUBS = {}


class _Publisher:
    def __init__(self, topic, *_a, **_k):
        self.topic = topic
        PUBS.setdefault(topic, [])

    def publish(self, msg):
        PUBS[self.topic].append(msg)


rospy_stub = types.ModuleType('rospy')
_params = {
    '~rotation_euler_deg': [180.0, 27.5, 0.0],
    '~translation_body_m': [0.08, 0.02, -0.01],
    '~radar_imu_offset_sec': 0.135,
    '~imu_mocap_offset_sec': 0.015,
    '~doppler_v_max': 3.136,
    '~threaded': False,          # deterministic: strides close inside the IMU callback
}
# RIO_REPLAY_REALTIME=<speed>: wall-clock-paced replay (2026-10-13 real-time
# test): events are fed at bag speed x <speed>, the node runs THREADED (its
# solver worker thread), rospy.get_time() returns the bag clock of the newest
# fed event, so status 'lag_ms' = data stamp -> publish latency.
RT = float(os.environ.get('RIO_REPLAY_REALTIME', '0') or 0)
if RT > 0:
    _params['~threaded'] = True
CLOCK = [0.0]
# RIO_NODE_PROFILE: ':'-separated yaml files loaded as node params (the
# spline node's generated profile, 2026-10-13); RIO_NODE_PARAMS overrides them
for _yf in [f for f in os.environ.get('RIO_NODE_PROFILE', '').split(':') if f]:
    for _k, _v in (yaml.safe_load(open(_yf)) or {}).items():
        _params['~' + _k] = _v
_params['~threaded'] = RT > 0     # the replay decides (a loaded params file says true)
# RIO_NODE_PARAMS: JSON dict of '~key' overrides (2026-09-17: gyro_scale, alias_repair, yaw, clocks)
import json as _json
_params.update(_json.loads(os.environ.get('RIO_NODE_PARAMS', '{}')))
print('[replay] node params:', {k: v for k, v in _params.items() if k not in ('~translation_body_m',)})
rospy_stub.get_param = lambda k, d=None: _params.get(k, d)
rospy_stub.Publisher = _Publisher
rospy_stub.Subscriber = lambda *a, **k: None
rospy_stub.loginfo = lambda *a: print('[info]', a[0] % tuple(a[1:]) if len(a) > 1 else a[0])
rospy_stub.logwarn = lambda *a: print('[warn]', a[0] % tuple(a[1:]) if len(a) > 1 else a[0])
rospy_stub.logwarn_throttle = lambda _t, *a: print('[warn]', a[0] % tuple(a[1:]) if len(a) > 1 else a[0])
rospy_stub.init_node = lambda *a, **k: None
rospy_stub.spin = lambda: None
rospy_stub.Time = _NS(from_sec=lambda t: _Stamp(t))
rospy_stub.logerr = lambda *a: print('[error]', a[0] % tuple(a[1:]) if len(a) > 1 else a[0])
rospy_stub.is_shutdown = lambda: False
rospy_stub.get_time = lambda: CLOCK[0]
sys.modules['rospy'] = rospy_stub

for name, attrs in [
        ('geometry_msgs', {}), ('nav_msgs', {}), ('std_msgs', {}),
        ('sensor_msgs', {})]:
    sys.modules.setdefault(name, types.ModuleType(name))
gm = types.ModuleType('geometry_msgs.msg'); gm.PoseStamped = object
sys.modules['geometry_msgs.msg'] = gm
nm = types.ModuleType('nav_msgs.msg'); nm.Odometry = _Odometry
sys.modules['nav_msgs.msg'] = nm
sm = types.ModuleType('std_msgs.msg')
sm.Float64MultiArray, sm.String = _F64MA, _String
sys.modules['std_msgs.msg'] = sm
smm = types.ModuleType('sensor_msgs.msg')
smm.Imu, smm.PointCloud2 = object, object
sys.modules['sensor_msgs.msg'] = smm
spc = types.ModuleType('sensor_msgs.point_cloud2')
spc.read_points = lambda *a, **k: []
sys.modules['sensor_msgs.point_cloud2'] = spc
sys.modules['sensor_msgs'].point_cloud2 = spc

# ------------------------------------------------------- import the node ---
SCRIPTS = REPO / 'mmwave_ti_ros/ros1_driver/src/rio_ros/scripts'
sys.path.insert(0, str(SCRIPTS))   # rio_frontend / rio_runtime (shared, 2026-10-13)
# RIO_NODE: spline (default) | gp
NODE_KIND = os.environ.get('RIO_NODE', 'spline')
_node_file, _node_cls, PFX = {'gp': ('rio_gp_node', 'RioGpNode', 'rio_gp'),
                              'spline': ('rio_spline_node', 'RioSplineNode', 'rio_spline')}[NODE_KIND]
node_py = SCRIPTS / f'{_node_file}.py'
spec = importlib.util.spec_from_file_location(_node_file, node_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

node = getattr(mod, _node_cls)()

# ------------------------------------------------------------- load bag ---
t0b = bag.start_time
print(f"bag {BAG}: {len(bag.imu_data)} imu, {len(bag.radar_velocity)} radar, "
      f"{len(bag.mocap_pose)} mocap; flight window rel [{win[0]}, "
      f"{win[0]+win[1]}]")

events = []
for d in bag.imu_data:
    events.append((d.timestamp, 'imu', d))
for f in bag.radar_velocity:
    if f.positions is not None and len(f.positions):
        events.append((f.timestamp, 'radar', f))
for m in bag.mocap_pose:
    events.append((m.timestamp, 'pose', m))
events.sort(key=lambda e: e[0])

import time as _wt
_wall0, _tb0 = _wt.time(), events[0][0]
for t, kind, obj in events:
    if RT > 0:
        _dt = _wall0 + (t - _tb0) / RT - _wt.time()
        if _dt > 0:
            _wt.sleep(_dt)
        CLOCK[0] = t
    if kind == 'imu':
        a, g = obj.linear_acceleration, obj.angular_velocity
        msg = _NS(header=_NS(stamp=_Stamp(obj.timestamp)),
                  linear_acceleration=_NS(x=a[0], y=a[1], z=a[2]),
                  angular_velocity=_NS(x=g[0], y=g[1], z=g[2]))
        node.cb_imu(msg)
    elif kind == 'radar':
        n = obj.num_points() if hasattr(obj, 'num_points') else len(obj.positions)
        if n == 0:
            continue
        pts = np.zeros((n, 5))
        pts[:, :3] = obj.positions[:n]
        pts[:, 3] = obj.velocities[:n] if obj.velocities is not None else 0.0
        pts[:, 4] = obj.intensities[:n] if obj.intensities is not None else 0.0
        tc = obj.timestamp - node.radar_imu_offset
        with node.lock:
            if tc > node.last_radar_t + 1e-6:
                node.last_radar_t = tc
                node.radar_buf.append((tc, pts))
    else:
        q, p = obj.orientation, obj.position
        msg = _NS(header=_NS(stamp=_Stamp(obj.timestamp)),
                  pose=_NS(position=_NS(x=p[0], y=p[1], z=p[2]),
                           orientation=_NS(x=q[0], y=q[1], z=q[2], w=q[3])))
        node.cb_pose(msg)

if RT > 0:
    _wt.sleep(3.0)               # let the worker drain the last strides
    _stl = [json.loads(x.data) for x in PUBS.get(PFX + '/status', [])]
    _lag = np.array([x['lag_ms'] for x in _stl if 'lag_ms' in x])
    _wall = np.array([x['wall_ms'] for x in _stl])
    print(f"[realtime x{RT:g}] strides {len(_stl)}, latency data->publish ms: median {np.median(_lag):.0f} "
          f"p95 {np.percentile(_lag, 95):.0f} max {_lag.max():.0f}; stride wall ms: median {np.median(_wall):.0f} "
          f"p95 {np.percentile(_wall, 95):.0f} max {_wall.max():.0f} (budget {1000*node.stride:.0f}); "
          f"rearms {_stl[-1]['rearms']}; flags {sorted(set(f for x in _stl for f in x['flags']))}")

# ------------------------------------------------------------------ eval ---
odo = PUBS.get(PFX + '/odom', [])
print(f"published: {len(odo)} odom, "
      f"{len(PUBS.get(PFX + '/odom_fast', []))} odom_fast, "
      f"{len(PUBS.get(PFX + '/state_cov', []))} state_cov")
if not odo:
    sys.exit('NO ODOMETRY PUBLISHED')

mc_t = np.array([m.timestamp - 0.015 for m in bag.mocap_pose])
mc_p = np.array([m.position for m in bag.mocap_pose])
lo, hi = t0b + win[0], t0b + win[0] + win[1]
errs, ts = [], []
for msg in odo:
    t = msg.header.stamp.to_sec()
    if not (lo <= t <= hi) or t < mc_t[0] or t > mc_t[-1]:
        continue
    p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                  msg.pose.pose.position.z])
    gt = np.array([np.interp(t, mc_t, mc_p[:, i]) for i in range(3)])
    errs.append(np.linalg.norm(p - gt))
    ts.append(t - t0b)
errs, ts = np.array(errs), np.array(ts)
print(f"\nflight-window live-edge samples: {len(errs)}")
print(f"pos AE (absolute, incl. pre-window drift + arm yaw gauge):")
print(f"  rmse {np.sqrt((errs**2).mean()):.3f}  mean {errs.mean():.3f}"
      f"  max {errs.max():.3f}")
# harness-commensurable: start-anchored alignment at the window start
# (remove the error vector at window entry, the way the harness aligns)
vecs = []
for msg in odo:
    t = msg.header.stamp.to_sec()
    if not (lo <= t <= hi) or t < mc_t[0] or t > mc_t[-1]:
        continue
    p = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                  msg.pose.pose.position.z])
    gt = np.array([np.interp(t, mc_t, mc_p[:, i]) for i in range(3)])
    vecs.append(p - gt)
vecs = np.array(vecs)
rel = np.linalg.norm(vecs - vecs[0], axis=1)
print(f"pos AE (start-anchored at window entry, ~harness metric):")
print(f"  rmse {np.sqrt((rel**2).mean()):.3f}  mean {rel.mean():.3f}"
      f"  max {rel.max():.3f}  (harness live ~0.19 slow)")
# velocity / orientation / split (2026-10-13): world velocity from the body
# twist, orientation geodesic vs MoCap, horizontal / vertical of the anchored error
from scipy.spatial.transform import Rotation as _Rot, Slerp as _Slerp
_mq = np.array([m.orientation for m in bag.mocap_pose]); _ok = np.r_[True, np.diff(mc_t) > 1e-6]
_sl = _Slerp(mc_t[_ok], _Rot.from_quat(_mq[_ok]))
# reference velocity: the loader's pose-derived reference (agiros_state is
# re-derived from the pose, zero-phase filtered; raw MoCap differences glitch)
_st = bag.agiros_state
_stt = np.array([x.timestamp for x in _st]) - 0.005
_stv = np.array([x.velocity for x in _st])
_mv = np.column_stack([np.interp(mc_t, _stt, _stv[:, i]) for i in range(3)])
ve, oe = [], []
for msg in odo:
    t = msg.header.stamp.to_sec()
    if not (lo <= t <= hi) or t < mc_t[0] or t > mc_t[-1]:
        continue
    q = msg.pose.pose.orientation; R = _Rot.from_quat([q.x, q.y, q.z, q.w])
    tw = msg.twist.twist.linear; vw = R.apply([tw.x, tw.y, tw.z])
    vg = np.array([np.interp(t, mc_t, _mv[:, i]) for i in range(3)])
    ve.append(np.linalg.norm(vw - vg))
    oe.append(np.degrees((R.inv() * _sl([t])[0]).magnitude()))
ve, oe = np.array(ve), np.array(oe)
_d = vecs - vecs[0]
print(f"vel rmse {np.sqrt((ve**2).mean()):.3f} m/s   ori rmse {np.sqrt((oe**2).mean()):.2f} deg (absolute, incl. arm yaw)"
      f"   anchored pos hor/ver rmse {np.sqrt((np.sum(_d[:, :2]**2, 1)).mean()):.3f} / {np.sqrt((_d[:, 2]**2).mean()):.3f}")
print("error-vs-time buckets (absolute | start-anchored):")
for lo2 in np.arange(ts.min(), ts.max(), 5.0):
    m = (ts >= lo2) & (ts < lo2 + 5.0)
    if m.any():
        print(f"  t=[{lo2:5.1f},{lo2+5:5.1f}): {errs[m].mean():.3f} | "
              f"{rel[m].mean():.3f}   vel {ve[m].mean():.3f}  ori {oe[m].mean():.2f}")
covs = PUBS.get(PFX + '/state_cov', [])
if covs:
    S = np.array(covs[-1].data[20:]).reshape(9, 9)
    print(f"last Sigma9 diag sqrt: pos {np.sqrt(S[0,0]):.3f} "
          f"{np.sqrt(S[1,1]):.3f} {np.sqrt(S[2,2]):.3f}  "
          f"ori {np.degrees(np.sqrt(S[3,3])):.2f} deg  "
          f"vel {np.sqrt(S[6,6]):.3f}")
st = PUBS.get(PFX + '/status', [])
if st:
    import json
    last = json.loads(st[-1].data)
    print(f"last status: {last}")
    # per-stride diagnosis: velocity error + front-end counts over time
    print("\nper-stride diag (t_rel | v_err | frames pts n_re reuw gn "
          "wmax flags):")
    mc_v = np.gradient(mc_p, mc_t, axis=0)
    sc = PUBS.get(PFX + '/state_cov', [])
    stat = [json.loads(s.data) for s in st]
    for i, s in enumerate(stat):
        if i % 10 and 'divg' not in sys.argv:
            pass
    # align state_cov entries (one per publish) with status entries
    k = 0
    for i, s in enumerate(stat):
        if i % 7:
            continue
        tr = s['t'] - t0b - 0.3
        verr = float('nan')
        if k < len(sc):
            pass
        # find the state_cov whose t is closest to this status stride
        best = None
        for e in sc:
            if abs(e.data[0] - (s['t'])) < 0.35:
                best = e
        if best is not None and mc_t[0] < best.data[0] < mc_t[-1]:
            v_est = np.array(best.data[8:11])
            v_gt = np.array([np.interp(best.data[0], mc_t, mc_v[:, j])
                             for j in range(3)])
            verr = float(np.linalg.norm(v_est - v_gt))
        print(f"  {tr:6.1f} | {verr:5.2f} | {s.get('frames','-'):>2} "
              f"{s.get('pts','-'):>3} {s.get('n_re','-'):>3} "
              f"{int(s.get('reuw', False))} {s['gn_iters']} "
              f"{s['wmax']:5.1f} {','.join(s['flags'])}")

# RIO_REPLAY_DUMP: npz of the published live edge + status JSON (2026-10-13)
if os.environ.get('RIO_REPLAY_DUMP'):
    _o = [(m.header.stamp.to_sec(), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
           m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w,
           m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z) for m in odo]
    np.savez(os.environ['RIO_REPLAY_DUMP'], odom=np.array(_o), status=np.array([x.data for x in PUBS.get(PFX + '/status', [])]),
             mc_t=mc_t, mc_p=mc_p, mc_q=np.array([m.orientation for m in bag.mocap_pose]), ref_t=_stt, ref_v=_stv, win=np.array([lo, hi]))
    print('dumped', os.environ['RIO_REPLAY_DUMP'])
