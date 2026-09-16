#!/usr/bin/env python3
"""Convert a Doer ICINS-2021 bag (+pseudo-GT csv) into OUR pipeline's layout
so validate_live_solver.py can run on it unchanged.

Mappings:
  /sensor_platform/imu  -> /angrybird2/imu          (payload unchanged)
  /sensor_platform/radar/scan -> /mmWaveDataHdl/RScanVelocity with fields
      x,y,z       : their radar frame REMAPPED to our x-boresight convention
                    p_ours = Rz(-90deg) p_theirs (x=y', y=-x', z=z')
      velocity    : v_doppler_mps (same TI sign convention: + = receding)
      intensity   : snr_db
      noise       : noise_db
  pseudo GT csv -> /angrybird2/agiros_pilot/state (QuadState; our eval GT) and
                   /mocap/angrybird2/pose (PoseStamped)
      velocity = smoothed finite difference of csv positions
      angular_velocity = finite difference of csv quaternions

All record times := header stamps (their hardware clock is the good one and
our loader reads record time). Time offsets are zero for these bags (their
radar is hardware-triggered/stamped); add per-alias extrinsics_overrides with
zero offsets in bags.yaml.

The matching solver extrinsics (radar->body, OUR euler convention
R = Rz(yaw)Ry(pitch)Rx(roll)) are printed at the end:
  R_b_r_ours = R(q_b_r_theirs) . Rz(+90deg)

Usage:
  .venv/bin/python3 baselines/adapters/convert_icins_to_ours.py flight_1
  (alias resolves under baselines/datasets/icins2021/.../{flight,carried}_datasets)
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
ICINS = REPO / 'baselines/datasets/icins2021/radar_inertial_datasets_icins_2021'
OUT_DIR = REPO / 'baselines/datasets/our_format'

# msgdefs for the synthesized topics are copied from one of our real bags at
# conversion time (so md5/typedefs match what the loader expects).
DONOR_BAG = REPO / 'rosbags/Wed_11032026_1503/slowracing_oldconfig_2026-03-11-17-18-43.bag'

PC2_FIELDS = [('x', 7, 0), ('y', 7, 4), ('z', 7, 8),
              ('velocity', 7, 12), ('intensity', 7, 16), ('noise', 7, 20)]
POINT_STEP = 24


def read_their_radar(raw, typestore):
    msg = typestore.deserialize_ros1(raw, 'sensor_msgs/msg/PointCloud2')
    names = {f.name: (f.offset, f.datatype) for f in msg.fields}
    n = msg.width * msg.height
    step = msg.point_step
    data = bytes(msg.data)
    out = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        base = i * step
        for j, key in enumerate(['x', 'y', 'z', 'v_doppler_mps', 'snr_db', 'noise_db']):
            off, dt = names[key]
            out[i, j] = struct.unpack_from('<f', data, base + off)[0]
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return t, out


def build_our_radar_cloud(t_ns, points_theirs, typestore):
    """points_theirs: (n,6) x,y,z,v,snr,noise in THEIR radar frame."""
    # remap into our x-boresight convention: x=y', y=-x', z=z'
    P = points_theirs
    n = len(P)
    buf = bytearray(n * POINT_STEP)
    for i in range(n):
        struct.pack_into('<6f', buf, i * POINT_STEP,
                         P[i, 1], -P[i, 0], P[i, 2], P[i, 3], P[i, 4], P[i, 5])
    Pointfield = typestore.types['sensor_msgs/msg/PointField']
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']
    PC2 = typestore.types['sensor_msgs/msg/PointCloud2']
    fields = [Pointfield(name=nm, offset=off, datatype=dt, count=1)
              for nm, dt, off in PC2_FIELDS]
    hdr = Header(seq=0, stamp=Time(sec=int(t_ns // 1_000_000_000),
                                   nanosec=int(t_ns % 1_000_000_000)),
                 frame_id='ti_mmwave')
    msg = PC2(header=hdr, height=1, width=n, fields=fields,
              is_bigendian=False, point_step=POINT_STEP,
              row_step=n * POINT_STEP,
              data=np.frombuffer(bytes(buf), dtype=np.uint8),
              is_dense=True)
    return typestore.serialize_ros1(msg, 'sensor_msgs/msg/PointCloud2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', help='e.g. flight_1 or carried_3')
    ap.add_argument('--elev-gate-deg', type=float, default=None,
                    help='drop points with |elevation above radar horizon| > this '
                         '(mirrors reve min/elevation preprocessing; their ekf_rio '
                         'config uses 60)')
    args = ap.parse_args()

    sub = 'flight_datasets' if args.name.startswith('flight') else 'carried_datasets'
    bag_in = ICINS / sub / f'{args.name}.bag'
    gt_csv = ICINS / sub / f'{args.name}_pseudo_ground_truth.csv'
    calib = yaml.safe_load((ICINS / sub / 'calib.yaml').read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bag_out = OUT_DIR / f'icins_{args.name}.bag'
    if bag_out.exists():
        bag_out.unlink()

    typestore = get_typestore(Stores.ROS1_NOETIC)

    # --- donor msgdefs for QuadState / PoseStamped --------------------------
    donor = {}
    with Reader(DONOR_BAG) as r:
        for c in r.connections:
            if c.topic in ('/angrybird2/agiros_pilot/state', '/mocap/angrybird2/pose'):
                donor[c.topic] = (c.msgtype, c.msgdef, c.digest)
        # also register QuadState type for serialization
        from rosbags.typesys.msg import get_types_from_msg
        qs = donor['/angrybird2/agiros_pilot/state']
        typestore.register(get_types_from_msg(
            qs[1].data if hasattr(qs[1], 'data') else qs[1], qs[0]))

    # --- GT from csv ---------------------------------------------------------
    G = np.loadtxt(gt_csv, skiprows=1)
    gt_t = G[:, 0]
    gt_p = G[:, 1:4]
    gt_q = G[:, 4:8]  # x y z w
    # velocity: central differences, then light smoothing
    v = np.gradient(gt_p, gt_t, axis=0)
    from scipy.signal import savgol_filter
    win = min(11, len(v) - (1 - len(v) % 2))
    if win >= 5:
        v = savgol_filter(v, win, 2, axis=0)
    # angular velocity from quaternion finite differences (body frame)
    rots = R.from_quat(gt_q)
    w = np.zeros_like(gt_p)
    dts = np.diff(gt_t)
    drot = (rots[:-1].inv() * rots[1:]).as_rotvec()
    w[:-1] = drot / dts[:, None]
    w[-1] = w[-2]

    # --- write ---------------------------------------------------------------
    msgs = []
    with Reader(bag_in) as r:
        conns = [c for c in r.connections
                 if c.topic in ('/sensor_platform/imu', '/sensor_platform/radar/scan')]
        for c, t_ns, raw in r.messages(connections=conns):
            # header stamp = authoritative clock; use as record time
            seq, sec, nsec = struct.unpack_from('<III', raw, 0)
            t_out = sec * 1_000_000_000 + nsec
            if c.topic == '/sensor_platform/imu':
                msgs.append((t_out, '/angrybird2/imu', 'sensor_msgs/msg/Imu',
                             c.msgdef, c.digest, raw))
            else:
                _, pts = read_their_radar(raw, typestore)
                if args.elev_gate_deg is not None and len(pts):
                    rho = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
                    elev = np.degrees(np.abs(np.arctan2(pts[:, 2], rho)))
                    pts = pts[elev < args.elev_gate_deg]
                raw2 = build_our_radar_cloud(t_out, pts, typestore)
                msgs.append((t_out, '/mmWaveDataHdl/RScanVelocity',
                             'sensor_msgs/msg/PointCloud2', c.msgdef, c.digest, raw2))

    QuadState = donor['/angrybird2/agiros_pilot/state']
    Pose = donor['/mocap/angrybird2/pose']
    qs_type = QuadState[0]
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']

    def mk_time(t):
        return Time(sec=int(t), nanosec=int((t % 1.0) * 1e9))

    QS = typestore.types[qs_type]
    PoseT = typestore.types['geometry_msgs/msg/PoseStamped']
    Point = typestore.types['geometry_msgs/msg/Point']
    Quat = typestore.types['geometry_msgs/msg/Quaternion']
    Vec3 = typestore.types['geometry_msgs/msg/Vector3']
    PoseG = typestore.types['geometry_msgs/msg/Pose']

    qs_fields = {f for f in QS.__dataclass_fields__}  # introspect required fields

    for k in range(len(gt_t)):
        t = gt_t[k]
        t_ns = int(t * 1e9)
        hdr = Header(seq=k, stamp=mk_time(t), frame_id='world')
        pose = PoseG(position=Point(x=gt_p[k, 0], y=gt_p[k, 1], z=gt_p[k, 2]),
                     orientation=Quat(x=gt_q[k, 0], y=gt_q[k, 1],
                                      z=gt_q[k, 2], w=gt_q[k, 3]))
        pmsg = PoseT(header=hdr, pose=pose)
        msgs.append((t_ns, '/mocap/angrybird2/pose', 'geometry_msgs/msg/PoseStamped',
                     Pose[1], Pose[2],
                     typestore.serialize_ros1(pmsg, 'geometry_msgs/msg/PoseStamped')))
        Twist = typestore.types['geometry_msgs/msg/Twist']
        zero3 = Vec3(x=0.0, y=0.0, z=0.0)
        zero_twist = Twist(linear=zero3, angular=zero3)
        qmsg = QS(header=hdr,
                  t=float(t - gt_t[0]),
                  pose=pose,
                  velocity=Twist(
                      linear=Vec3(x=v[k, 0], y=v[k, 1], z=v[k, 2]),
                      angular=Vec3(x=w[k, 0], y=w[k, 1], z=w[k, 2])),
                  acceleration=zero_twist,
                  acc_bias=zero3, gyr_bias=zero3,
                  jerk=zero3, snap=zero3,
                  motors=np.zeros(4))
        msgs.append((t_ns, '/angrybird2/agiros_pilot/state', qs_type,
                     QuadState[1], QuadState[2],
                     typestore.serialize_ros1(qmsg, qs_type)))

    msgs.sort(key=lambda m: m[0])
    with Writer(bag_out) as w_:
        wconns = {}
        for t_ns, topic, msgtype, msgdef, digest, raw in msgs:
            if topic not in wconns:
                md = msgdef.data if hasattr(msgdef, 'data') else msgdef
                wconns[topic] = w_.add_connection(topic, msgtype,
                                                  msgdef=md, md5sum=digest)
            w_.write(wconns[topic], t_ns, raw)

    counts = {}
    for _, topic, *_ in msgs:
        counts[topic] = counts.get(topic, 0) + 1
    for topic, n in sorted(counts.items()):
        print(f'  {topic:40s} {n}')

    # --- matching solver extrinsics ------------------------------------------
    q = [calib['q_b_r_x'], calib['q_b_r_y'], calib['q_b_r_z'], calib['q_b_r_w']]
    R_ours = R.from_quat(q) * R.from_euler('z', 90, degrees=True)
    yaw, pitch, roll = R_ours.as_euler('ZYX', degrees=True)
    l = [calib['l_b_r_x'], calib['l_b_r_y'], calib['l_b_r_z']]
    print(f'\nwrote {bag_out}')
    print(f"solver extrinsics for this bag (radar->body, our ZYX convention):")
    print(f"  --set-ext 'rotation_euler_deg=[{roll:.3f},{pitch:.3f},{yaw:.3f}]' "
          f"--set-ext 'translation_body_m=[{l[0]},{l[1]},{l[2]}]'")


if __name__ == '__main__':
    sys.exit(main())
