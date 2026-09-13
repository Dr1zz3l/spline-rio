#!/usr/bin/env python3
"""Convert our quadrotor bags into the topic layout expected by the
Doer/Trommer rio toolbox (ekf_rio / x_rio).

Faithful adapter: NO field math. The radar PointCloud2 payload is passed
through unchanged — rio's own pcl2msgToPcl() (rio_utils/radar_point_cloud.cpp)
natively consumes the ti_mmwave field layout (x,y,z,intensity,velocity;
extra fields are ignored by pcl::fromPCLPointCloud2) and applies its own
axis remap (x'=-y, y'=x: TI x-forward -> rio x-right/y-forward). The matching
extrinsics q_b_r therefore include Rz(-90 deg), computed by
gen_rio_calib.py — keep the two scripts in sync.

What this script DOES change:
  * topic names -> /sensor_platform/imu, /sensor_platform/radar/scan,
    /ground_truth/pose, /ground_truth/twist
  * ALL header stamps := bag record time. Our whole pipeline runs on record
    time (rosbag_loader uses t.to_sec() everywhere) because the IMU header
    stamps in these bags are a sample counter, NOT a clock (~1.007 s/msg,
    505809..564558 over a 59 s bag). All calibrated offsets live on the
    record-time axis.
  * radar header stamps additionally -= radar_imu_offset_sec (our calibrated
    USB/processing latency, analysis/config/extrinsics.yaml). This is
    calibration information, legitimately supplied to the baseline.

Usage (from repo root, analysis venv):
  .venv/bin/python3 baselines/adapters/convert_bag_to_rio.py slow_racing_best_velocity \
      [--out baselines/datasets/our_bags/<alias>_rio.bag]
"""
import argparse
import sys
from pathlib import Path

import yaml
from rosbags.rosbag1 import Reader, Writer

REPO = Path(__file__).resolve().parents[2]

TOPIC_MAP = {
    '/angrybird2/imu': '/sensor_platform/imu',
    '/mmWaveDataHdl/RScanVelocity': '/sensor_platform/radar/scan',
    '/mocap/angrybird2/pose': '/ground_truth/pose',
    '/mocap/angrybird2/twist': '/ground_truth/twist',
}
RADAR_TOPIC = '/mmWaveDataHdl/RScanVelocity'
IMU_TOPIC = '/angrybird2/imu'


def load_configs(alias: str):
    bags = yaml.safe_load((REPO / 'analysis/config/bags.yaml').read_text())
    ext = yaml.safe_load((REPO / 'analysis/config/extrinsics.yaml').read_text())
    bag_rel = bags['bags'][alias]
    overrides = (bags.get('extrinsics_overrides') or {}).get(alias, {})
    radar_imu_offset = overrides.get('radar_imu_offset_sec',
                                     ext['radar_imu_offset_sec'])
    # bags.yaml paths are relative to analysis/ (e.g. "rosbags/..." -> ../rosbags)
    bag_path = (REPO / 'analysis' / '..' / bag_rel).resolve()
    if not bag_path.exists():
        bag_path = (REPO / '..' / bag_rel).resolve()
    return bag_path, radar_imu_offset


def set_header_stamp(raw: bytes, t_ns: int) -> bytes:
    """Set the std_msgs/Header stamp at the start of a ROS1 message body.

    ROS1 serialization: uint32 seq, uint32 stamp.sec, uint32 stamp.nsec, ...
    """
    import struct
    seq, _, _ = struct.unpack_from('<III', raw, 0)
    return struct.pack('<III', seq, t_ns // 1_000_000_000,
                       t_ns % 1_000_000_000) + raw[12:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('alias', help='bag alias from analysis/config/bags.yaml')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    bag_path, radar_imu_offset = load_configs(args.alias)
    out = Path(args.out) if args.out else \
        REPO / 'baselines/datasets/our_bags' / f'{args.alias}_rio.bag'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    print(f'[convert] {bag_path.name} -> {out}')
    print(f'[convert] radar stamps shifted by -{radar_imu_offset}s onto the IMU clock')

    msgs = []  # (t_ns, out_topic, msgtype, raw)
    imu_idx = []
    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic in TOPIC_MAP]
        for conn, t_ns, raw in reader.messages(connections=conns):
            t_out = t_ns  # record time: the axis our whole pipeline runs on
            if conn.topic == RADAR_TOPIC:
                t_out = t_ns - int(radar_imu_offset * 1e9)
            if conn.topic == IMU_TOPIC:
                imu_idx.append(len(msgs))
            msgs.append((t_out, TOPIC_MAP[conn.topic], conn.msgtype, conn, raw))

    # IMU stamps: uniform grid between first and last record time. The FC
    # samples at a uniform ~1 kHz; record times carry USB batching jitter
    # (1%..99% dt = 0.02..2.1 ms) which rio's strapdown integrates as
    # omega*dt -> rectification drift with our vibration levels (gyro std
    # 0.5 rad/s). The uniform grid is the honest reconstruction of the
    # sampling clock; our own pipeline is insensitive to this jitter
    # (spline factors, not per-sample integration).
    if len(imu_idx) > 1:
        t_first = msgs[imu_idx[0]][0]
        t_last = msgs[imu_idx[-1]][0]
        for k, mi in enumerate(imu_idx):
            t_u = t_first + (t_last - t_first) * k // (len(imu_idx) - 1)
            t, topic, mt, conn, raw = msgs[mi]
            msgs[mi] = (t_u, topic, mt, conn, raw)

    msgs = [(t, topic, mt, conn, set_header_stamp(raw, t))
            for t, topic, mt, conn, raw in msgs]

    msgs.sort(key=lambda m: m[0])

    with Writer(out) as writer:
        wconns = {}
        for t_out, topic, msgtype, conn, raw in msgs:
            if topic not in wconns:
                msgdef = getattr(conn.msgdef, 'data', conn.msgdef)
                wconns[topic] = writer.add_connection(
                    topic, conn.msgtype,
                    msgdef=msgdef, md5sum=conn.digest)
            writer.write(wconns[topic], t_out, raw)

    counts = {}
    for _, topic, *_ in msgs:
        counts[topic] = counts.get(topic, 0) + 1
    for topic, n in sorted(counts.items()):
        print(f'[convert]   {topic:35s} {n}')
    print('[convert] done')


if __name__ == '__main__':
    sys.exit(main())
