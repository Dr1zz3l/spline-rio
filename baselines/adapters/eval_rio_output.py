#!/usr/bin/env python3
"""Evaluate a recorded rio (ekf_rio / x_rio) output bag against our MoCap GT
with EXACTLY the metrics of analysis/validate_live_solver.py (lines ~1980-2210):
  * eval window  = bags.yaml timing [start_offset, start_offset+duration-3s]
  * GT           = /angrybird2/agiros_pilot/state samples in the window
                   (near-duplicate stamps <1ms removed), velocity lowpassed
                   with the same 4th-order 10 Hz Butterworth
  * alignment    = single constant SE(3) at the FIRST eval sample
                   (R_align = R_gt0 @ R_est0^T) -- start-anchored
  * metrics      = pos RMSE (norm), vel RMSE (norm), ori RMSE (geodesic deg),
                   drift % = 100 * pos_rmse / GT path length
The rio estimate is causal/live by construction (forward EKF), so these
numbers compare against OUR "live" columns.

Time axes: rio outputs stamps on the IMU clock; our eval shifts IMU stamps by
+imu_mocap_offset_sec onto the MoCap/GT axis -- we apply the same shift here.

Usage (repo root):
  .venv/bin/python3 baselines/adapters/eval_rio_output.py \
      <alias> baselines/results/slow_racing_ekf_rio.bag [--filter-topic /ekf_rio]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'analysis' / 'lib'))

from rosbag_loader import load_bag_topics  # noqa: E402
from rosbags.rosbag1 import Reader  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402


def load_rio_estimate(result_bag: Path, prefix: str):
    ts = get_typestore(Stores.ROS1_NOETIC)
    pose, twist = [], []
    with Reader(result_bag) as r:
        conns = [c for c in r.connections
                 if c.topic in (f'{prefix}/pose', f'{prefix}/twist')]
        for c, _, raw in r.messages(connections=conns):
            m = ts.deserialize_ros1(raw, c.msgtype)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if c.topic.endswith('/pose'):
                o = m.pose.orientation
                pose.append([t, m.pose.position.x, m.pose.position.y,
                             m.pose.position.z, o.x, o.y, o.z, o.w])
            else:
                twist.append([t, m.twist.linear.x, m.twist.linear.y,
                              m.twist.linear.z])
    return np.array(pose), np.array(twist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('alias')
    ap.add_argument('result_bag')
    ap.add_argument('--filter-topic', default='/ekf_rio')
    args = ap.parse_args()

    bags = yaml.safe_load((REPO / 'analysis/config/bags.yaml').read_text())
    ext = yaml.safe_load((REPO / 'analysis/config/extrinsics.yaml').read_text())
    overrides = (bags.get('extrinsics_overrides') or {}).get(args.alias, {})
    imu_mocap_offset = overrides.get('imu_mocap_offset_sec',
                                     ext['imu_mocap_offset_sec'])
    start_offset, duration = bags['timing'][args.alias]

    bag_rel = bags['bags'][args.alias]
    orig_bag = (REPO / bag_rel).resolve()
    if not orig_bag.exists():
        orig_bag = (REPO / '..' / bag_rel).resolve()

    pose, twist = load_rio_estimate(Path(args.result_bag), args.filter_topic)
    if len(pose) < 10:
        print(f'ERROR: only {len(pose)} pose msgs in {args.result_bag}')
        return 1
    # rio stamps (IMU clock) -> MoCap/GT axis, same shift as our eval
    pose[:, 0] += imu_mocap_offset
    twist[:, 0] += imu_mocap_offset
    # publisher can emit duplicate stamps (filter-time gated); keep first
    keep = np.concatenate([[True], np.diff(pose[:, 0]) > 0])
    pose = pose[keep]
    keep = np.concatenate([[True], np.diff(twist[:, 0]) > 0])
    twist = twist[keep]
    print(f'[eval] rio estimate: {len(pose)} poses, {len(twist)} twists, '
          f'span [{pose[0,0]:.2f}, {pose[-1,0]:.2f}]')

    bag_data = load_bag_topics(str(orig_bag), verbose=False)
    t0 = bag_data.start_time + start_offset
    t1 = t0 + duration
    t1_eval = t1 - 3.0          # same 3s tail trim as our eval
    states = [s for s in bag_data.agiros_state if t0 <= s.timestamp <= t1_eval]
    # near-duplicate stamp filter (same as our eval)
    filt = [states[0]]
    for s in states[1:]:
        if s.timestamp - filt[-1].timestamp >= 1e-3:
            filt.append(s)
    states = filt

    gt_t = np.array([s.timestamp for s in states])
    # clamp to rio output span
    m = (gt_t >= pose[0, 0]) & (gt_t <= pose[-1, 0]) & \
        (gt_t >= twist[0, 0]) & (gt_t <= twist[-1, 0])
    states = [s for s, k in zip(states, m) if k]
    gt_t = gt_t[m]
    gt_pos = np.array([s.position for s in states])
    gt_vel = np.array([s.velocity for s in states])
    # loader orientation is [x,y,z,w] (radar_velocity_utils.quat_to_rotation_matrix)
    gt_rot = Rotation.from_quat(np.array([s.orientation for s in states]))

    print(f'[eval] window [{gt_t[0]-bag_data.start_time:.1f}, '
          f'{gt_t[-1]-bag_data.start_time:.1f}]s rel bag start, '
          f'{len(gt_t)} GT samples')

    # GT velocity lowpass exactly like our eval
    dt = np.median(np.diff(gt_t))
    fs = 1.0 / dt
    fc = min(10.0, fs * 0.4)
    b, a = butter(4, fc / (fs / 2), btype='low')
    if len(gt_vel) > 27:
        for d in range(3):
            gt_vel[:, d] = filtfilt(b, a, gt_vel[:, d])

    # interpolate rio estimate at GT times
    est_pos = np.column_stack([np.interp(gt_t, pose[:, 0], pose[:, 1 + d])
                               for d in range(3)])
    est_vel = np.column_stack([np.interp(gt_t, twist[:, 0], twist[:, 1 + d])
                               for d in range(3)])
    slerp = Slerp(pose[:, 0], Rotation.from_quat(pose[:, 4:8]))
    est_rot = slerp(gt_t)

    # start-anchored SE(3) alignment (identical to our eval)
    R_est = est_rot.as_matrix()
    R_gt = gt_rot.as_matrix()
    R_align = R_gt[0] @ R_est[0].T
    t_align = gt_pos[0] - R_align @ est_pos[0]
    est_pos_a = (R_align @ est_pos.T).T + t_align
    est_vel_a = (R_align @ est_vel.T).T
    est_rot_a = np.einsum('ij,njk->nik', R_align, R_est)

    al_eul = np.degrees(Rotation.from_matrix(R_align).as_euler('xyz'))
    print(f'[eval] SE3 alignment R: [{al_eul[0]:.1f}, {al_eul[1]:.1f}, '
          f'{al_eul[2]:.1f}] deg  t: [{t_align[0]:.3f}, {t_align[1]:.3f}, '
          f'{t_align[2]:.3f}] m')

    pos_rmse = float(np.sqrt(np.mean(np.sum((est_pos_a - gt_pos) ** 2, axis=1))))
    vel_rmse = float(np.sqrt(np.mean(np.sum((est_vel_a - gt_vel) ** 2, axis=1))))
    R_err = np.einsum('nji,njk->nik', gt_rot.as_matrix(), est_rot_a)  # gt^T est
    cosang = np.clip((np.einsum('nii->n', R_err) - 1) / 2, -1, 1)
    rot_err = np.degrees(np.arccos(cosang))
    rot_rmse = float(np.sqrt(np.mean(rot_err ** 2)))
    path_len = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))
    drift = 100.0 * pos_rmse / path_len

    print(f'\n=== {args.alias} :: {Path(args.result_bag).stem} (causal/live) ===')
    print(f'  Position RMSE : {pos_rmse:.3f} m   (drift {drift:.2f}% of '
          f'{path_len:.1f} m path)')
    print(f'  Velocity RMSE : {vel_rmse:.3f} m/s')
    print(f'  Orientation RMSE: {rot_rmse:.2f} deg  (max {rot_err.max():.1f})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
