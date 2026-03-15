import sys
import numpy as np
sys.path.insert(0, '/home/mouse/MyData/radar-iwr6843-driver')
from analysis.rosbag_loader.loader import BagReaderWrapper

bag_path = 'rosbags/circle_forward_2025-12-17-17-37-38.bag'
bag = BagReaderWrapper(bag_path)
topic = '/angrybird2/imu'

has_valid_ori = False
has_valid_cov = False
total_msgs = 0

for _, msg, t in bag.read_messages(topics=[topic]):
    total_msgs += 1
    
    # Check orientation
    ori = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    if not np.isnan(ori).any():
        if not has_valid_ori:
            print(f"First valid orientation found at msg {total_msgs} (t={t.to_sec():.3f}): {ori}")
        has_valid_ori = True
        
    # Check covariance
    ori_cov = msg.orientation_covariance
    ang_cov = msg.angular_velocity_covariance
    lin_cov = msg.linear_acceleration_covariance
    
    if any(c != 0.0 for c in ori_cov) or any(c != 0.0 for c in ang_cov) or any(c != 0.0 for c in lin_cov):
        if not has_valid_cov:
            print(f"First non-zero covariance found at msg {total_msgs} (t={t.to_sec():.3f})")
            print(f"  ori_cov: {ori_cov}")
            print(f"  ang_cov: {ang_cov}")
            print(f"  lin_cov: {lin_cov}")
        has_valid_cov = True

bag.close()

if not has_valid_ori:
    print(f"Checked {total_msgs} messages. ALL orientations are NaN.")
if not has_valid_cov:
    print(f"Checked {total_msgs} messages. ALL covariances are purely 0.0.")
