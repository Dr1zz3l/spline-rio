import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""Check what topics and message types are in each bag."""
import struct
import sys

from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

BAGS = {
    'original':    'rosbags/2025-12-17-16-02-22.bag',
    'circle':      'rosbags/circle_2025-12-17-17-21-37.bag',
    'circle_fwd':  'rosbags/circle_forward_2025-12-17-17-37-38.bag',
    'backflips':   'rosbags/backflips_2025-12-17-17-41-24.bag',
}

typestore = get_typestore(Stores.ROS1_NOETIC)

for name, path in BAGS.items():
    print(f"\n{'='*60}")
    print(f"BAG: {name} ({path})")

    with Reader(Path(path)) as reader:
        # Show radar-related topics
        for conn in reader.connections:
            topic = conn.topic
            if 'radar' in topic.lower() or 'mmwave' in topic.lower() or 'ti_' in topic.lower():
                print(f"  {topic}: type={conn.msgtype} msgs={conn.msgcount}")

        # Check first message of radar topic for field names
        radar_topic = '/ti_mmwave/radar_scan_pcl_0'
        radar_conns = [c for c in reader.connections if c.topic == radar_topic]
        count = 0
        for conn, timestamp, rawdata in reader.messages(connections=radar_conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            field_names = [f.name for f in msg.fields]
            field_types = [f.datatype for f in msg.fields]
            print(f"  Fields: {list(zip(field_names, field_types))}")
            print(f"  Width: {msg.width}, Height: {msg.height}, Point step: {msg.point_step}")

            # Read first few points from raw data
            point_step = msg.point_step
            data = bytes(msg.data)
            n_points = msg.width * msg.height
            if n_points > 0:
                # Build struct format from fields
                dtype_map = {2: 'B', 7: 'f', 8: 'd'}  # UINT8, FLOAT32, FLOAT64
                fmt = '<' + ''.join(dtype_map.get(f.datatype, 'f') for f in msg.fields)
                for i in range(min(n_points, 2)):
                    offset = i * point_step
                    pt = struct.unpack_from(fmt, data, offset)
                    print(f"  Point {i}: {pt}")

            count += 1
            if count >= 2:
                break
