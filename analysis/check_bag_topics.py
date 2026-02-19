"""Check what topics and message types are in each bag."""
import rosbag
import sys, os
sys.path.insert(0, 'analysis')

BAGS = {
    'original':    'rosbags/2025-12-17-16-02-22.bag',
    'circle':      'rosbags/circle_2025-12-17-17-21-37.bag',
    'circle_fwd':  'rosbags/circle_forward_2025-12-17-17-37-38.bag',
    'backflips':   'rosbags/backflips_2025-12-17-17-41-24.bag',
}

for name, path in BAGS.items():
    print(f"\n{'='*60}")
    print(f"BAG: {name} ({path})")
    bag = rosbag.Bag(path)
    info = bag.get_type_and_topic_info()
    
    # Show radar-related topics
    for topic, info_t in info.topics.items():
        if 'radar' in topic.lower() or 'mmwave' in topic.lower() or 'ti_' in topic.lower():
            print(f"  {topic}: type={info_t.msg_type} msgs={info_t.message_count}")
    
    # Check first message of radar topic for field names
    radar_topic = '/ti_mmwave/radar_scan_pcl_0'
    count = 0
    for _, msg, t in bag.read_messages(topics=[radar_topic]):
        field_names = [f.name for f in msg.fields]
        field_types = [f.datatype for f in msg.fields]
        print(f"  Fields: {list(zip(field_names, field_types))}")
        print(f"  Width: {msg.width}, Height: {msg.height}, Point step: {msg.point_step}")
        
        # Read first few points to check values
        import sensor_msgs.point_cloud2 as pc2
        pts = list(pc2.read_points(msg, skip_nans=True))
        if len(pts) > 0:
            print(f"  First point: {pts[0]}")
            if len(pts) > 1:
                print(f"  Second point: {pts[1]}")
        count += 1
        if count >= 2:
            break
    
    bag.close()
