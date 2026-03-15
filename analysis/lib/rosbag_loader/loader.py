"""ROS bag loading and inspection utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy.stats import linregress
from copy import copy
import dataclasses


try:
    import rosbag

    HAS_ROSBAG = True
except ImportError:
    HAS_ROSBAG = False

try:
    from rosbags.rosbag1 import Reader as Rosbag1Reader
    from rosbags.typesys import Stores, get_typestore
    from rosbags.typesys.msg import get_types_from_msg, normalize_msgtype

    HAS_ROSbags = True
except ImportError:
    HAS_ROSbags = False

try:
    from scipy.stats import linregress

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .structures import (
    MocapPose,
    MocapAccel,
    AgirosState,
    AgirosOdometry,
    IMUData,
    RadarPointCloud,
    RadarVelocity,
    BagData,
)


class BagReaderWrapper:
    """Wrapper to provide a unified interface for reading rosbags via either 'rosbag' or 'rosbags' library."""

    def __init__(self, path: str):
        self.path = str(path)
        self.use_rosbags = not HAS_ROSBAG and HAS_ROSbags
        if not HAS_ROSBAG and not HAS_ROSbags:
            raise ImportError(
                "Neither rosbag nor rosbags is available. Install with: pip install rosbags"
            )

        if self.use_rosbags:
            self.typestore = get_typestore(Stores.ROS1_NOETIC)
            self.reader = Rosbag1Reader(self.path)
            self.reader.open()
            # Register custom message types from bag connection headers
            custom_typs = {}
            for c in self.reader.connections:
                if c.msgtype not in self.typestore.fielddefs:
                    _, msgdef_text = c.msgdef
                    custom_typs.update(
                        get_types_from_msg(msgdef_text, normalize_msgtype(c.msgtype))
                    )
            if custom_typs:
                self.typestore.register(custom_typs)
        else:
            self.bag = rosbag.Bag(self.path)

    def close(self):
        if self.use_rosbags:
            self.reader.close()
        else:
            self.bag.close()

    def get_start_time(self):
        if self.use_rosbags:
            return self.reader.start_time / 1e9
        else:
            return self.bag.get_start_time()

    def get_end_time(self):
        if self.use_rosbags:
            return self.reader.end_time / 1e9
        else:
            return self.bag.get_end_time()

    def read_messages(self, topics):
        if self.use_rosbags:
            connections = [x for x in self.reader.connections if x.topic in topics]
            for connection, timestamp, rawdata in self.reader.messages(
                connections=connections
            ):
                msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)

                # mock the ROS time object for t.to_sec()
                class MockTime:
                    def __init__(self, ns):
                        self.ns = ns

                    def to_sec(self):
                        return self.ns / 1e9

                yield connection.topic, msg, MockTime(timestamp)
        else:
            for topic, msg, t in self.bag.read_messages(topics=topics):
                yield topic, msg, t

    def get_type_and_topic_info(self):
        """Returns simplified topic info dict."""
        topics_info = {}
        if self.use_rosbags:
            for connection in self.reader.connections:
                if connection.topic not in topics_info:
                    topics_info[connection.topic] = {
                        "msg_type": connection.msgtype,
                        "message_count": connection.msgcount,
                    }
                else:
                    topics_info[connection.topic]["message_count"] += (
                        connection.msgcount
                    )
        else:
            info = self.bag.get_type_and_topic_info()
            for topic, t_info in info.topics.items():
                topics_info[topic] = {
                    "msg_type": t_info.msg_type,
                    "message_count": t_info.message_count,
                }
        return topics_info


def inspect_bag_topics(bag_path: str) -> Dict[str, Any]:
    """Inspect rosbag topics and print their structure.

    Args:
        bag_path: Path to the rosbag file

    Returns:
        Dictionary with topic information
    """
    bag = BagReaderWrapper(bag_path)

    topics_full_info = {}
    try:
        topics_info = bag.get_type_and_topic_info()

        print(f"\n{'=' * 60}")
        print(f"Bag inspection: {bag_path}")
        print(f"Duration: {bag.get_end_time() - bag.get_start_time():.2f} seconds")
        print(f"{'=' * 60}\n")

        print(f"{'Topic':<50} {'Type':<40} {'Count':>8}")
        print("-" * 100)

        for topic in sorted(topics_info.keys()):
            topic_info = topics_info[topic]
            msg_type = topic_info["msg_type"]
            msg_count = topic_info["message_count"]

            print(f"{topic:<50} {msg_type:<40} {msg_count:>8}")
            topics_full_info[topic] = {
                "type": msg_type,
                "count": msg_count,
            }

            # Sample first message
            try:
                for _, msg, _ in bag.read_messages(topics=[topic]):
                    topics_full_info[topic]["sample"] = str(msg)[:200]
                    break
            except Exception as e:
                topics_full_info[topic]["sample"] = f"Error sampling: {e}"

        print(f"\n{'=' * 60}\n")

    finally:
        bag.close()

    return topics_full_info


def stitch_cpu_counter_resets(
    radar_frames: List[Any],
    verbose: bool = False,
    gap_threshold_billion: float = 0.5,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Detect and correct CPU counter resets in radar data.

    The radar CPU counter can reset during data collection (e.g., 32-bit overflow).
    This function detects reset events by analyzing the CPU cycles vs ROS time relationship
    to identify discontinuous segments, then stitches them together.

    Note: ROS bag data is chronologically sorted, so we look for two-cluster patterns
    in the CPU vs time scatter plot rather than monotonicity violations.

    Args:
        radar_frames: List of RadarVelocity or RadarPointCloud objects
        verbose: Print diagnostic information
        gap_threshold_billion: Threshold in billions of cycles to detect gaps

    Returns:
        Tuple of (corrected_frames, diagnostics_dict)
        - corrected_frames: New list with corrected CPU cycles
        - diagnostics_dict: Contains reset detection info and fit quality
    """
    if not HAS_SCIPY:
        if verbose:
            print("⚠️  scipy not available - skipping CPU counter reset correction")
        return radar_frames, {"enabled": False}

    if len(radar_frames) == 0:
        return radar_frames, {"enabled": False, "reason": "no_data"}

    # Extract CPU cycles and ROS timestamps
    cpu_cycles = []
    ros_times = []
    valid_indices = []

    for idx, frame in enumerate(radar_frames):
        if frame.time_cpu_cycles is not None and len(frame.time_cpu_cycles) > 0:
            cpu_cycles.append(frame.time_cpu_cycles[0])
            ros_times.append(frame.timestamp)
            valid_indices.append(idx)

    if len(cpu_cycles) < 20:
        if verbose:
            print("⚠️  Not enough frames with CPU cycles for reset detection")
        return radar_frames, {"enabled": False, "reason": "insufficient_data"}

    cpu_cycles = np.array(cpu_cycles)
    ros_times = np.array(ros_times)

    # Detect resets by analyzing CPU cycle distribution
    # After reading from bag (sorted by time), a reset creates a gap in CPU cycles
    cpu_min, cpu_max = cpu_cycles.min(), cpu_cycles.max()
    cpu_range = cpu_max - cpu_min

    if cpu_range < gap_threshold_billion * 1e9:
        # No significant range, unlikely to have resets
        if verbose:
            print(f"✓ CPU counter appears continuous (range: {cpu_range / 1e9:.3f}B)")
        return radar_frames, {
            "enabled": True,
            "resets_detected": False,
            "cpu_range_billion": cpu_range / 1e9,
        }

    # Use histogram to find gaps in CPU cycle distribution
    cpu_bins = 20
    hist, bin_edges = np.histogram(cpu_cycles, bins=cpu_bins)

    # Find gaps (bins with zero counts)
    gap_mask = hist == 0
    gap_indices = np.where(gap_mask)[0]

    if len(gap_indices) == 0:
        if verbose:
            print(f"✓ No gaps detected in CPU cycle distribution")
        return radar_frames, {
            "enabled": True,
            "resets_detected": False,
            "cpu_range_billion": cpu_range / 1e9,
        }

    # Find the largest contiguous gap to use as split point
    largest_gap_idx = gap_indices[0]
    for i in range(len(gap_indices) - 1):
        if gap_indices[i + 1] != gap_indices[i] + 1:
            break
        largest_gap_idx = gap_indices[i + 1]

    gap_boundary = bin_edges[largest_gap_idx + 1]

    if verbose:
        print(f"\n⚠️  CPU counter reset detected!")
        print(f"  Split boundary: {gap_boundary / 1e9:.3f}B cycles")

    # Split data into two segments
    seg1_mask = cpu_cycles < gap_boundary
    seg2_mask = cpu_cycles >= gap_boundary

    seg1_cpu = cpu_cycles[seg1_mask]
    seg1_ros = ros_times[seg1_mask]
    seg1_indices = np.array(valid_indices)[seg1_mask]

    seg2_cpu = cpu_cycles[seg2_mask]
    seg2_ros = ros_times[seg2_mask]
    seg2_indices = np.array(valid_indices)[seg2_mask]

    if len(seg1_cpu) < 5 or len(seg2_cpu) < 5:
        if verbose:
            print(
                f"⚠️  Segments too small for stitching (n1={len(seg1_cpu)}, n2={len(seg2_cpu)})"
            )
        return radar_frames, {
            "enabled": True,
            "resets_detected": True,
            "stitching_failed": "segments_too_small",
        }

    # Fit each segment to verify clock consistency
    slope1, int1, r1, _, _ = linregress(seg1_cpu, seg1_ros)
    slope2, int2, r2, _, _ = linregress(seg2_cpu, seg2_ros)

    if verbose:
        print(
            f"  Segment 1: {len(seg1_cpu)} frames, R²={r1**2:.6f}, {1.0 / slope1 / 1e6:.2f} MHz"
        )
        print(
            f"  Segment 2: {len(seg2_cpu)} frames, R²={r2**2:.6f}, {1.0 / slope2 / 1e6:.2f} MHz"
        )

    # Determine chronological order (data is already sorted by ROS time)
    if seg2_ros[0] < seg1_ros[0]:
        # Segment 2 comes first
        first_cpu, first_ros, first_indices = seg2_cpu, seg2_ros, seg2_indices
        second_cpu, second_ros, second_indices = seg1_cpu, seg1_ros, seg1_indices
        first_slope = slope2
    else:
        # Segment 1 comes first
        first_cpu, first_ros, first_indices = seg1_cpu, seg1_ros, seg1_indices
        second_cpu, second_ros, second_indices = seg2_cpu, seg2_ros, seg2_indices
        first_slope = slope1

    # Calculate offset to stitch segments
    cpu_offset = first_cpu[-1]
    ros_gap = second_ros[0] - first_ros[-1]
    estimated_gap_cycles = ros_gap / first_slope
    total_offset = cpu_offset + estimated_gap_cycles

    if verbose:
        print(
            f"  Stitching: offset={total_offset / 1e9:.3f}B, gap={ros_gap * 1000:.1f}ms"
        )

    # Create corrected frames
    # Build mapping from original valid_indices to segment membership
    valid_to_segment = {}
    for vidx in first_indices:
        valid_to_segment[vidx] = "first"
    for vidx in second_indices:
        valid_to_segment[vidx] = "second"

    # Safety check: all valid indices should be classified
    if len(valid_to_segment) != len(valid_indices):
        if verbose:
            print(
                f"⚠️  Warning: {len(valid_indices)} valid frames but {len(valid_to_segment)} classified"
            )

    corrected_frames = []
    valid_idx_counter = 0

    for idx, frame in enumerate(radar_frames):
        has_cpu = frame.time_cpu_cycles is not None and len(frame.time_cpu_cycles) > 0

        # Determine if this frame needs correction
        needs_correction = False
        if has_cpu and valid_idx_counter < len(valid_indices):
            if valid_indices[valid_idx_counter] == idx:
                # This is a valid frame - check which segment
                segment = valid_to_segment.get(idx)
                if segment is None:
                    raise ValueError(
                        f"Frame {idx} has CPU cycles but not classified into any segment"
                    )
                needs_correction = segment == "second"
                valid_idx_counter += 1

        if needs_correction and has_cpu:
            # Apply offset to second segment
            corrected_cycles = [c + total_offset for c in frame.time_cpu_cycles]
            # Create new frame with corrected cycles
            if isinstance(frame, RadarVelocity):
                new_frame = RadarVelocity(
                    timestamp=frame.timestamp,
                    positions=frame.positions,
                    velocities=frame.velocities,
                    intensities=frame.intensities,
                    ranges=frame.ranges,
                    noise=frame.noise,
                    time_cpu_cycles=corrected_cycles,
                    frame_number=frame.frame_number,
                )
            elif isinstance(frame, RadarPointCloud):
                new_frame = RadarPointCloud(
                    timestamp=frame.timestamp,
                    positions=frame.positions,
                    velocities=frame.velocities,
                    intensities=frame.intensities,
                    ranges=frame.ranges,
                    noise=frame.noise,
                    time_cpu_cycles=corrected_cycles,
                    frame_number=frame.frame_number,
                )
            else:
                new_frame = frame
            corrected_frames.append(new_frame)
        else:
            # No correction needed
            corrected_frames.append(frame)

    # Verify stitching quality
    cpu_stitched = np.array(
        [
            f.time_cpu_cycles[0]
            for f in corrected_frames
            if f.time_cpu_cycles is not None and len(f.time_cpu_cycles) > 0
        ]
    )
    ros_stitched = np.array(
        [
            f.timestamp
            for f in corrected_frames
            if f.time_cpu_cycles is not None and len(f.time_cpu_cycles) > 0
        ]
    )

    slope_stitched, int_stitched, r_stitched, _, _ = linregress(
        cpu_stitched, ros_stitched
    )

    if verbose:
        print(
            f"  Result: R²={r_stitched**2:.6f}, clock={1.0 / slope_stitched / 1e6:.2f} MHz"
        )
        print(f"  ✓ CPU counter reset corrected successfully\n")

    diagnostics = {
        "enabled": True,
        "resets_detected": True,
        # Note: Current implementation only handles single reset (2 segments)
        # Multiple resets would require iterative gap detection
        "segment1_size": len(first_cpu),
        "segment2_size": len(second_cpu),
        "gap_boundary_billion": gap_boundary / 1e9,
        "total_offset_billion": total_offset / 1e9,
        "r_squared_before": [r1**2, r2**2],
        "r_squared_after": r_stitched**2,
        "clock_freq_mhz": 1.0 / slope_stitched / 1e6,
    }

    return corrected_frames, diagnostics


def stitch_cpu_counter_resets_improved(
    radar_frames: List[Any],
    verbose: bool = False,
    use_clustering: bool = True,
    reset_drop_threshold: float = -1e9,  # A drop of >1 billion cycles implies a reset
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Robustly stitches an arbitrary number of CPU counter resets (N-resets).

    Algorithm:
    1. Detects resets using clustering (DBSCAN) or derivatives
    2. Iterates through segments defined by these resets
    3. For each gap, calculates the expected CPU cycle jump based on ROS timestamp gap
    4. Accumulates a running offset to stitch segments into a single linear timeline

    Args:
        radar_frames: List of radar frame objects
        verbose: Print diagnostic information
        use_clustering: Use DBSCAN clustering to detect resets (more robust)
        reset_drop_threshold: Threshold for derivative-based detection (billions)
    """

    if not radar_frames:
        return radar_frames, {"enabled": False, "reason": "no_data"}

    # --- 1. Fast Vectorized Extraction ---
    valid_data = []
    for idx, frame in enumerate(radar_frames):
        if frame.time_cpu_cycles is not None and len(frame.time_cpu_cycles) > 0:
            valid_data.append((idx, frame.timestamp, frame.time_cpu_cycles[0]))

    if len(valid_data) < 10:
        if verbose:
            print("⚠️ Not enough data points to perform stitching.")
        return radar_frames, {"enabled": False, "reason": "insufficient_data"}

    indices, ros_times, cpu_starts = map(np.array, zip(*valid_data))

    # --- 2. Reset Detection ---
    reset_indices = []

    if use_clustering:
        # Use DBSCAN clustering to find linear segments
        try:
            from sklearn.cluster import DBSCAN

            # Normalize data for clustering
            cpu_norm = (cpu_starts - cpu_starts.mean()) / cpu_starts.std()
            ros_norm = (ros_times - ros_times.mean()) / ros_times.std()
            X = np.column_stack([cpu_norm, ros_norm])

            db = DBSCAN(eps=0.5, min_samples=5)
            labels = db.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters <= 1:
                if verbose:
                    print("✓ No CPU counter resets detected (single cluster).")
                return radar_frames, {"resets_detected": False, "method": "clustering"}

            # Find boundaries between clusters based on ROS time ordering
            # Since data is sorted by ROS time, find where cluster labels change
            label_changes = np.where(np.diff(labels) != 0)[0]
            reset_indices = label_changes

            if verbose:
                print(
                    f"⚠️  Clustering detected {n_clusters} segments, {len(reset_indices)} resets"
                )

        except ImportError:
            if verbose:
                print("⚠️  sklearn not available, falling back to derivative detection")
            use_clustering = False

    if not use_clustering or len(reset_indices) == 0:
        # Fallback: Derivative-based detection
        diffs = np.diff(cpu_starts.astype(float))
        reset_indices = np.where(diffs < reset_drop_threshold)[0]

        if len(reset_indices) == 0:
            if verbose:
                print("✓ No CPU counter resets detected (derivative method).")
            return radar_frames, {"resets_detected": False, "method": "derivative"}

    if verbose:
        print(
            f"⚠️  Detected {len(reset_indices)} resets. Stitching {len(reset_indices) + 1} segments..."
        )

    # --- 3. Iterative Stitching ---
    corrected_frames = list(radar_frames)  # Shallow copy list

    # Define segment boundaries: [0, reset_1, reset_2, ..., end]
    # These are indices into the `valid_data` arrays, not the original list
    segment_boundaries = np.concatenate(([-1], reset_indices, [len(indices) - 1]))

    total_offset = 0.0
    diagnostics_log = []

    # Loop through each segment gap
    for i in range(len(segment_boundaries) - 1):
        # Current segment indices (inclusive)
        curr_seg_start = segment_boundaries[i] + 1
        curr_seg_end = segment_boundaries[i + 1]

        # Determine the offset for the NEXT segment (if there is one)
        if i < len(segment_boundaries) - 2:
            # We need to bridge the gap between curr_seg and next_seg

            # Get data for current segment to fit clock speed
            seg_cpu = cpu_starts[curr_seg_start : curr_seg_end + 1]
            seg_ros = ros_times[curr_seg_start : curr_seg_end + 1]

            # Default to simple 32-bit wrap if fit fails or segment too short
            # (2**32 is approx 4.29 billion)
            segment_gap_offset = 2**32
            method = "naive_wrap"

            if len(seg_cpu) > 5:
                # Fit line: ROS = slope * CPU + intercept
                # slope units: seconds per cycle
                slope, _, _, _, _ = linregress(seg_cpu, seg_ros)

                # Check if slope is sane (e.g., around 1/expect_freq)
                # If slope is negative or wildly off, skip smart stitching
                if slope > 0:
                    last_ros = seg_ros[-1]
                    last_cpu_corrected = (
                        seg_cpu[-1] + total_offset
                    )  # Use currently accumulated offset

                    next_idx = curr_seg_end + 1
                    next_ros = ros_times[next_idx]
                    next_cpu_raw = cpu_starts[next_idx]

                    # Calculate gap in ROS time
                    ros_gap = next_ros - last_ros

                    # Convert ROS gap to CPU cycles using current clock speed
                    estimated_cpu_gap = ros_gap / slope

                    # Where the next segment SHOULD start
                    target_next_start = last_cpu_corrected + estimated_cpu_gap

                    # The offset needed to get there
                    # offset = target - raw
                    new_total_offset = target_next_start - next_cpu_raw

                    segment_gap_offset = new_total_offset - total_offset
                    method = "linear_fit"

            # Update the running total for the *next* loop iteration
            total_offset += segment_gap_offset

            if verbose:
                print(
                    f"   Gap {i + 1}: method={method}, added_offset={segment_gap_offset:.0f}, total={total_offset:.0f}"
                )

        # Apply the CURRENT total_offset to all frames in the current segment
        # (Note: For the first segment, total_offset is 0, which is correct)
        # However, because we calculate the offset for the NEXT segment at the end of the loop,
        # we need to be careful.

        # Actually, simpler logic:
        # Loop 0: Apply offset 0. Calculate offset for Loop 1.
        # Loop 1: Apply offset 1. Calculate offset for Loop 2.

        # We need to apply `total_offset` to the segment we just analyzed?
        # No, we apply `total_offset` (which starts at 0) to current segment.
        # Then we calculate the JUMP for the next one.

        # WAIT: The offset calculation above updated `total_offset` for the NEXT segment.
        # So we must apply the offset BEFORE updating it.
        pass

    # --- Corrected Loop Logic ---
    total_offset = 0.0

    for i in range(len(segment_boundaries) - 1):
        curr_seg_start = segment_boundaries[i] + 1
        curr_seg_end = segment_boundaries[i + 1]

        # 1. Apply current cumulative offset to this segment
        if total_offset != 0:
            for k in range(curr_seg_start, curr_seg_end + 1):
                original_idx = indices[k]
                frame = radar_frames[original_idx]

                # Create corrected cycles
                new_cycles = [c + total_offset for c in frame.time_cpu_cycles]

                # Generic object update
                if dataclasses.is_dataclass(frame):
                    new_frame = dataclasses.replace(frame, time_cpu_cycles=new_cycles)
                else:
                    new_frame = copy(frame)
                    new_frame.time_cpu_cycles = new_cycles

                corrected_frames[original_idx] = new_frame

        # 2. Calculate jump for the NEXT segment (if not last segment)
        if i < len(segment_boundaries) - 2:
            seg_cpu = cpu_starts[curr_seg_start : curr_seg_end + 1]
            seg_ros = ros_times[curr_seg_start : curr_seg_end + 1]

            next_idx = curr_seg_end + 1
            next_ros = ros_times[next_idx]
            next_cpu_raw = cpu_starts[next_idx]

            # Linear Fit to project gap
            jump = 2**32  # Default naive wrap

            if len(seg_cpu) > 5:
                slope, _, _, _, _ = linregress(seg_cpu, seg_ros)
                if slope > 0:
                    last_ros = seg_ros[-1]
                    # Important: Use raw CPU for slope calc, but we need the END of the stitched line
                    last_cpu_stitched = seg_cpu[-1] + total_offset

                    ros_gap = next_ros - last_ros
                    estimated_cycles_gap = ros_gap / slope

                    target_next_start = last_cpu_stitched + estimated_cycles_gap

                    # The total offset required for the next segment
                    required_total_offset = target_next_start - next_cpu_raw

                    # The jump is the difference between new total and current total
                    jump = required_total_offset - total_offset

            total_offset += jump

    return corrected_frames, {
        "enabled": True,
        "resets_detected": True,
        "resets_count": len(reset_indices),
        "final_offset": total_offset,
    }


def _extract_position_and_orientation(pose_msg) -> tuple[np.ndarray, np.ndarray]:
    """Extract position [x,y,z] and orientation [qx,qy,qz,qw] from pose message.

    Handles both PoseStamped (msg.pose.position) and Pose (msg.position) formats.
    """
    # Try PoseStamped format first (msg.pose.position)
    if hasattr(pose_msg, "pose") and hasattr(pose_msg.pose, "position"):
        pos = np.array(
            [
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z,
            ]
        )
        ori = np.array(
            [
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w,
            ]
        )
    # Fall back to Pose format (msg.position)
    elif hasattr(pose_msg, "position"):
        pos = np.array(
            [
                pose_msg.position.x,
                pose_msg.position.y,
                pose_msg.position.z,
            ]
        )
        ori = np.array(
            [
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w,
            ]
        )
    else:
        raise AttributeError(
            f"Cannot extract position/orientation from {type(pose_msg)}"
        )

    return pos, ori


def _extract_vector3(vec) -> np.ndarray:
    """Extract [x, y, z] from a Vector3 message."""
    return np.array([vec.x, vec.y, vec.z])


def _extract_quaternion(quat) -> np.ndarray:
    """Extract [x, y, z, w] from a Quaternion message."""
    return np.array([quat.x, quat.y, quat.z, quat.w])


def _extract_point_cloud_velocities(msg):
    """
    Extract all fields from radar PointCloud2 messages.

    The radar publishes PointCloud2 messages with custom fields including velocity.
    - /mmWaveDataHdl/RScanVelocity has 9 fields: x, y, z, velocity, intensity, range, noise, time_cpu_cycles, frame_number
    - /ti_mmwave/radar_scan_pcl_0 has 5 fields: x, y, z, intensity, velocity

    Args:
        msg: sensor_msgs/PointCloud2 message

    Returns:
        Tuple of (positions, velocities, intensities, ranges, noise, time_cpu_cycles, frame_number)
        Arrays are numpy arrays or None if field not present
    """
    field_names = [
        field.name if hasattr(field, "name") else field["name"] for field in msg.fields
    ]

    # Try using sensor_msgs.point_cloud2 if available and the message is a standard ROS msg
    try:
        import sensor_msgs.point_cloud2 as pc2

        HAS_PC2 = True
    except ImportError:
        HAS_PC2 = False

    is_standard_ros = hasattr(msg, "_type") and msg._type == "sensor_msgs/PointCloud2"

    if HAS_PC2 and is_standard_ros:
        (
            positions,
            velocities,
            intensities,
            ranges,
            noise_vals,
            time_cycles,
            frame_nums,
        ) = [], [], [], [], [], [], []
        for point in pc2.read_points(msg, skip_nans=True):
            if "x" in field_names and "y" in field_names and "z" in field_names:
                x_idx, y_idx, z_idx = (
                    field_names.index("x"),
                    field_names.index("y"),
                    field_names.index("z"),
                )
                positions.append([point[x_idx], point[y_idx], point[z_idx]])
            if "velocity" in field_names:
                velocities.append(point[field_names.index("velocity")])
            if "intensity" in field_names:
                intensities.append(point[field_names.index("intensity")])
            if "range" in field_names:
                ranges.append(point[field_names.index("range")])
            if "noise" in field_names:
                noise_vals.append(point[field_names.index("noise")])
            if "time_cpu_cycles" in field_names:
                time_cycles.append(point[field_names.index("time_cpu_cycles")])
            if "frame_number" in field_names:
                frame_nums.append(point[field_names.index("frame_number")])

        positions = (
            np.array(positions) if len(positions) > 0 else np.array([]).reshape(0, 3)
        )
        velocities = np.array(velocities) if len(velocities) > 0 else None
        intensities = np.array(intensities) if len(intensities) > 0 else None
        ranges = np.array(ranges) if len(ranges) > 0 else None
        noise_vals = np.array(noise_vals) if len(noise_vals) > 0 else None
        time_cycles = (
            np.array(time_cycles, dtype=np.uint32) if len(time_cycles) > 0 else None
        )
        frame_nums = (
            np.array(frame_nums, dtype=np.uint32) if len(frame_nums) > 0 else None
        )

        return (
            positions,
            velocities,
            intensities,
            ranges,
            noise_vals,
            time_cycles,
            frame_nums,
        )

    # Custom parsing for rosbags messages or when sensor_msgs is not available
    type_mapping = {
        1: np.int8,
        2: np.uint8,
        3: np.int16,
        4: np.uint16,
        5: np.int32,
        6: np.uint32,
        7: np.float32,
        8: np.float64,
    }

    struct_dtype_dict = {"names": [], "formats": [], "offsets": []}
    for field in msg.fields:
        fname = field.name if hasattr(field, "name") else field["name"]
        ftype = field.datatype if hasattr(field, "datatype") else field["datatype"]
        foffset = field.offset if hasattr(field, "offset") else field["offset"]
        struct_dtype_dict["names"].append(fname)
        struct_dtype_dict["formats"].append(type_mapping.get(ftype, np.float32))
        struct_dtype_dict["offsets"].append(foffset)

    struct_dtype_dict["itemsize"] = msg.point_step
    data_bytes = bytes(msg.data)

    points = np.frombuffer(data_bytes, dtype=struct_dtype_dict)

    if (
        "x" in struct_dtype_dict["names"]
        and "y" in struct_dtype_dict["names"]
        and "z" in struct_dtype_dict["names"]
    ):
        valid_mask = (
            ~np.isnan(points["x"]) & ~np.isnan(points["y"]) & ~np.isnan(points["z"])
        )
        points = points[valid_mask]

    if len(points) == 0:
        return np.array([]).reshape(0, 3), None, None, None, None, None, None

    names = struct_dtype_dict["names"]
    positions = (
        np.column_stack((points["x"], points["y"], points["z"]))
        if "x" in names
        else np.array([]).reshape(0, 3)
    )
    velocities = points["velocity"] if "velocity" in names else None
    intensities = points["intensity"] if "intensity" in names else None
    ranges = points["range"] if "range" in names else None
    noise_vals = points["noise"] if "noise" in names else None
    time_cycles = (
        points["time_cpu_cycles"].astype(np.uint32)
        if "time_cpu_cycles" in names
        else None
    )
    frame_nums = (
        points["frame_number"].astype(np.uint32) if "frame_number" in names else None
    )

    return (
        positions,
        velocities,
        intensities,
        ranges,
        noise_vals,
        time_cycles,
        frame_nums,
    )


def load_bag_topics(bag_path: str, verbose: bool = True) -> BagData:
    """Load relevant topics from a rosbag file.

    Args:
        bag_path: Path to the rosbag file
        verbose: Print loading progress

    Returns:
        BagData object containing all extracted data
    """
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    bag = BagReaderWrapper(bag_path)

    # Initialize storage
    mocap_pose_list = []
    mocap_accel_list = []
    agiros_state_list = []
    agiros_odometry_list = []
    imu_list = []
    radar_pcl_list = []
    radar_velocity_list = []

    try:
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()
        duration = end_time - start_time

        if verbose:
            print(f"\nLoading rosbag: {bag_path.name}")
            print(f"Duration: {duration:.2f}s")

        # Load MoCap Pose
        if verbose:
            print("  Loading /mocap/angrybird2/pose...")
        for _, msg, t in bag.read_messages(topics=["/mocap/angrybird2/pose"]):
            try:
                pos, ori = _extract_position_and_orientation(msg)
                mocap_pose_list.append(
                    MocapPose(
                        timestamp=t.to_sec(),
                        position=pos,
                        orientation=ori,
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"    Error parsing message: {e}")

        # Load MoCap Accel (TwistStamped with linear/angular velocity, not accel)
        if verbose:
            print("  Loading /mocap/angrybird2/accel...")
        for _, msg, t in bag.read_messages(topics=["/mocap/angrybird2/accel"]):
            try:
                # Note: Despite the topic name "accel", this is actually TwistStamped
                # containing linear and angular velocity
                linear_vel = _extract_vector3(msg.twist.linear)
                angular_vel = _extract_vector3(msg.twist.angular)
                mocap_accel_list.append(
                    MocapAccel(
                        timestamp=t.to_sec(),
                        linear_acceleration=linear_vel,  # Actually velocity data
                        angular_velocity=angular_vel,
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"    Error parsing message: {e}")

        # Load Agiros State (QuadState message, not Odometry)
        if verbose:
            print("  Loading /angrybird2/agiros_pilot/state...")
        for _, msg, t in bag.read_messages(topics=["/angrybird2/agiros_pilot/state"]):
            try:
                pos = _extract_vector3(msg.pose.position)
                ori = _extract_quaternion(msg.pose.orientation)
                vel = _extract_vector3(msg.velocity.linear)
                ang_vel = _extract_vector3(msg.velocity.angular)

                # Extract optional fields from QuadState message
                accel = (
                    _extract_vector3(msg.acceleration.linear)
                    if hasattr(msg, "acceleration")
                    else None
                )
                ang_accel = (
                    _extract_vector3(msg.acceleration.angular)
                    if hasattr(msg, "acceleration")
                    and hasattr(msg.acceleration, "angular")
                    else None
                )
                jerk = _extract_vector3(msg.jerk) if hasattr(msg, "jerk") else None
                snap = _extract_vector3(msg.snap) if hasattr(msg, "snap") else None
                acc_bias = (
                    _extract_vector3(msg.acc_bias) if hasattr(msg, "acc_bias") else None
                )
                gyr_bias = (
                    _extract_vector3(msg.gyr_bias) if hasattr(msg, "gyr_bias") else None
                )
                motors = (
                    np.array(msg.motors)
                    if hasattr(msg, "motors") and len(msg.motors) > 0
                    else None
                )

                agiros_state_list.append(
                    AgirosState(
                        timestamp=t.to_sec(),
                        position=pos,
                        velocity=vel,
                        orientation=ori,
                        angular_velocity=ang_vel,
                        acceleration=accel,
                        angular_acceleration=ang_accel,
                        jerk=jerk,
                        snap=snap,
                        acc_bias=acc_bias,
                        gyr_bias=gyr_bias,
                        motors=motors,
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"    Error parsing message: {e}")

        # Load Agiros Odometry
        if verbose:
            print("  Loading /angrybird2/agiros_pilot/odometry...")
        for _, msg, t in bag.read_messages(
            topics=["/angrybird2/agiros_pilot/odometry"]
        ):
            try:
                pos = _extract_vector3(msg.pose.pose.position)
                ori = _extract_quaternion(msg.pose.pose.orientation)
                vel = _extract_vector3(msg.twist.twist.linear)
                ang_vel = _extract_vector3(msg.twist.twist.angular)

                agiros_odometry_list.append(
                    AgirosOdometry(
                        timestamp=t.to_sec(),
                        position=pos,
                        velocity=vel,
                        orientation=ori,
                        angular_velocity=ang_vel,
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"    Error parsing message: {e}")

        # Load IMU Data
        if verbose:
            print("  Loading /angrybird2/imu...")
        for _, msg, t in bag.read_messages(topics=["/angrybird2/imu"]):
            try:
                accel = _extract_vector3(msg.linear_acceleration)
                ang_vel = _extract_vector3(msg.angular_velocity)
                # Note: msg.orientation exists but contains NaN values in this dataset
                # Only extract if valid (not NaN)
                ori = None
                if hasattr(msg, "orientation"):
                    quat = _extract_quaternion(msg.orientation)
                    if not np.any(np.isnan(quat)):
                        ori = quat

                imu_list.append(
                    IMUData(
                        timestamp=t.to_sec(),
                        linear_acceleration=accel,
                        angular_velocity=ang_vel,
                        orientation=ori,
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"    Error parsing message: {e}")

        # Load Radar Point Cloud
        if verbose:
            print("  Loading /ti_mmwave/radar_scan_pcl_0...")
        for _, msg, t in bag.read_messages(topics=["/ti_mmwave/radar_scan_pcl_0"]):
            try:
                # Extract all fields from PointCloud2 (PCL format with x, y, z, intensity, velocity)
                (
                    positions,
                    velocities,
                    intensities,
                    ranges,
                    noise,
                    time_cpu_cycles,
                    frame_number,
                ) = _extract_point_cloud_velocities(msg)

                if len(positions) > 0:
                    radar_pcl_list.append(
                        RadarPointCloud(
                            timestamp=t.to_sec(),
                            positions=positions,
                            velocities=velocities,
                            intensities=intensities,
                            ranges=ranges,
                            noise=noise,
                            time_cpu_cycles=time_cpu_cycles,
                            frame_number=frame_number,
                        )
                    )
            except Exception as e:
                if verbose:
                    print(f"    Error parsing message: {e}")

        # Load Radar Velocity (Most important radar topic - PointCloud2 format with 9 fields)
        if verbose:
            print("  Loading /mmWaveDataHdl/RScanVelocity...")
        for _, msg, t in bag.read_messages(topics=["/mmWaveDataHdl/RScanVelocity"]):
            try:
                # RScanVelocity is a PointCloud2 message with 9 fields
                (
                    positions,
                    velocities,
                    intensities,
                    ranges,
                    noise,
                    time_cpu_cycles,
                    frame_number,
                ) = _extract_point_cloud_velocities(msg)

                if len(positions) > 0:
                    radar_velocity_list.append(
                        RadarVelocity(
                            timestamp=t.to_sec(),
                            positions=positions,
                            velocities=velocities,
                            intensities=intensities,
                            ranges=ranges,
                            noise=noise,
                            time_cpu_cycles=time_cpu_cycles,
                            frame_number=frame_number,
                        )
                    )
            except Exception as e:
                if verbose:
                    print(f"    Error parsing message: {e}")

        if verbose:
            print("  Done!\n")

    finally:
        bag.close()

    # Create BagData object
    bag_data = BagData(
        mocap_pose=mocap_pose_list,
        mocap_accel=mocap_accel_list,
        agiros_state=agiros_state_list,
        agiros_odometry=agiros_odometry_list,
        imu_data=imu_list,
        radar_pcl=radar_pcl_list,
        radar_velocity=radar_velocity_list,
        bag_path=str(bag_path),
        start_time=start_time,
        end_time=end_time,
        duration=duration,
    )

    return bag_data
