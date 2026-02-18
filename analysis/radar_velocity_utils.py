"""
Radar velocity estimation and signal processing utilities.

This module provides functions for:
- Weighted least squares ego-velocity estimation from radar point clouds
- Signal filtering (highpass, lowpass)
- IMU acceleration integration
- Forward model for radar Doppler prediction
- Extrinsic calibration and time alignment
"""

import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt, detrend
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares, minimize
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Dict, Any

def solve_ego_velocity_weighted(positions, velocities, intensities, 
                                  min_intensity=5.0, min_range=0.2, min_points=5,
                                  use_huber=False, huber_delta=1.0):
    """
    Solve for 3D body velocity using Weighted Least Squares or robust Huber loss.
    
    Standard WLS minimizes: Σ w_i (r̂_i · v - v_rad,i)²
    Huber loss minimizes: Σ w_i * ρ((r̂_i · v - v_rad,i) / δ)
    where ρ(x) = x² for |x|≤1, 2|x|-1 for |x|>1
    
    Args:
        positions: Array of shape (N, 3) with [x, y, z] positions
        velocities: Array of shape (N,) with radial velocities
        intensities: Array of shape (N,) with signal intensities
        min_intensity: Minimum intensity threshold
        min_range: Minimum range threshold (meters)
        min_points: Minimum number of points required (still needed for Huber!)
        use_huber: If True, use Huber loss instead of L2
        huber_delta: Huber loss threshold parameter (m/s)
        
    Returns:
        v_body: 3D velocity vector [vx, vy, vz] or None if insufficient data
        
    Note:
        min_points is still required with Huber loss because we need at least
        3 points to solve for 3D velocity (more for numerical stability).
        Huber loss handles outliers better but doesn't eliminate the need for
        sufficient measurements.
    """
    H = []
    z = []
    weights = []
    
    for i in range(len(positions)):
        x, y, z_coord = positions[i]
        v_rad = velocities[i]
        intensity = intensities[i]
        
        r = np.sqrt(x**2 + y**2 + z_coord**2)
        
        # Filter weak/close returns
        if intensity < min_intensity or r < min_range:
            continue
        
        # Unit direction vector
        dir_vec = np.array([x/r, y/r, z_coord/r])
        
        H.append(dir_vec)
        z.append(v_rad)
        weights.append(intensity)
    
    if len(z) < min_points:
        return None
    
    H = np.array(H)
    z = np.array(z)
    weights = np.array(weights)
    
    if use_huber:
        # Robust estimation using Huber loss
        def residual_func(v):
            residuals = H @ v - z
            # Weight residuals by intensity
            weighted_residuals = np.sqrt(weights) * residuals
            return weighted_residuals
        
        try:
            # Initial guess from standard WLS
            W = np.diag(weights)
            lhs = H.T @ W @ H
            rhs = H.T @ W @ z
            v_init = np.linalg.solve(lhs, rhs)
            
            # Optimize with Huber loss
            result = least_squares(
                residual_func, 
                v_init, 
                loss='huber',
                f_scale=huber_delta,
                method='trf'
            )
            return result.x
        except (np.linalg.LinAlgError, ValueError):
            return None
    else:
        # Standard Weighted Least Squares
        W = np.diag(weights)
        
        try:
            # Weighted Least Squares: (H^T W H)^-1 H^T W z
            lhs = H.T @ W @ H
            rhs = H.T @ W @ z
            v_body = np.linalg.solve(lhs, rhs)
            return v_body
        except np.linalg.LinAlgError:
            return None


def process_radar_frames(radar_frames, min_intensity=5.0, min_range=0.2, min_points=5,
                         use_huber=False, huber_delta=1.0):
    """
    Process all radar frames to extract ego-velocity estimates.
    
    Args:
        radar_frames: List of RadarVelocityFrame objects
        min_intensity: Minimum intensity threshold
        min_range: Minimum range threshold (meters)
        min_points: Minimum number of points required
        use_huber: If True, use Huber loss instead of L2
        huber_delta: Huber loss threshold parameter (m/s)
        
    Returns:
        Dictionary with arrays:
            'times_ros': ROS timestamps
            'cpu_cycles': CPU cycle counts
            'vx', 'vy', 'vz': Velocity components
    """
    times_ros = []
    cpu_cycles = []
    vx_list = []
    vy_list = []
    vz_list = []
    
    for frame in radar_frames:
        if frame.velocities is None or frame.intensities is None:
            continue
        
        v_body = solve_ego_velocity_weighted(
            frame.positions,
            frame.velocities,
            frame.intensities,
            min_intensity=min_intensity,
            min_range=min_range,
            min_points=min_points,
            use_huber=use_huber,
            huber_delta=huber_delta
        )
        
        if v_body is not None:
            times_ros.append(frame.timestamp)
            vx_list.append(v_body[0])
            vy_list.append(v_body[1])
            vz_list.append(v_body[2])
            
            if frame.time_cpu_cycles is not None and len(frame.time_cpu_cycles) > 0:
                cpu_cycles.append(frame.time_cpu_cycles[0])
            else:
                cpu_cycles.append(0)
    
    return {
        'times_ros': np.array(times_ros),
        'cpu_cycles': np.array(cpu_cycles),
        'vx': np.array(vx_list),
        'vy': np.array(vy_list),
        'vz': np.array(vz_list)
    }


def integrate_imu_acceleration(imu_data, axis='x'):
    """
    Integrate IMU acceleration to velocity with detrending.
    
    This method integrates raw acceleration and then applies linear detrending
    to the velocity. This is more robust than pre-removing bias for oscillatory
    motion, as it handles both:
    - Integration constant (initial velocity offset)
    - Constant acceleration bias (manifests as linear drift in velocity)
    
    Args:
        imu_data: List of IMUData objects
        axis: Axis to integrate ('x', 'y', or 'z')
        
    Returns:
        Dictionary with:
            'times': Timestamps
            'acceleration': Raw acceleration
            'velocity_raw': Integrated and detrended velocity
            'bias': 0.0 (bias implicitly handled by detrending)
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    idx = axis_map[axis.lower()]
    
    times = np.array([imu.timestamp for imu in imu_data])
    accel = np.array([imu.linear_acceleration[idx] for imu in imu_data])
    
    # Integrate raw acceleration immediately (no pre-bias removal)
    velocity_integrated = cumulative_trapezoid(accel, times, initial=0.0)
    
    # Linear detrend on velocity removes:
    # - Integration constant (initial velocity offset)
    # - Linear trend from constant acceleration bias
    velocity_raw = detrend(velocity_integrated, type='linear')
    
    return {
        'times': times,
        'acceleration': accel,
        'velocity_raw': velocity_raw,
        'bias': 0.0,  # Bias implicitly handled by detrending
    }


def apply_highpass_filter(signal, cutoff_hz, sample_rate_hz, order=2):
    """
    Apply highpass Butterworth filter to remove low-frequency drift.
    
    Args:
        signal: Input signal
        cutoff_hz: Cutoff frequency in Hz
        sample_rate_hz: Sample rate in Hz
        order: Filter order (default: 2)
        
    Returns:
        Filtered signal
    """
    b, a = butter(order, cutoff_hz, btype='high', fs=sample_rate_hz)
    return filtfilt(b, a, signal)


def apply_lowpass_filter(signal, cutoff_hz, sample_rate_hz, order=2):
    """
    Apply lowpass Butterworth filter to reduce high-frequency noise.
    
    Args:
        signal: Input signal
        cutoff_hz: Cutoff frequency in Hz
        sample_rate_hz: Sample rate in Hz
        order: Filter order (default: 2)
        
    Returns:
        Filtered signal
    """
    b, a = butter(order, cutoff_hz, btype='low', fs=sample_rate_hz)
    return filtfilt(b, a, signal)


def filter_valid_cpu_cycles(data_dict):
    """
    Filter out entries with invalid (zero) CPU cycle counts.
    
    Args:
        data_dict: Dictionary with 'cpu_cycles' and other aligned arrays
        
    Returns:
        Filtered dictionary with only valid entries
    """
    valid_mask = data_dict['cpu_cycles'] > 0
    
    return {
        key: value[valid_mask] if isinstance(value, np.ndarray) else value
        for key, value in data_dict.items()
    }


def find_time_shift(t_reference, v_reference, t_sensor, v_sensor, 
                    search_window=(-0.2, 0.2), crop_duration=1.0):
    """
    Find optimal time shift between two velocity signals.
    
    Finds the time shift 'dt' such that v_sensor(t + dt) ~ v_reference(t).
    A positive dt means the sensor data is DELAYED (arrives late).
    
    Args:
        t_reference: Reference timestamps (e.g., IMU)
        v_reference: Reference velocity signal
        t_sensor: Sensor timestamps (e.g., radar)
        v_sensor: Sensor velocity signal
        search_window: Search range for time shift in seconds (min, max)
        crop_duration: Duration in seconds to crop from start/end (0 or False to disable)
        
    Returns:
        Tuple of (optimal_dt, min_rmse, correlation_at_optimum)
    """

    # Create interpolator for sensor data
    sensor_interp = interp1d(t_sensor, v_sensor, kind='linear', 
                            fill_value="extrapolate", bounds_error=False)
    
    # Compute crop indices (0 if cropping disabled)
    if crop_duration:
        sample_rate = len(t_reference) / (t_reference[-1] - t_reference[0])
        crop_samples = int(crop_duration * sample_rate)
    else:
        crop_samples = 0
    
    # Crop edges to avoid filter artifacts (or use full signal if crop_samples=0)
    if crop_samples > 0:
        t_ref_cropped = t_reference[crop_samples:-crop_samples]
        v_ref_cropped = v_reference[crop_samples:-crop_samples]
    else:
        t_ref_cropped = t_reference
        v_ref_cropped = v_reference
    
    # Define cost function (RMSE)
    def cost_function(dt):
        # Query sensor at (t_reference - dt)
        # If dt is negative (delay), we look BACK in sensor time
        # We found the optimal data later in the sensor stream
        v_sensor_shifted = sensor_interp(t_ref_cropped - dt)
        
        # Calculate RMSE
        err = v_ref_cropped - v_sensor_shifted
        return np.sqrt(np.mean(err**2))
    
    # Optimize with higher precision
    result = minimize_scalar(
        cost_function, 
        bounds=search_window, 
        method='bounded',
        options={'xatol': 1e-6}  # Aim for microsecond precision
    )
    
    # Compute correlation at optimum
    v_sensor_optimal = sensor_interp(t_ref_cropped - result.x)
    correlation = np.corrcoef(v_ref_cropped, v_sensor_optimal)[0, 1]
    
    return result.x, result.fun, correlation


def compute_alignment_metrics(t_reference, v_reference, t_sensor, v_sensor, dt_shift, crop_duration=1.0):
    """
    Compute alignment quality metrics after applying time shift.
    
    Args:
        t_reference: Reference timestamps
        v_reference: Reference velocity
        t_sensor: Sensor timestamps
        v_sensor: Sensor velocity
        dt_shift: Time shift to apply (positive = sensor delayed)
        crop_duration: Duration in seconds to crop from edges (0 or False to disable)
        
    Returns:
        Dictionary with rmse, correlation, residuals, aligned signal, and crop_samples
    """
    
    # Apply shift and interpolate
    sensor_interp = interp1d(t_sensor, v_sensor, kind='linear', 
                            fill_value="extrapolate", bounds_error=False)
    v_sensor_aligned = sensor_interp(t_reference - dt_shift)
    
    # Compute crop indices (0 if cropping disabled)
    if crop_duration:
        sample_rate = len(t_reference) / (t_reference[-1] - t_reference[0])
        crop_samples = int(crop_duration * sample_rate)
    else:
        crop_samples = 0
    
    # Compute metrics on cropped data (or full data if crop_samples=0)
    if crop_samples > 0:
        v_ref_cropped = v_reference[crop_samples:-crop_samples]
        v_sensor_cropped = v_sensor_aligned[crop_samples:-crop_samples]
    else:
        v_ref_cropped = v_reference
        v_sensor_cropped = v_sensor_aligned
    
    residuals_cropped = v_ref_cropped - v_sensor_cropped
    rmse = np.sqrt(np.mean(residuals_cropped**2))
    correlation = np.corrcoef(v_ref_cropped, v_sensor_cropped)[0, 1]
    
    # Full residuals for plotting
    residuals_full = v_reference - v_sensor_aligned
    
    return {
        'rmse': rmse,
        'correlation': correlation,
        'residuals': residuals_full,
        'v_sensor_aligned': v_sensor_aligned,
        'crop_samples': crop_samples
    }


def analyze_point_level_noise(radar_frames, t_imu, v_imu, time_shift_dt, 
                               min_range=0.2, percentile_filter=99.5):
    """
    Analyze raw sensor noise at the individual radar point level.
    
    For every point in every frame, compares measured Doppler velocity against
    expected velocity (IMU ground truth projected onto point direction).
    
    Method:
        1. Interpolate IMU velocity at frame timestamp (time-shift corrected)
        2. For each point: compute expected_doppler = direction · v_imu
        3. Compute residual = measured_doppler - expected_doppler
        4. Aggregate all residuals and compute statistics
    
    Args:
        radar_frames: List of RadarVelocityFrame objects
        t_imu: IMU timestamps (numpy array)
        v_imu: IMU velocity reference signal (numpy array, x-axis)
        time_shift_dt: Time shift to apply (from find_time_shift)
        min_range: Minimum range threshold for valid points (meters)
        percentile_filter: Percentile to use for outlier filtering (default 99.5)
        
    Returns:
        Dictionary with:
            'sigma_point': Standard deviation of point residuals (m/s)
            'mean_bias': Mean bias of residuals (m/s)
            'all_residuals': Raw array of all residuals
            'clean_residuals': Filtered residuals (outliers removed)
            'frames_analyzed': Number of frames processed
            'points_analyzed': Total number of points
            'outlier_fraction': Fraction of outliers removed
            'skewness': Skewness of clean residuals
            'kurtosis': Excess kurtosis of clean residuals
    """
    
    # Create interpolator for IMU velocity (ground truth)
    v_imu_interp = interp1d(t_imu, v_imu, kind='linear', fill_value='extrapolate')
    
    # Collect all point-level residuals
    all_residuals = []
    frames_analyzed = 0
    points_analyzed = 0
    
    for frame in radar_frames:
        if frame.velocities is None or len(frame.velocities) == 0:
            continue
        
        # Apply time shift correction
        t_corrected = frame.timestamp - time_shift_dt
        
        # Check if within IMU time bounds
        if t_corrected < t_imu[0] or t_corrected > t_imu[-1]:
            continue
        
        # Get ground truth body velocity at this instant
        v_body_gt_x = float(v_imu_interp(t_corrected))
        
        # Construct 3D body velocity vector
        # (Assuming Y and Z are approximately zero for dominant X-axis motion)
        v_body_gt = np.array([v_body_gt_x, 0.0, 0.0])
        
        # Extract point data
        positions = np.array(frame.positions)
        measured_dopplers = np.array(frame.velocities)
        
        # Calculate ranges and filter valid points
        ranges = np.linalg.norm(positions, axis=1)
        valid_mask = ranges > min_range
        
        if np.sum(valid_mask) == 0:
            continue
        
        # Unit direction vectors
        directions = positions[valid_mask] / ranges[valid_mask, np.newaxis]
        valid_measurements = measured_dopplers[valid_mask]
        
        # Expected Doppler velocity: v_expected = direction · v_body
        expected_dopplers = directions @ v_body_gt
        
        # Residuals: measured - expected
        residuals = valid_measurements - expected_dopplers
        
        all_residuals.extend(residuals)
        frames_analyzed += 1
        points_analyzed += len(residuals)
    
    all_residuals = np.array(all_residuals)
    
    # Filter extreme outliers for cleaner statistics
    # Keep data within specified percentile to remove gross errors
    lower_bound = np.percentile(all_residuals, (100 - percentile_filter) / 2)
    upper_bound = np.percentile(all_residuals, 100 - (100 - percentile_filter) / 2)
    clean_residuals = all_residuals[
        (all_residuals >= lower_bound) & (all_residuals <= upper_bound)
    ]
    
    outlier_fraction = (len(all_residuals) - len(clean_residuals)) / len(all_residuals)
    
    # Compute statistics
    sigma_point = np.std(clean_residuals)
    mean_bias = np.mean(clean_residuals)
    skewness = stats.skew(clean_residuals)
    kurtosis = stats.kurtosis(clean_residuals)
    
    return {
        'sigma_point': sigma_point,
        'mean_bias': mean_bias,
        'all_residuals': all_residuals,
        'clean_residuals': clean_residuals,
        'frames_analyzed': frames_analyzed,
        'points_analyzed': points_analyzed,
        'outlier_fraction': outlier_fraction,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def analyze_point_noise_vs_ground_truth(radar_frames, times_imu, v_imu_x, time_shift_dt,
                                         min_intensity=2.0, min_range=0.2,
                                         outlier_percentile=99.0):
    """
    Calculate raw sensor noise (sigma_point) by comparing individual radar point
    Doppler measurements against interpolated IMU ground truth velocity.
    
    For each radar point in each frame:
    1. Correct frame timestamp: t_true = t_ros + time_shift_dt
    2. Interpolate IMU velocity at t_true
    3. Project IMU velocity onto point's direction: v_expected = r̂ · v_imu
    4. Calculate residual: r = v_measured - v_expected
    5. Aggregate all residuals to estimate sensor noise
    
    Args:
        radar_frames: List of radar frame objects with timestamp, positions, velocities, intensities
        times_imu: Array of IMU timestamps (s)
        v_imu_x: Array of IMU velocity X-axis (m/s) - ground truth
        time_shift_dt: Optimal time shift from find_time_shift (s)
                       Sign convention: t_corrected = t_ros - time_shift_dt
        min_intensity: Filter points below this intensity
        min_range: Filter points below this range (m)
        outlier_percentile: Keep residuals within [1, percentile] for clean statistics
        
    Returns:
        dict with keys:
            'sigma_point': Standard deviation of clean residuals (m/s)
            'mean_bias': Mean of clean residuals (m/s) - should be ~0
            'all_residuals': All residuals before outlier filtering
            'clean_residuals': Residuals after outlier filtering
            'total_points': Total number of points analyzed
            'frames_analyzed': Number of frames with valid data
            'skewness': Skewness of clean residuals
            'kurtosis': Excess kurtosis of clean residuals
    """
    from scipy import stats
    
    # Create interpolator for IMU velocity (ground truth)
    # Assume v_y and v_z are ~0 for X-axis pumping motion
    v_imu_interp = interp1d(times_imu, v_imu_x, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    
    all_residuals = []
    frames_analyzed = 0
    
    for frame in radar_frames:
        if frame.velocities is None or len(frame.velocities) == 0:
            continue
        if frame.positions is None or len(frame.positions) == 0:
            continue
            
        # 1. Correct frame timestamp
        t_corrected = frame.timestamp + time_shift_dt
        
        # 2. Get ground truth body velocity at this instant
        try:
            v_body_gt_x = float(v_imu_interp(t_corrected))
        except (ValueError, RuntimeError):
            continue  # Out of interpolation bounds
            
        # Construct 3D body velocity vector (assume Y and Z ≈ 0 for pumping motion)
        v_body_gt = np.array([v_body_gt_x, 0.0, 0.0])
        
        # 3. Process all points in this frame
        positions = np.array(frame.positions)
        measured_dopplers = np.array(frame.velocities)
        intensities = np.array(frame.intensities) if frame.intensities is not None else np.ones(len(positions))
        
        # Apply filtering thresholds
        ranges = np.linalg.norm(positions, axis=1)
        valid_mask = (ranges >= min_range) & (intensities >= min_intensity)
        
        if np.sum(valid_mask) == 0:
            continue
            
        # Unit direction vectors
        valid_positions = positions[valid_mask]
        valid_ranges = ranges[valid_mask]
        directions = valid_positions / valid_ranges[:, None]
        
        valid_measurements = measured_dopplers[valid_mask]
        
        # 4. Calculate expected Doppler: r̂ · v_body
        expected_dopplers = directions @ v_body_gt
        
        # 5. Calculate residuals: measured - expected
        residuals = valid_measurements - expected_dopplers
        
        all_residuals.extend(residuals)
        frames_analyzed += 1
    
    # Convert to array
    all_residuals = np.array(all_residuals)
    total_points = len(all_residuals)
    
    if total_points == 0:
        return {
            'sigma_point': np.nan,
            'mean_bias': np.nan,
            'all_residuals': np.array([]),
            'clean_residuals': np.array([]),
            'total_points': 0,
            'frames_analyzed': 0,
            'skewness': np.nan,
            'kurtosis': np.nan
        }
    
    # Filter outliers for clean statistics
    lower_bound = np.percentile(all_residuals, 100 - outlier_percentile)
    upper_bound = np.percentile(all_residuals, outlier_percentile)
    clean_residuals = all_residuals[(all_residuals >= lower_bound) & 
                                     (all_residuals <= upper_bound)]
    
    # Calculate statistics
    sigma_point = np.std(clean_residuals)
    mean_bias = np.mean(clean_residuals)
    skewness = stats.skew(clean_residuals)
    kurtosis = stats.kurtosis(clean_residuals)  # Excess kurtosis
    
    return {
        'sigma_point': sigma_point,
        'mean_bias': mean_bias,
        'all_residuals': all_residuals,
        'clean_residuals': clean_residuals,
        'total_points': total_points,
        'frames_analyzed': frames_analyzed,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def analyze_noise_vs_intensity(radar_frames, times_imu, v_imu_x, time_shift_dt,
                                intensity_bins=None, min_range=0.2,
                                outlier_percentile=99.0):
    """
    Analyze how Doppler measurement noise varies with signal intensity.
    
    This helps determine the optimal MIN_INTENSITY threshold by finding the 
    "knee of the curve" where noise stops decreasing significantly with 
    increasing intensity.
    
    For each intensity bin:
    1. Collect all radar points with intensity in that bin
    2. Calculate residuals against IMU ground truth
    3. Compute σ (standard deviation) of residuals
    
    Args:
        radar_frames: List of radar frame objects
        times_imu: Array of IMU timestamps (s)
        v_imu_x: Array of IMU velocity X-axis (m/s) - ground truth
        time_shift_dt: Optimal time shift from find_time_shift (s)
        intensity_bins: Array of bin edges (e.g., [0, 2, 4, 6, 8, 10, 15, 20, 30])
                       If None, uses default: np.arange(0, 31, 1)
        min_range: Filter points below this range (m)
        outlier_percentile: Keep residuals within [1, percentile] for clean statistics
        
    Returns:
        dict with keys:
            'bin_centers': Center of each intensity bin
            'bin_edges': Edges used for binning
            'sigma_per_bin': Standard deviation of residuals in each bin
            'mean_per_bin': Mean of residuals in each bin (bias)
            'count_per_bin': Number of points in each bin
            'all_residuals': List of residual arrays (one per bin)
            'all_intensities': Array of all intensities analyzed
    """
    from scipy import stats
    
    # Default bins: 0-1, 1-2, 2-3, ..., 29-30
    if intensity_bins is None:
        intensity_bins = np.arange(0, 31, 1)
    
    # Create interpolator for IMU velocity (ground truth)
    v_imu_interp = interp1d(times_imu, v_imu_x, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    
    # Collect all points with their intensities and residuals
    all_intensities = []
    all_residuals = []
    
    for frame in radar_frames:
        if frame.velocities is None or len(frame.velocities) == 0:
            continue
        if frame.positions is None or len(frame.positions) == 0:
            continue
        if frame.intensities is None:
            continue
            
        # Correct frame timestamp
        t_corrected = frame.timestamp - time_shift_dt
        
        # Get ground truth body velocity at this instant
        try:
            v_body_gt_x = float(v_imu_interp(t_corrected))
        except (ValueError, RuntimeError):
            continue
            
        # Construct 3D body velocity vector
        v_body_gt = np.array([v_body_gt_x, 0.0, 0.0])
        
        # Process all points in this frame
        positions = np.array(frame.positions)
        measured_dopplers = np.array(frame.velocities)
        intensities = np.array(frame.intensities)
        
        # Apply range filtering only
        ranges = np.linalg.norm(positions, axis=1)
        valid_mask = ranges >= min_range
        
        if np.sum(valid_mask) == 0:
            continue
            
        # Unit direction vectors
        valid_positions = positions[valid_mask]
        valid_ranges = ranges[valid_mask]
        directions = valid_positions / valid_ranges[:, None]
        
        valid_measurements = measured_dopplers[valid_mask]
        valid_intensities = intensities[valid_mask]
        
        # Calculate expected Doppler and residuals
        expected_dopplers = directions @ v_body_gt
        residuals = valid_measurements - expected_dopplers
        
        all_intensities.extend(valid_intensities)
        all_residuals.extend(residuals)
    
    all_intensities = np.array(all_intensities)
    all_residuals = np.array(all_residuals)
    
    # Bin the data by intensity
    bin_indices = np.digitize(all_intensities, intensity_bins) - 1
    
    # Calculate statistics for each bin
    n_bins = len(intensity_bins) - 1
    sigma_per_bin = []
    mean_per_bin = []
    count_per_bin = []
    residuals_per_bin = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        bin_residuals = all_residuals[mask]
        
        if len(bin_residuals) < 10:  # Need at least 10 points for reliable statistics
            sigma_per_bin.append(np.nan)
            mean_per_bin.append(np.nan)
            count_per_bin.append(len(bin_residuals))
            residuals_per_bin.append(bin_residuals)
            continue
        
        # Filter outliers for this bin
        lower = np.percentile(bin_residuals, 100 - outlier_percentile)
        upper = np.percentile(bin_residuals, outlier_percentile)
        clean = bin_residuals[(bin_residuals >= lower) & (bin_residuals <= upper)]
        
        sigma_per_bin.append(np.std(clean))
        mean_per_bin.append(np.mean(clean))
        count_per_bin.append(len(bin_residuals))
        residuals_per_bin.append(bin_residuals)
    
    # Calculate bin centers
    bin_centers = (intensity_bins[:-1] + intensity_bins[1:]) / 2
    
    return {
        'bin_centers': bin_centers,
        'bin_edges': intensity_bins,
        'sigma_per_bin': np.array(sigma_per_bin),
        'mean_per_bin': np.array(mean_per_bin),
        'count_per_bin': np.array(count_per_bin),
        'all_residuals': residuals_per_bin,
        'all_intensities': all_intensities,
        'all_residual_values': all_residuals
    }


# ==================== FORWARD MODEL FUNCTIONS ====================

def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.
    
    Args:
        q: Quaternion as [qx, qy, qz, qw]
        
    Returns:
        3x3 rotation matrix R that rotates vectors from body to world frame
    """
    qx, qy, qz, qw = q
    
    # Normalize
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Rotation matrix (body to world)
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (ZYX convention).
    
    Args:
        roll: Roll angle in radians (rotation around X)
        pitch: Pitch angle in radians (rotation around Y)
        yaw: Yaw angle in radians (rotation around Z)
        
    Returns:
        3x3 rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # ZYX convention (yaw, then pitch, then roll)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    
    return R


def predict_doppler_velocity(
    v_body_world: np.ndarray,
    omega_body: np.ndarray,
    R_world_from_body: np.ndarray,
    radar_positions_sensor: np.ndarray,
    T_body_from_sensor: np.ndarray,
    R_body_from_sensor: np.ndarray
) -> np.ndarray:
    """
    Predict Doppler velocities for radar points using the forward model.
    
    Based on Forward Model.md Section 4:
    1. Calculate antenna velocity in body frame: v_ant = v_body + omega × T_b<-s
    2. Transform ray directions from sensor to body frame: u_b = R_b<-s * u_s
    3. Compute Doppler: v_D = u_b · v_ant
    
    Args:
        v_body_world: Linear velocity of body center in world frame [3]
        omega_body: Angular velocity in body frame [3] rad/s
        R_world_from_body: Rotation matrix from body to world (3x3)
        radar_positions_sensor: Radar point positions in sensor frame (N x 3)
        T_body_from_sensor: Translation vector from body to sensor in body frame [3]
        R_body_from_sensor: Rotation matrix from sensor to body (3x3)
        
    Returns:
        predicted_dopplers: Array of predicted Doppler velocities (N,)
    """
    # Convert body velocity to body frame
    R_body_from_world = R_world_from_body.T
    v_body_in_body = R_body_from_world @ v_body_world
    
    # Calculate lever arm effect: omega × T_b<-s
    lever_arm_velocity = np.cross(omega_body, T_body_from_sensor)
    
    # Total antenna velocity in body frame
    v_ant_body = v_body_in_body + lever_arm_velocity
    
    # Calculate unit direction vectors in sensor frame
    ranges = np.linalg.norm(radar_positions_sensor, axis=1, keepdims=True)
    u_sensor = radar_positions_sensor / ranges
    
    # Transform ray directions to body frame
    u_body = (R_body_from_sensor @ u_sensor.T).T
    
    # Compute Doppler as dot product
    predicted_dopplers = np.sum(u_body * v_ant_body, axis=1)
    
    return predicted_dopplers


def compute_doppler_residuals(
    agiros_states,
    radar_frames,
    T_body_from_sensor: np.ndarray,
    R_body_from_sensor: np.ndarray,
    time_offset: float = 0.0,
    min_range: float = 0.2
) -> Dict[str, Any]:
    """
    Compute residuals between predicted and measured Doppler velocities.
    
    Args:
        agiros_states: List of AgirosState objects (MoCap ground truth)
        radar_frames: List of RadarVelocity objects
        T_body_from_sensor: Translation from body to sensor in body frame [3]
        R_body_from_sensor: Rotation matrix from sensor to body (3x3)
        time_offset: Time offset to add to radar timestamps (seconds)
        min_range: Minimum range threshold for filtering radar points
        
    Returns:
        Dictionary with residuals, predictions, measurements, and statistics
    """
    # Create interpolators for MoCap data
    agiros_times = np.array([s.timestamp for s in agiros_states])
    agiros_positions = np.array([s.position for s in agiros_states])
    agiros_velocities = np.array([s.velocity for s in agiros_states])
    agiros_quats = np.array([s.orientation for s in agiros_states])
    agiros_omegas = np.array([s.angular_velocity for s in agiros_states])
    
    # Create interpolators
    pos_interp = interp1d(agiros_times, agiros_positions, axis=0, kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
    vel_interp = interp1d(agiros_times, agiros_velocities, axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    omega_interp = interp1d(agiros_times, agiros_omegas, axis=0, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    
    # Quaternion SLERP would be better, but linear interpolation is simpler
    quat_interp = interp1d(agiros_times, agiros_quats, axis=0, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    
    all_residuals = []
    all_predictions = []
    all_measurements = []
    all_intensities = []
    all_ranges = []
    frame_indices = []
    point_indices = []
    
    for frame_idx, frame in enumerate(radar_frames):
        # Apply time offset to radar timestamp
        t_corrected = frame.timestamp + time_offset
        
        # Skip if outside MoCap range
        if t_corrected < agiros_times[0] or t_corrected > agiros_times[-1]:
            continue
        
        # Interpolate MoCap state
        try:
            v_world = vel_interp(t_corrected)
            omega_body = omega_interp(t_corrected)
            quat = quat_interp(t_corrected)
        except (ValueError, RuntimeError):
            continue
        
        # Get rotation matrix
        R_world_from_body = quat_to_rotation_matrix(quat)
        
        # Get radar measurements
        positions = np.array(frame.positions)
        measured_dopplers = np.array(frame.velocities)
        intensities = np.array(frame.intensities) if frame.intensities is not None else np.ones(len(positions))
        
        # Filter by range
        ranges = np.linalg.norm(positions, axis=1)
        valid_mask = ranges >= min_range
        
        if np.sum(valid_mask) == 0:
            continue
        
        valid_positions = positions[valid_mask]
        valid_measurements = measured_dopplers[valid_mask]
        valid_intensities = intensities[valid_mask]
        valid_ranges = ranges[valid_mask]
        
        # Predict Doppler velocities
        predicted_dopplers = predict_doppler_velocity(
            v_world, omega_body, R_world_from_body,
            valid_positions, T_body_from_sensor, R_body_from_sensor
        )
        
        # Compute residuals
        residuals = valid_measurements - predicted_dopplers
        
        # Store results
        all_residuals.extend(residuals)
        all_predictions.extend(predicted_dopplers)
        all_measurements.extend(valid_measurements)
        all_intensities.extend(valid_intensities)
        all_ranges.extend(valid_ranges)
        frame_indices.extend([frame_idx] * len(residuals))
        point_indices.extend(np.where(valid_mask)[0])
    
    all_residuals = np.array(all_residuals)
    all_predictions = np.array(all_predictions)
    all_measurements = np.array(all_measurements)
    all_intensities = np.array(all_intensities)
    all_ranges = np.array(all_ranges)
    
    # Compute statistics
    if len(all_residuals) > 0:
        # Remove outliers for stats
        q1, q99 = np.percentile(all_residuals, [1, 99])
        inlier_mask = (all_residuals >= q1) & (all_residuals <= q99)
        
        stats = {
            'mean': np.mean(all_residuals[inlier_mask]),
            'std': np.std(all_residuals[inlier_mask]),
            'rmse': np.sqrt(np.mean(all_residuals[inlier_mask]**2)),
            'median': np.median(all_residuals),
            'q1': q1,
            'q99': q99,
            'num_points': len(all_residuals),
            'num_inliers': np.sum(inlier_mask)
        }
    else:
        stats = None
    
    return {
        'residuals': all_residuals,
        'predictions': all_predictions,
        'measurements': all_measurements,
        'intensities': all_intensities,
        'ranges': all_ranges,
        'frame_indices': frame_indices,
        'point_indices': point_indices,
        'stats': stats
    }


def calibrate_radar_extrinsics_and_timing(
    agiros_states,
    radar_frames,
    initial_translation: np.ndarray = np.array([0.07, 0.0, 0.0]),
    initial_rotation_euler: np.ndarray = np.array([0.0, -30.0 * np.pi/180, 0.0]),
    initial_time_offset: float = -0.018879,
    calibrate_translation: bool = True,
    calibrate_rotation: bool = True,
    calibrate_time: bool = True,
    min_range: float = 0.2
) -> Dict[str, Any]:
    """
    Calibrate radar extrinsics (position and orientation) and time offset.
    
    Uses scipy.optimize.minimize to find parameters that minimize RMSE of
    Doppler residuals.
    
    Args:
        agiros_states: List of AgirosState objects
        radar_frames: List of RadarVelocity objects
        initial_translation: Initial guess for T_body_from_sensor [x, y, z] in meters
        initial_rotation_euler: Initial guess for rotation [roll, pitch, yaw] in radians
        initial_time_offset: Initial guess for time offset in seconds
        calibrate_translation: Whether to optimize translation
        calibrate_rotation: Whether to optimize rotation
        calibrate_time: Whether to optimize time offset
        min_range: Minimum range for filtering radar points
        
    Returns:
        Dictionary with optimized parameters and results
    """
    # Build parameter vector
    param_names = []
    x0 = []
    
    if calibrate_translation:
        param_names.extend(['tx', 'ty', 'tz'])
        x0.extend(initial_translation)
    
    if calibrate_rotation:
        param_names.extend(['roll', 'pitch', 'yaw'])
        x0.extend(initial_rotation_euler)
    
    if calibrate_time:
        param_names.append('time_offset')
        x0.append(initial_time_offset)
    
    x0 = np.array(x0)
    
    if len(x0) == 0:
        raise ValueError("At least one parameter must be calibrated")
    
    # Define cost function
    def cost_function(x):
        # Unpack parameters
        idx = 0
        if calibrate_translation:
            T_body_from_sensor = x[idx:idx+3]
            idx += 3
        else:
            T_body_from_sensor = initial_translation
        
        if calibrate_rotation:
            euler = x[idx:idx+3]
            R_body_from_sensor = rotation_matrix_from_euler(*euler)
            idx += 3
        else:
            R_body_from_sensor = rotation_matrix_from_euler(*initial_rotation_euler)
        
        if calibrate_time:
            time_offset = x[idx]
        else:
            time_offset = initial_time_offset
        
        # Compute residuals
        result = compute_doppler_residuals(
            agiros_states, radar_frames,
            T_body_from_sensor, R_body_from_sensor,
            time_offset, min_range
        )
        
        if result['stats'] is None or result['stats']['num_inliers'] < 100:
            return 1e6  # Penalize bad parameters
        
        # Return RMSE
        return result['stats']['rmse']
    
    # Optimize
    print(f"Optimizing {len(x0)} parameters: {param_names}")
    print(f"Initial parameters: {x0}")
    print(f"Initial cost: {cost_function(x0):.6f}")
    
    result = minimize(
        cost_function,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': True}
    )
    
    # Unpack optimized parameters
    x_opt = result.x
    idx = 0
    
    if calibrate_translation:
        T_opt = x_opt[idx:idx+3]
        idx += 3
    else:
        T_opt = initial_translation
    
    if calibrate_rotation:
        euler_opt = x_opt[idx:idx+3]
        R_opt = rotation_matrix_from_euler(*euler_opt)
        idx += 3
    else:
        euler_opt = initial_rotation_euler
        R_opt = rotation_matrix_from_euler(*initial_rotation_euler)
    
    if calibrate_time:
        time_offset_opt = x_opt[idx]
    else:
        time_offset_opt = initial_time_offset
    
    # Compute final residuals
    final_result = compute_doppler_residuals(
        agiros_states, radar_frames,
        T_opt, R_opt, time_offset_opt, min_range
    )
    
    return {
        'success': result.success,
        'translation': T_opt,
        'rotation_matrix': R_opt,
        'rotation_euler': euler_opt,
        'time_offset': time_offset_opt,
        'initial_cost': cost_function(x0),
        'final_cost': result.fun,
        'iterations': result.nit,
        'residual_stats': final_result['stats'],
        'residual_data': final_result,
        'param_names': param_names,
        'x0': x0,
        'x_opt': x_opt
    }

