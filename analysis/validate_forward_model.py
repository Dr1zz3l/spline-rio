"""
Validate Forward Model for Radar-Inertial Odometry

This script validates the forward model defined in Forward Model.md by:
1. Loading MoCap ground truth data (position, velocity, orientation, angular velocity)
2. Using the forward model to predict what Doppler velocities the radar should measure
3. Comparing predictions with actual radar measurements
4. Optionally calibrating radar extrinsics (position, orientation) and time offset

The goal is to isolate the correctness of:
- The forward model equations
- Radar mounting position and orientation (extrinsics)
- Time synchronization between radar and IMU/MoCap

This must work BEFORE attempting the inverse problem (B-spline fitting).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rosbag_loader import load_bag_topics
from radar_velocity_utils import (
    rotation_matrix_from_euler,
    compute_doppler_residuals,
    calibrate_radar_extrinsics_and_timing
)


def plot_residual_analysis(result: Dict[str, Any], title_prefix: str = ""):
    """Create comprehensive plots of Doppler residual analysis."""
    residuals = result['residuals']
    predictions = result['predictions']
    measurements = result['measurements']
    intensities = result['intensities']
    ranges = result['ranges']
    stats = result['stats']
    
    if stats is None or len(residuals) == 0:
        print("No valid data to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title_prefix}Doppler Residual Analysis\n'
                 f'RMSE: {stats["rmse"]:.4f} m/s, Mean: {stats["mean"]:.4f} m/s, '
                 f'Std: {stats["std"]:.4f} m/s, N: {stats["num_points"]}',
                 fontsize=14, fontweight='bold')
    
    # 1. Residual histogram
    ax = axes[0, 0]
    ax.hist(residuals, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(stats['mean'], color='r', linestyle='--', label=f'Mean: {stats["mean"]:.4f}')
    ax.axvline(stats['median'], color='g', linestyle='--', label=f'Median: {stats["median"]:.4f}')
    ax.set_xlabel('Residual (m/s)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Predicted vs Measured scatter
    ax = axes[0, 1]
    ax.scatter(predictions, measurements, alpha=0.3, s=1)
    lims = [min(predictions.min(), measurements.min()), 
            max(predictions.max(), measurements.max())]
    ax.plot(lims, lims, 'r--', label='Perfect prediction')
    ax.set_xlabel('Predicted Doppler (m/s)')
    ax.set_ylabel('Measured Doppler (m/s)')
    ax.set_title('Predicted vs Measured')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 3. Residuals vs Intensity
    ax = axes[0, 2]
    ax.scatter(intensities, residuals, alpha=0.3, s=1)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Residual (m/s)')
    ax.set_title('Residuals vs Intensity')
    ax.grid(True, alpha=0.3)
    
    # 4. Residuals vs Range
    ax = axes[1, 0]
    ax.scatter(ranges, residuals, alpha=0.3, s=1)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Residual (m/s)')
    ax.set_title('Residuals vs Range')
    ax.grid(True, alpha=0.3)
    
    # 5. Q-Q plot for normality check
    ax = axes[1, 1]
    from scipy import stats as sp_stats
    # Use inliers only
    q1, q99 = np.percentile(residuals, [1, 99])
    inlier_residuals = residuals[(residuals >= q1) & (residuals <= q99)]
    sp_stats.probplot(inlier_residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Inliers)')
    ax.grid(True, alpha=0.3)
    
    # 6. Cumulative distribution
    ax = axes[1, 2]
    sorted_residuals = np.sort(np.abs(residuals))
    cumulative = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
    ax.plot(sorted_residuals, cumulative)
    ax.set_xlabel('|Residual| (m/s)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of |Residuals|')
    ax.grid(True, alpha=0.3)
    
    # Add percentile markers
    for p in [50, 90, 95, 99]:
        val = np.percentile(np.abs(residuals), p)
        ax.axvline(val, color='r', linestyle=':', alpha=0.5)
        ax.text(val, p/100, f'{p}%: {val:.3f}', rotation=90, va='bottom')
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 80)
    print("FORWARD MODEL VALIDATION FOR RADAR-INERTIAL ODOMETRY")
    print("=" * 80)
    
    # ==================== Configuration ====================
    BAG_PATH = r"C:\Users\luchs\MyData\Education\Master_TUM\25WS\Guided Research\radar-iwr6843-driver\rosbags\2025-12-17-16-02-22.bag"
    START_TIME_OFFSET = 31.5  # Skip first N seconds
    DURATION = 15.0           # Analyze this many seconds
    
    # Initial extrinsics from documentation
    # "The radar is mounted tilting downwards 30deg from horizontal, about 7cm forward in x-axis"
    INITIAL_TRANSLATION = np.array([0.07, 0.0, 0.0])  # 7cm forward in body frame
    INITIAL_ROTATION_EULER = np.array([0.0, -30.0 * np.pi/180, 0.0])  # 30deg pitch down
    INITIAL_TIME_OFFSET = -0.018879  # From previous analysis
    
    MIN_RANGE = 0.2  # Minimum range for filtering radar points (meters)
    
    print(f"\n{'Dataset Configuration':-^80}")
    print(f"Bag file: {Path(BAG_PATH).name}")
    print(f"Time window: {START_TIME_OFFSET:.1f}s to {START_TIME_OFFSET + DURATION:.1f}s")
    print(f"Duration: {DURATION:.1f}s")
    
    print(f"\n{'Initial Extrinsics':-^80}")
    print(f"Translation (body frame): {INITIAL_TRANSLATION} m")
    print(f"Rotation (roll, pitch, yaw): {np.rad2deg(INITIAL_ROTATION_EULER)} deg")
    print(f"Time offset: {INITIAL_TIME_OFFSET:.6f} s")
    print(f"Min range filter: {MIN_RANGE} m")
    
    # ==================== Load Data ====================
    print(f"\n{'Loading ROS Bag Data':-^80}")
    
    bag_data = load_bag_topics(BAG_PATH, verbose=True)
    
    # Filter data by time window
    t_start = bag_data.start_time + START_TIME_OFFSET
    t_end = t_start + DURATION
    
    print(f"\nFiltering data to time window: [{t_start:.2f}, {t_end:.2f}]")
    
    agiros_states = [s for s in bag_data.agiros_state if t_start <= s.timestamp <= t_end]
    radar_frames = [f for f in bag_data.radar_velocity if t_start <= f.timestamp <= t_end]
    
    print(f"\nLoaded {len(agiros_states)} MoCap states")
    print(f"Loaded {len(radar_frames)} radar frames")
    
    if len(agiros_states) == 0:
        print("ERROR: No MoCap data loaded!")
        return
    
    if len(radar_frames) == 0:
        print("ERROR: No radar data loaded!")
        return
    
    # ==================== Validate with Initial Parameters ====================
    print(f"\n{'Step 1: Validate Forward Model with Initial Parameters':#^80}")
    
    R_body_from_sensor = rotation_matrix_from_euler(*INITIAL_ROTATION_EULER)
    
    initial_result = compute_doppler_residuals(
        agiros_states,
        radar_frames,
        INITIAL_TRANSLATION,
        R_body_from_sensor,
        INITIAL_TIME_OFFSET,
        min_range=MIN_RANGE
    )
    
    if initial_result['stats'] is not None:
        stats = initial_result['stats']
        print(f"\n{'Initial Parameters Performance':-^80}")
        print(f"Total points: {stats['num_points']}")
        print(f"Inlier points (1-99 percentile): {stats['num_inliers']}")
        print(f"Mean residual: {stats['mean']:.4f} m/s")
        print(f"Std residual: {stats['std']:.4f} m/s")
        print(f"RMSE: {stats['rmse']:.4f} m/s")
        print(f"Median residual: {stats['median']:.4f} m/s")
        print(f"1st percentile: {stats['q1']:.4f} m/s")
        print(f"99th percentile: {stats['q99']:.4f} m/s")
        
        # Plot initial results
        fig1 = plot_residual_analysis(initial_result, "Initial Parameters: ")
        plt.savefig('forward_model_validation_initial.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved plot: forward_model_validation_initial.png")
    else:
        print("ERROR: No valid residuals computed with initial parameters!")
        return
    
    # ==================== Decision: Calibrate or Not? ====================
    print(f"\n{'Step 2: Assess Need for Calibration':#^80}")
    
    # Heuristic: If RMSE < 0.5 m/s and |mean| < 0.1 m/s, parameters are probably good
    needs_calibration = False
    if stats['rmse'] > 0.5:
        print(f"❌ RMSE ({stats['rmse']:.4f} m/s) is high (> 0.5 m/s)")
        needs_calibration = True
    else:
        print(f"✓ RMSE ({stats['rmse']:.4f} m/s) is acceptable (< 0.5 m/s)")
    
    if abs(stats['mean']) > 0.1:
        print(f"❌ Mean bias ({stats['mean']:.4f} m/s) is significant (|mean| > 0.1 m/s)")
        needs_calibration = True
    else:
        print(f"✓ Mean bias ({stats['mean']:.4f} m/s) is small (|mean| < 0.1 m/s)")
    
    if needs_calibration:
        print("\n⚠️  CALIBRATION RECOMMENDED")
    else:
        print("\n✅ INITIAL PARAMETERS ARE GOOD - NO CALIBRATION NEEDED")
    
    # ==================== Calibration (if needed) ====================
    if needs_calibration:
        print(f"\n{'Step 3: Calibrate Extrinsics and Time Offset':#^80}")
        
        # First: Calibrate time offset only
        print(f"\n{'Calibrating Time Offset Only':-^80}")
        time_calib = calibrate_radar_extrinsics_and_timing(
            agiros_states,
            radar_frames,
            initial_translation=INITIAL_TRANSLATION,
            initial_rotation_euler=INITIAL_ROTATION_EULER,
            initial_time_offset=INITIAL_TIME_OFFSET,
            calibrate_translation=False,
            calibrate_rotation=False,
            calibrate_time=True,
            min_range=MIN_RANGE
        )
        
        print(f"\nTime Offset Calibration Results:")
        print(f"Initial time offset: {INITIAL_TIME_OFFSET:.6f} s")
        print(f"Optimized time offset: {time_calib['time_offset']:.6f} s")
        print(f"Change: {time_calib['time_offset'] - INITIAL_TIME_OFFSET:.6f} s")
        print(f"RMSE improvement: {time_calib['initial_cost']:.4f} → {time_calib['final_cost']:.4f} m/s")
        
        # Then: Calibrate all parameters
        print(f"\n{'Calibrating All Parameters (Translation, Rotation, Time)':-^80}")
        full_calib = calibrate_radar_extrinsics_and_timing(
            agiros_states,
            radar_frames,
            initial_translation=INITIAL_TRANSLATION,
            initial_rotation_euler=INITIAL_ROTATION_EULER,
            initial_time_offset=time_calib['time_offset'],  # Use time-calibrated offset
            calibrate_translation=True,
            calibrate_rotation=True,
            calibrate_time=True,
            min_range=MIN_RANGE
        )
        
        print(f"\n{'Full Calibration Results':-^80}")
        print(f"Success: {full_calib['success']}")
        print(f"Iterations: {full_calib['iterations']}")
        print(f"RMSE improvement: {full_calib['initial_cost']:.4f} → {full_calib['final_cost']:.4f} m/s")
        
        print(f"\n{'Parameter Changes':-^80}")
        print(f"Translation:")
        print(f"  Initial: {INITIAL_TRANSLATION}")
        print(f"  Final:   {full_calib['translation']}")
        print(f"  Change:  {full_calib['translation'] - INITIAL_TRANSLATION}")
        
        print(f"\nRotation (Euler angles in degrees):")
        print(f"  Initial: {np.rad2deg(INITIAL_ROTATION_EULER)}")
        print(f"  Final:   {np.rad2deg(full_calib['rotation_euler'])}")
        print(f"  Change:  {np.rad2deg(full_calib['rotation_euler'] - INITIAL_ROTATION_EULER)}")
        
        print(f"\nTime Offset:")
        print(f"  Initial: {INITIAL_TIME_OFFSET:.6f} s")
        print(f"  Final:   {full_calib['time_offset']:.6f} s")
        print(f"  Change:  {full_calib['time_offset'] - INITIAL_TIME_OFFSET:.6f} s")
        
        # Plot calibrated results
        fig2 = plot_residual_analysis(full_calib['residual_data'], "Calibrated Parameters: ")
        plt.savefig('forward_model_validation_calibrated.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved plot: forward_model_validation_calibrated.png")
        
        # ==================== Summary ====================
        print(f"\n{'FINAL SUMMARY':#^80}")
        print(f"\n{'Initial Parameters':-^80}")
        print(f"RMSE: {stats['rmse']:.4f} m/s")
        print(f"Mean: {stats['mean']:.4f} m/s")
        print(f"Std:  {stats['std']:.4f} m/s")
        
        print(f"\n{'Optimized Parameters':-^80}")
        calib_stats = full_calib['residual_stats']
        print(f"RMSE: {calib_stats['rmse']:.4f} m/s")
        print(f"Mean: {calib_stats['mean']:.4f} m/s")
        print(f"Std:  {calib_stats['std']:.4f} m/s")
        
        improvement_pct = (stats['rmse'] - calib_stats['rmse']) / stats['rmse'] * 100
        print(f"\nRMSE Improvement: {improvement_pct:.1f}%")
        
        # Save calibration results to file
        calib_file = Path("forward_model_calibration_results.txt")
        with open(calib_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RADAR EXTRINSIC AND TIME OFFSET CALIBRATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OPTIMIZED PARAMETERS (copy these to your config):\n")
            f.write("-" * 80 + "\n")
            f.write(f"T_BODY_FROM_SENSOR = np.array([{full_calib['translation'][0]:.6f}, "
                   f"{full_calib['translation'][1]:.6f}, {full_calib['translation'][2]:.6f}])  # meters\n")
            f.write(f"ROTATION_EULER = np.array([{full_calib['rotation_euler'][0]:.6f}, "
                   f"{full_calib['rotation_euler'][1]:.6f}, {full_calib['rotation_euler'][2]:.6f}])  # radians\n")
            f.write(f"ROTATION_EULER_DEG = np.array([{np.rad2deg(full_calib['rotation_euler'][0]):.3f}, "
                   f"{np.rad2deg(full_calib['rotation_euler'][1]):.3f}, "
                   f"{np.rad2deg(full_calib['rotation_euler'][2]):.3f}])  # degrees\n")
            f.write(f"TIME_OFFSET = {full_calib['time_offset']:.9f}  # seconds\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Initial RMSE: {stats['rmse']:.4f} m/s\n")
            f.write(f"Final RMSE:   {calib_stats['rmse']:.4f} m/s\n")
            f.write(f"Improvement:  {improvement_pct:.1f}%\n")
            f.write(f"\nFinal Mean Bias: {calib_stats['mean']:.4f} m/s\n")
            f.write(f"Final Std Dev:   {calib_stats['std']:.4f} m/s\n")
            
        print(f"\nSaved calibration parameters to: {calib_file}")
        
    else:
        print(f"\n{'FINAL SUMMARY':#^80}")
        print("✅ Initial parameters validated successfully")
        print("✅ Forward model is correctly implemented")
        print("✅ Extrinsics and time offset are accurate")
        print("\nYou can proceed with confidence to the inverse problem (B-spline fitting)!")
    
    plt.show()
    print(f"\n{'VALIDATION COMPLETE':#^80}")


if __name__ == "__main__":
    main()
