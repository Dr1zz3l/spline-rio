import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""
Test and validate B-spline implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from bspline_utils import UniformBSpline


def test_bspline_basics():
    """Test basic B-spline evaluation and derivatives."""
    print("=" * 80)
    print("B-SPLINE VALIDATION TESTS")
    print("=" * 80)
    
    # Create a simple test case: straight line motion
    degree = 5
    n_points = 20
    dt = 0.1
    
    # Control points forming a simple trajectory (sinusoidal in x, linear in y, constant z)
    t_control = np.arange(n_points) * dt
    x = np.sin(0.5 * t_control)
    y = t_control * 0.5
    z = np.ones_like(t_control) * 2.0
    
    control_points = np.column_stack([x, y, z])
    
    bspline = UniformBSpline(control_points, degree, dt)
    
    print(f"\nTest Configuration:")
    print(f"  Degree: {degree}")
    print(f"  Control points: {n_points}")
    print(f"  dt: {dt}")
    print(f"  Valid time range: [{bspline.t_start:.2f}, {bspline.t_end:.2f}]")
    
    # Evaluate at many points (avoid exact boundaries for numerical stability)
    epsilon = (bspline.t_end - bspline.t_start) * 0.001
    t_eval = np.linspace(bspline.t_start + epsilon, bspline.t_end - epsilon, 200)
    
    positions = np.array([bspline(t, derivative=0) for t in t_eval])
    velocities = np.array([bspline(t, derivative=1) for t in t_eval])
    accelerations = np.array([bspline(t, derivative=2) for t in t_eval])
    
    # Numerical derivative check
    dt_num = t_eval[1] - t_eval[0]
    vel_numerical = np.gradient(positions, dt_num, axis=0)
    acc_numerical = np.gradient(velocities, dt_num, axis=0)
    
    # Compare analytical vs numerical derivatives
    vel_error = np.linalg.norm(velocities - vel_numerical, axis=1)
    acc_error = np.linalg.norm(accelerations - acc_numerical, axis=1)
    
    print(f"\nDerivative Validation:")
    print(f"  Velocity error (analytical vs numerical):")
    print(f"    Mean: {vel_error.mean():.6f}, Max: {vel_error.max():.6f}")
    print(f"  Acceleration error (analytical vs numerical):")
    print(f"    Mean: {acc_error.mean():.6f}, Max: {acc_error.max():.6f}")
    
    if vel_error.mean() > 0.01:
        print("  ⚠️  WARNING: Large velocity derivative error!")
    else:
        print("  ✅ Velocity derivatives look good")
    
    if acc_error.mean() > 0.1:
        print("  ⚠️  WARNING: Large acceleration derivative error!")
    else:
        print("  ✅ Acceleration derivatives look good")
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Position
    ax = axes[0, 0]
    ax.plot(t_eval, positions[:, 0], 'b-', label='X')
    ax.plot(t_eval, positions[:, 1], 'g-', label='Y')
    ax.plot(t_eval, positions[:, 2], 'r-', label='Z')
    ax.scatter(t_control, control_points[:, 0], c='b', marker='x', s=30)
    ax.scatter(t_control, control_points[:, 1], c='g', marker='x', s=30)
    ax.scatter(t_control, control_points[:, 2], c='r', marker='x', s=30)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position (with control points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity analytical
    ax = axes[1, 0]
    ax.plot(t_eval, velocities[:, 0], 'b-', label='X')
    ax.plot(t_eval, velocities[:, 1], 'g-', label='Y')
    ax.plot(t_eval, velocities[:, 2], 'r-', label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity (Analytical)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity numerical
    ax = axes[1, 1]
    ax.plot(t_eval, vel_numerical[:, 0], 'b--', label='X (numerical)')
    ax.plot(t_eval, vel_numerical[:, 1], 'g--', label='Y (numerical)')
    ax.plot(t_eval, vel_numerical[:, 2], 'r--', label='Z (numerical)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity (Numerical from position)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Velocity error
    ax = axes[0, 1]
    ax.plot(t_eval, vel_error, 'k-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m/s)')
    ax.set_title('Velocity Error (Analytical - Numerical)')
    ax.grid(True, alpha=0.3)
    
    # Acceleration analytical
    ax = axes[2, 0]
    ax.plot(t_eval, accelerations[:, 0], 'b-', label='X')
    ax.plot(t_eval, accelerations[:, 1], 'g-', label='Y')
    ax.plot(t_eval, accelerations[:, 2], 'r-', label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Acceleration (Analytical)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Acceleration numerical
    ax = axes[2, 1]
    ax.plot(t_eval, acc_numerical[:, 0], 'b--', label='X (numerical)')
    ax.plot(t_eval, acc_numerical[:, 1], 'g--', label='Y (numerical)')
    ax.plot(t_eval, acc_numerical[:, 2], 'r--', label='Z (numerical)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Acceleration (Numerical from velocity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bspline_validation_test.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: bspline_validation_test.png")
    
    return vel_error.mean() < 0.01 and acc_error.mean() < 0.1


if __name__ == "__main__":
    success = test_bspline_basics()
    
    print(f"\n{'='*80}")
    if success:
        print("✅ B-SPLINE VALIDATION PASSED")
    else:
        print("❌ B-SPLINE VALIDATION FAILED - Derivatives have errors!")
    print("="*80)
