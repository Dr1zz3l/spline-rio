"""
Simple test to check velocity derivative calculation.
"""

import numpy as np
from bspline_utils import UniformBSpline

# Create a very simple spline - quadratic forward motion
n_points = 5
degree = 3
dt = 1.0

# Simple parabolic motion: p(t) = t^2
control_points = np.zeros((n_points, 3))
for i in range(n_points):
    t = i * dt
    control_points[i, 0] = t ** 2
    control_points[i, 1] = 0
    control_points[i, 2] = 0

bspline = UniformBSpline(control_points, degree, dt)

print(f"Spline valid range: [{bspline.t_start}, {bspline.t_end}]")
print(f"Knots: {bspline.knots}")
print()

# Test at several points - avoid exact boundaries
test_times = np.linspace(bspline.t_start + 0.001, bspline.t_end - 0.001, 20)

print("Testing velocity derivatives:")
print("="*60)

for t in test_times:
    pos = bspline(t, derivative=0)
    vel_analytical = bspline(t, derivative=1)
    
    # Numerical velocity using centered difference
    h = 1e-6
    pos_plus = bspline(min(t + h, bspline.t_end), derivative=0)
    pos_minus = bspline(max(t - h, bspline.t_start), derivative=0)
    vel_numerical = (pos_plus - pos_minus) / (2 * h)
    
    error = np.linalg.norm(vel_analytical - vel_numerical)
    
    print(f"t={t:.3f}: pos={pos[0]:.4f}, vel_ana={vel_analytical[0]:.4f}, "
          f"vel_num={vel_numerical[0]:.4f}, error={error:.8f}")
    
    if error > 0.01:
        print(f"  ⚠️ Large error!")

print("\n" + "="*60)
print("Testing acceleration derivatives:")
print("="*60)

for t in test_times:
    vel = bspline(t, derivative=1)
    acc_analytical = bspline(t, derivative=2)
    
    # Numerical acceleration
    h = 1e-6
    vel_plus = bspline(min(t + h, bspline.t_end), derivative=1)
    vel_minus = bspline(max(t - h, bspline.t_start), derivative=1)
    acc_numerical = (vel_plus - vel_minus) / (2 * h)
    
    error = np.linalg.norm(acc_analytical - acc_numerical)
    
    print(f"t={t:.3f}: vel={vel[0]:.4f}, acc_ana={acc_analytical[0]:.4f}, "
          f"acc_num={acc_numerical[0]:.4f}, error={error:.8f}")
    
    if error > 0.1:
        print(f"  ⚠️ Large error!")
