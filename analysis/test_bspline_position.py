"""
Test that position evaluation works correctly after Cox-de Boor fix.
"""

import numpy as np
from bspline_utils import UniformBSpline

# Create a simple linear motion spline
n_points = 5
degree = 3
dt = 1.0

# Linear motion: p(t) = t
control_points = np.zeros((n_points, 3))
for i in range(n_points):
    t = i * dt
    control_points[i, 0] = t  # x = t
    control_points[i, 1] = 0  # y = 0
    control_points[i, 2] = 0  # z = 0

bspline = UniformBSpline(control_points, degree, dt)

print("Control points (x-coordinate):", control_points[:, 0])
print(f"Spline range: [{bspline.t_start}, {bspline.t_end}]")
print(f"Knots: {bspline.knots}")
print(f"Number of knots: {len(bspline.knots)}")
print()

# Test at several points
test_times = [3.5, 4.0, 4.5]

for t in test_times:
    pos = bspline(t, derivative=0)
    k = bspline.find_knot_span(t)
    N = bspline.basis_functions(t, k, derivative=0)
    
    # Manually compute position
    manual_pos = 0.0
    for i in range(degree + 1):
        idx = k - degree + i
        if 0 <= idx < len(control_points):
            manual_pos += N[i] * control_points[idx, 0]
            print(f"  Basis[{i}] = {N[i]:.4f} * CP[{idx}] = {N[i]:.4f} * {control_points[idx, 0]} = {N[i] * control_points[idx, 0]:.4f}")
    
    print(f"t={t}, k={k}: position={pos[0]:.4f}, manual={manual_pos:.4f}")
    print()

print("\nTesting partition of unity:")
for t in test_times:
    k = bspline.find_knot_span(t)
    N = bspline.basis_functions(t, k, derivative=0)
    sum_N = np.sum(N)
    print(f"t={t}, k={k}: basis sum = {sum_N:.10f} (should be 1.0)")
    if abs(sum_N - 1.0) > 1e-10:
        print(f"  ⚠️ Partition of unity violated!")
        print(f"  Basis values: {N}")
