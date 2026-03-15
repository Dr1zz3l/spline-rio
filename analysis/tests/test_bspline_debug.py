import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

"""
Debug B-spline derivative calculation with detailed output.
"""

import numpy as np
import matplotlib.pyplot as plt
from bspline_utils import UniformBSpline

# Create a simple test spline
np.random.seed(42)
n_points = 10
degree = 3  # Use cubic for simpler debugging
dt = 0.1

# Create simple control points (sine wave)
control_points = np.zeros((n_points, 3))
for i in range(n_points):
    t = i * dt
    control_points[i, 0] = np.sin(2 * np.pi * t)
    control_points[i, 1] = np.cos(2 * np.pi * t)
    control_points[i, 2] = t

spline = UniformBSpline(control_points, degree, dt)

# Test at a single point
t = 0.5
k = spline.find_knot_span(t)

print(f"Testing at t={t}, knot span k={k}")
print(f"Knots: {spline.knots}")
print(f"Degree: {spline.degree}")
print(f"Control points shape: {spline.control_points.shape}")
print()

# Test 0th derivative
N0 = spline.basis_functions(t, k, derivative=0)
print(f"0th derivative (position basis functions):")
print(f"  Shape: {N0.shape}")
print(f"  Values: {N0}")
print(f"  Sum: {np.sum(N0)} (should be 1.0)")
print()

# Test 1st derivative manually
print("Testing 1st derivative calculation:")

# Get lower degree basis functions
start_idx = k - degree
lower_degree = degree - 1
num_basis = degree + 1 + 1  # (p+1) + 1 for first derivative

print(f"  Need {num_basis} basis functions of degree {lower_degree}")
print(f"  Starting from index {start_idx}")

N_lower = spline._eval_basis_from_index(t, start_idx, lower_degree, num_basis)
print(f"  Lower degree basis values: {N_lower}")
print()

# Manually compute the derivative 
N1_manual = np.zeros(degree + 1)
for i in range(degree + 1):
    idx = start_idx + i
    
    # First term
    denom1 = spline.knots[idx + degree] - spline.knots[idx]
    if abs(denom1) > 1e-10:
        term1 = degree / denom1 * N_lower[i]
        print(f"  i={i}, idx={idx}: first term = {degree}/{denom1:.4f} * {N_lower[i]:.6f} = {term1:.6f}")
    else:
        term1 = 0
        print(f"  i={i}, idx={idx}: first term = 0 (zero denom)")
    
    # Second term
    denom2 = spline.knots[idx + degree + 1] - spline.knots[idx + 1]
    if abs(denom2) > 1e-10:
        term2 = degree / denom2 * N_lower[i + 1]
        print(f"  i={i}, idx={idx}: second term = {degree}/{denom2:.4f} * {N_lower[i+1]:.6f} = {term2:.6f}")
    else:
        term2 = 0
        print(f"  i={i}, idx={idx}: second term = 0 (zero denom)")
    
    N1_manual[i] = term1 - term2
    print(f"  N1_manual[{i}] = {term1:.6f} - {term2:.6f} = {N1_manual[i]:.6f}")
    print()

# Get automatic calculation
N1_auto = spline.basis_functions(t, k, derivative=1)

print(f"Manual 1st derivative: {N1_manual}")
print(f"Auto 1st derivative:   {N1_auto}")
print(f"Difference: {N1_manual - N1_auto}")
print()

# Compute numerical derivative
dt = 1e-6
pos_plus = spline(t + dt, derivative=0)
pos_minus = spline(t - dt, derivative=0)
vel_numerical = (pos_plus - pos_minus) / (2 * dt)

vel_analytical = spline(t, derivative=1)

print(f"Velocity at t={t}:")
print(f"  Numerical:  {vel_numerical}")
print(f"  Analytical: {vel_analytical}")
print(f"  Error: {np.linalg.norm(vel_numerical - vel_analytical):.6f}")
