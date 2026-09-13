import numpy as np
import matplotlib.pyplot as plt

def basis_function(t, i, p, knots):
    """
    Calculate N_{i,p}(t) recursively using Cox-de Boor.
    
    Args:
        t: Time point (scalar).
        i: Index of the knot/control point.
        p: Degree of the B-spline.
        knots: Knot vector.
        
    Returns:
        Value of the basis function at t.
    """
    # Base case: degree 0
    if p == 0:
        if knots[i] <= t < knots[i+1]:
            return 1.0
        else:
            return 0.0
    
    # Recursive step
    # Term 1
    denom1 = knots[i+p] - knots[i]
    if denom1 == 0:
        term1 = 0
    else:
        term1 = (t - knots[i]) / denom1 * basis_function(t, i, p-1, knots)
        
    # Term 2
    denom2 = knots[i+p+1] - knots[i+1]
    if denom2 == 0:
        term2 = 0
    else:
        term2 = (knots[i+p+1] - t) / denom2 * basis_function(t, i+1, p-1, knots)
        
    return term1 + term2

def basis_derivative(t, i, p, knots):
    """
    Calculate N'_{i,p}(t).
    
    Args:
        t: Time point (scalar).
        i: Index of the knot/control point.
        p: Degree of the B-spline.
        knots: Knot vector.
    """
    if p == 0:
        return 0.0 # Derivative of step function is 0 (almost everywhere)
        
    # Term 1
    denom1 = knots[i+p] - knots[i]
    if denom1 == 0:
        term1 = 0
    else:
        term1 = basis_function(t, i, p-1, knots) / denom1
        
    # Term 2
    denom2 = knots[i+p+1] - knots[i+1]
    if denom2 == 0:
        term2 = 0
    else:
        term2 = basis_function(t, i+1, p-1, knots) / denom2
        
    return p * (term1 - term2)

def generate_design_matrix(times, knots, degree):
    num_bases = len(knots) - degree - 1
    # We want to fit Velocity, so we need the Derivative of the Position Basis
    H = np.zeros((len(times), num_bases))
    
    for row, t in enumerate(times):
        for col in range(num_bases):
            # Check support to optimize
            # Support of N_{i,p} is [u_i, u_{i+p+1})
            if knots[col] <= t < knots[col+degree+1]:
                H[row, col] = basis_derivative(t, col, degree, knots)
                
    return H

def main():
    print("Testing B-Spline Logic...")
    
    # 1. Setup
    degree = 3
    # Uniform knot vector
    knots = np.arange(0, 11, 1) # [0, 1, ..., 10]
    # Corresponding number of basis functions
    num_bases = len(knots) - degree - 1
    print(f"Knots: {knots}")
    print(f"Number of Basis Functions: {num_bases}")
    
    # 2. Ground Truth Trajectory (Position)
    # Let's say P(t) = sin(t)
    # We want to recover this from Velocity data V(t) = cos(t)
    
    # 3. Simulate Data
    # Random timestamps between start and end (with some margin for order)
    t_min, t_max = knots[degree], knots[-degree-1] # Valid domain
    print(f"Valid Domain: [{t_min}, {t_max}]")
    
    num_samples = 50
    t_samples = np.sort(np.random.uniform(t_min, t_max, num_samples))
    v_ground_truth = np.cos(t_samples)
    noise = np.random.normal(0, 0.05, num_samples)
    v_measured = v_ground_truth + noise
    
    # 4. Build Matrix H (Jacobian)
    # H * ControlPoints = Velocity
    # H contains derivatives of basis functions
    print("Building Jacobian H...")
    H = generate_design_matrix(t_samples, knots, degree)
    
    print(f"H shape: {H.shape}")
    print(f"H sparsity: {np.count_nonzero(H)} / {H.size} elements")
    
    # 5. Solve Linear System
    # H * P = v
    # Use Regularization: min ||HP - v||^2 + lambda ||QP||^2
    # Simple Tikhonov for now: min ||HP - v||^2 + lambda ||P||^2
    
    lmbda = 0.01
    reg_matrix = np.eye(num_bases) * np.sqrt(lmbda)
    
    # Augmented least squares
    # [H     ] * P = [v]
    # [sqrt(L)*I]     [0]
    
    lhs = np.vstack([H, reg_matrix])
    rhs = np.concatenate([v_measured, np.zeros(num_bases)])
    
    print("Solving Least Squares...")
    control_points, residuals, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)
    
    print(f"Solved Control Points: {control_points}")
    
    # 6. Verify Results
    # Reconstruct Velocity
    v_reconstructed = H @ control_points
    
    residuals = v_reconstructed - v_measured
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"RMSE: {rmse}")
    
    if rmse < 0.2:
        print("SUCCESS: RMSE is low, logic seems correct.")
    else:
        print("FAILURE: RMSE is high, check logic.")

if __name__ == "__main__":
    main()
