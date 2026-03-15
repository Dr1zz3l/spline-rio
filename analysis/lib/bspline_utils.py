"""
B-spline utilities for trajectory estimation.

Implements uniform B-splines of arbitrary degree with:
- Cox-de Boor recursion for basis function evaluation
- Analytical derivatives for velocity, acceleration, jerk, snap
- Minimum snap regularization
- Sparse Jacobian construction
"""

import numpy as np
from scipy import sparse
from typing import Tuple, List, Optional


class UniformBSpline:
    """
    Uniform B-spline curve in 3D.
    
    Supports arbitrary degree p:
    - p=3: Cubic (continuous acceleration)
    - p=4: Quartic (continuous jerk)
    - p=5: Quintic (continuous snap)
    """
    
    def __init__(self, control_points: np.ndarray, degree: int, dt: float):
        """
        Initialize uniform B-spline.
        
        Args:
            control_points: (N, 3) array of control points
            degree: Degree of the spline (3, 4, or 5 recommended)
            dt: Time spacing between knots
        """
        self.control_points = np.array(control_points)
        self.n_points = len(control_points)
        self.degree = degree
        self.dt = dt
        
        # Build uniform knot vector
        # For uniform B-spline with n control points and degree p,
        # we need n + p + 1 knots
        n_knots = self.n_points + degree + 1
        self.knots = np.arange(n_knots) * dt
        
        # Valid time range (interior knots)
        self.t_start = self.knots[degree]
        self.t_end = self.knots[-degree-1]
    
    def find_knot_span(self, t: float) -> int:
        """
        Find the knot span index for time t.
        Returns k such that t is in [knots[k], knots[k+1]).
        """
        # Clamp to valid range
        if t <= self.t_start:
            return self.degree
        if t >= self.t_end:
            return self.n_points - 1
        
        # Binary search
        k = np.searchsorted(self.knots, t, side='right') - 1
        
        # Ensure k is in valid range
        k = max(self.degree, min(k, self.n_points - 1))
        
        return k
    
    def _eval_basis_from_index(self, t: float, start_idx: int, degree: int, num_basis: int) -> np.ndarray:
        """
        Evaluate an array of basis functions starting from a specific index.
        
        Evaluates N_{start_idx,degree}(t), ..., N_{start_idx+num_basis-1,degree}(t).
        
        Args:
            t: Time to evaluate
            start_idx: Index of first basis function to evaluate
            degree: Degree of basis functions
            num_basis: Number of basis functions to evaluate
            
        Returns:
            Array of basis function values
        """
        N = np.zeros((degree + 1, num_basis))
        
        # Initialize degree 0 basis functions
        for j in range(num_basis):
            idx = start_idx + j
            if 0 <= idx < len(self.knots) - 1:
                if self.knots[idx] <= t < self.knots[idx + 1]:
                    N[0, j] = 1.0
                elif idx == len(self.knots) - 2 and abs(t - self.knots[idx + 1]) < 1e-10:
                    N[0, j] = 1.0
        
        # Cox-de Boor recursion
        for deg in range(1, degree + 1):
            for j in range(num_basis):
                idx = start_idx + j
                
                # Left term: (t - t_i) / (t_{i+deg} - t_i) * N_{i,deg-1}
                if 0 <= idx < len(self.knots) - deg - 1:
                    denom = self.knots[idx + deg] - self.knots[idx]
                    if abs(denom) > 1e-10:
                        N[deg, j] += (t - self.knots[idx]) / denom * N[deg - 1, j]
                
                # Right term: (t_{i+deg+1} - t) / (t_{i+deg+1} - t_{i+1}) * N_{i+1,deg-1}
                if j + 1 < num_basis and 0 <= idx + 1 < len(self.knots) - deg:
                    denom = self.knots[idx + deg + 1] - self.knots[idx + 1]
                    if abs(denom) > 1e-10:
                        N[deg, j] += (self.knots[idx + deg + 1] - t) / denom * N[deg - 1, j + 1]
        
        return N[degree, :]
    
    def basis_functions(self, t: float, k: int, derivative: int = 0) -> np.ndarray:
        """
        Evaluate basis functions using Cox-de Boor recursion.
        
        For derivative d, computes d-th derivatives of N_{k-p,p}, ..., N_{k,p}
        using the formula:
        N'_{i,p} = p/(t_{i+p}-t_i) * N_{i,p-1} - p/(t_{i+p+1}-t_{i+1}) * N_{i+1,p-1}
        
        Args:
            t: Time to evaluate
            k: Knot span index
            derivative: Order of derivative (0 = position, 1 = velocity, etc.)
            
        Returns:
            Array of (degree+1) basis function values for indices [k-degree, ..., k]
        """
        p = self.degree
        
        if derivative < 0 or derivative > p:
            return np.zeros(p + 1)
        
        if derivative == 0:
            # Evaluate N_{k-p,p}, ..., N_{k,p}
            return self._eval_basis_from_index(t, k - p, p, p + 1)
        
        # For the d-th derivative of degree-p functions N_{k-p,p}, ..., N_{k,p},
        # we apply the derivative formula d times.
        #
        # Start with degree (p-d) functions. To get d-th derivative of N_{k-p,p},...,N_{k,p},
        # we need (d-th derivative is computed from (p-d)-degree functions).
        # Working backwards: we need basis functions N_{k-p, p-d}, ..., N_{k+d, p-d}
        # That's (p+1+d) functions of degree (p-d).
        
        start_idx = k - p
        current_degree = p - derivative
        current_basis = self._eval_basis_from_index(t, start_idx, current_degree, p + 1 + derivative)
        
        # Apply the derivative formula 'derivative' times
        for deriv_step in range(derivative):
            next_degree = current_degree + 1
            next_basis = np.zeros(len(current_basis) - 1)
            
            for i in range(len(next_basis)):
                idx = start_idx + i
                
                # First term: degree/(t_{i+degree}-t_i) * N_{i,degree-1}
                if 0 <= idx < len(self.knots) - next_degree:
                    denom = self.knots[idx + next_degree] - self.knots[idx]
                    if abs(denom) > 1e-10:
                        next_basis[i] += next_degree / denom * current_basis[i]
                
                # Second term: -degree/(t_{i+degree+1}-t_{i+1}) * N_{i+1,degree-1}
                if 0 <= idx + 1 < len(self.knots) - next_degree:
                    denom = self.knots[idx + next_degree + 1] - self.knots[idx + 1]
                    if abs(denom) > 1e-10:
                        next_basis[i] -= next_degree / denom * current_basis[i + 1]
            
            current_basis = next_basis
            current_degree = next_degree
        
        return current_basis
    
    def __call__(self, t: float, derivative: int = 0) -> np.ndarray:
        """
        Evaluate spline at time t.
        
        Args:
            t: Time to evaluate
            derivative: Order of derivative (0=pos, 1=vel, 2=acc, 3=jerk, 4=snap)
            
        Returns:
            3D vector (position, velocity, acceleration, etc.)
        """
        k = self.find_knot_span(t)
        N = self.basis_functions(t, k, derivative)
        
        # Sum weighted control points
        result = np.zeros(3)
        for i in range(self.degree + 1):
            idx = k - self.degree + i
            if 0 <= idx < self.n_points:
                result += N[i] * self.control_points[idx]
        
        return result
    
    def get_basis_coefficients(self, t: float, derivative: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get basis function coefficients and corresponding control point indices.
        
        Useful for building Jacobians.
        
        Args:
            t: Time to evaluate
            derivative: Order of derivative
            
        Returns:
            (coefficients, indices) where result = sum(coeffs[i] * control_points[indices[i]])
        """
        k = self.find_knot_span(t)
        N = self.basis_functions(t, k, derivative)
        
        indices = []
        coeffs = []
        
        for i in range(self.degree + 1):
            idx = k - self.degree + i
            if 0 <= idx < self.n_points and abs(N[i]) > 1e-12:
                indices.append(idx)
                coeffs.append(N[i])
        
        return np.array(coeffs), np.array(indices, dtype=int)


def create_uniform_bspline_from_times(times: np.ndarray, degree: int, 
                                       boundary_order: int = 2) -> Tuple[UniformBSpline, int]:
    """
    Create a uniform B-spline that covers the given time range.
    
    Args:
        times: Array of measurement times
        degree: B-spline degree
        boundary_order: Number of extra control points at boundaries for clamping
        
    Returns:
        (bspline, n_control_points) with uninitialized control points
    """
    t_min, t_max = times.min(), times.max()
    duration = t_max - t_min
    
    # Choose dt such that we have reasonable resolution
    # Rule of thumb: one control point every few measurements
    n_measurements = len(times)
    target_points = max(10, min(100, n_measurements // 5))
    
    dt = duration / target_points
    
    # Number of control points needed
    n_interior = int(np.ceil(duration / dt)) + 1
    n_total = n_interior + 2 * boundary_order
    
    # Initialize with zero control points (will be optimized)
    control_points = np.zeros((n_total, 3))
    
    bspline = UniformBSpline(control_points, degree, dt)
    
    return bspline, n_total


def build_minimum_snap_regularization(bspline: UniformBSpline, 
                                       n_samples: int = 100) -> sparse.csr_matrix:
    """
    Build minimum snap regularization matrix R such that:
    E_reg = ||R * c||^2 approximates integral of ||p^(4)(t)||^2 dt
    
    Args:
        bspline: B-spline object
        n_samples: Number of time samples for numerical integration
        
    Returns:
        Sparse matrix R of shape (n_samples * 3, n_control_points * 3)
    """
    times = np.linspace(bspline.t_start, bspline.t_end, n_samples)
    
    rows = []
    cols = []
    vals = []
    
    for sample_idx, t in enumerate(times):
        coeffs, indices = bspline.get_basis_coefficients(t, derivative=4)
        
        if len(coeffs) == 0:
            continue
        
        # Weight by dt for numerical integration (trapezoidal rule)
        dt = (bspline.t_end - bspline.t_start) / (n_samples - 1)
        weight = np.sqrt(dt)
        
        # Add entries for each dimension (x, y, z)
        for dim in range(3):
            row_idx = sample_idx * 3 + dim
            for coeff, cp_idx in zip(coeffs, indices):
                col_idx = cp_idx * 3 + dim
                rows.append(row_idx)
                cols.append(col_idx)
                vals.append(weight * coeff)
    
    n_rows = n_samples * 3
    n_cols = bspline.n_points * 3
    
    R = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
    
    return R
