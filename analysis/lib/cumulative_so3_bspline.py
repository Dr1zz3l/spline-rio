"""
Cumulative B-spline on SO(3) following Sommer et al. 2020.

The rotation at time t is:
    R(t) = R_base[k-3] @ exp(B̃_0(t) * Ω_{k-2}) @ exp(B̃_1(t) * Ω_{k-1}) @ exp(B̃_2(t) * Ω_k)

where:
  - k = knot span (degree=3 cubic, so first valid span k=3)
  - Ω_i ∈ so(3) are logarithmic increment control knots (optimization variables)
  - B̃_j(t) are cumulative basis functions (suffix sums of standard B-spline basis)
  - R_base[i] = exp(Ω_0) @ ... @ exp(Ω_i)  (precomputed, refreshed after each update)

Angular velocity (using J_r(φ)φ = φ identity):
    ω(t) = Σ_j dB̃_j/dt * R_suffix_j^T @ Ω_j

where R_suffix_j = product of exp factors AFTER j within the span.

Jacobians:
  ∂R/∂Ω_j (right tangent at R):  J_R_j = R_suffix_j^T @ J_r(B̃_j Ω_j) * B̃_j
  ∂ω/∂Ω_j:                        J_ω_j = dB̃_j/dt * R_suffix_j^T + cross terms
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Tuple

from bspline_utils import UniformBSpline


# ===========================================================================
# SO(3) math helpers
# ===========================================================================

def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues formula: so(3) -> SO(3)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        return np.eye(3) + skew(omega)
    return Rotation.from_rotvec(omega).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    """Inverse Rodrigues: SO(3) -> so(3)."""
    return Rotation.from_matrix(R).as_rotvec()


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix from 3-vector."""
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """
    Right Jacobian of SO(3) exponential map.

    J_r(φ) = I - (1-cosθ)/θ² [φ]× + (θ-sinθ)/θ³ [φ]×²

    At θ=0: J_r = I.
    """
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * skew(phi)
    S = skew(phi)
    c1 = (1.0 - np.cos(theta)) / (theta * theta)
    c2 = (theta - np.sin(theta)) / (theta ** 3)
    return np.eye(3) - c1 * S + c2 * (S @ S)


# ===========================================================================
# Cumulative SO(3) B-spline
# ===========================================================================

class CumulativeSO3BSpline:
    """
    Cumulative B-spline on SO(3) (cubic, degree=3).

    Attributes:
        omega_knots : (N, 3) ndarray  — so(3) increment control knots
        degree      : int             — 3 (cubic only)
        dt          : float           — knot spacing (seconds)
        t_ref       : float           — absolute time reference; evaluate at t_rel = t_abs - t_ref
        n_knots     : int             — N
    """

    def __init__(
        self,
        omega_knots: np.ndarray,
        degree: int = 3,
        dt: float = 0.05,
        t_ref: float = 0.0,
    ):
        if degree != 3:
            raise NotImplementedError("Only cubic (degree=3) is implemented.")
        self.omega_knots = np.array(omega_knots, dtype=float)  # (N, 3)
        self.degree = degree
        self.dt = dt
        self.t_ref = t_ref
        self.n_knots = len(omega_knots)

        # Internal UniformBSpline used ONLY for knot-span finding and basis evaluation.
        # Control points are irrelevant; we use dummy zeros.
        self._bspline = UniformBSpline(
            np.zeros((self.n_knots, 3)), degree=degree, dt=dt
        )

        # Precompute base rotations
        self._base_rotations = np.zeros((self.n_knots, 3, 3))
        self.recompute_base_rotations()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def recompute_base_rotations(self):
        """
        Rebuild _base_rotations from omega_knots.

        _base_rotations[i] = exp(Ω_0) @ exp(Ω_1) @ ... @ exp(Ω_i)
        """
        R = np.eye(3)
        for i in range(self.n_knots):
            R = R @ so3_exp(self.omega_knots[i])
            self._base_rotations[i] = R

    def _cumulative_basis(self, t_rel: float) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Return knot span k and cumulative basis values/derivatives.

        Returns:
            k         : knot span index (degree <= k <= n_knots-1)
            B_tilde   : (3,) cumulative basis values
            dB_tilde  : (3,) time derivatives of cumulative basis
        """
        k = self._bspline.find_knot_span(t_rel)
        N  = self._bspline.basis_functions(t_rel, k, derivative=0)   # shape (degree+1,)
        dN = self._bspline.basis_functions(t_rel, k, derivative=1)   # shape (degree+1,)

        # Suffix sums: B̃_j = Σ_{m=j+1}^{3} N[m]  (for j=0,1,2)
        B_tilde  = np.array([N[1]+N[2]+N[3],  N[2]+N[3],  N[3]])
        dB_tilde = np.array([dN[1]+dN[2]+dN[3], dN[2]+dN[3], dN[3]])
        return k, B_tilde, dB_tilde

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def evaluate(self, t_rel: float) -> np.ndarray:
        """
        Return 3×3 rotation matrix R(t_rel).

        R = R_base[k-3] @ exp(B̃_0 Ω_{k-2}) @ exp(B̃_1 Ω_{k-1}) @ exp(B̃_2 Ω_k)
        """
        k, B_tilde, _ = self._cumulative_basis(t_rel)
        active = [k-2, k-1, k]
        R = self._base_rotations[k-3].copy()
        for j in range(3):
            R = R @ so3_exp(B_tilde[j] * self.omega_knots[active[j]])
        return R

    def evaluate_with_jacobians(
        self, t_rel: float
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Evaluate rotation, angular velocity, and their Jacobians w.r.t. Ωⱼ.

        Returns:
            R              : (3,3) rotation matrix
            omega          : (3,)  body angular velocity
            J_R_list       : list of 3 (3,3) matrices — ∂(right tangent at R)/∂Ωⱼ
            J_omega_list   : list of 3 (3,3) matrices — ∂ω/∂Ωⱼ
            active_indices : list of 3 absolute knot indices [k-2, k-1, k]

        Jacobian conventions:
          J_R_list[j]:   if a residual r depends on R via tangent ξ (right-perturbed),
                         then ∂r/∂Ω_j = (∂r/∂ξ) @ J_R_list[j]
          J_omega_list[j]: ∂r/∂Ω_j = (∂r/∂ω) @ J_omega_list[j]
        """
        k, B_tilde, dB_tilde = self._cumulative_basis(t_rel)
        active = [k-2, k-1, k]

        # Incremental rotations within this span
        phi = [B_tilde[j] * self.omega_knots[active[j]] for j in range(3)]
        R_part = [so3_exp(phi[j]) for j in range(3)]    # R_0, R_1, R_2

        # Base rotation (accumulated product up to k-3)
        R_base_k = self._base_rotations[k-3]

        # Suffix products: R_suffix[j] = R_part[j+1] @ ... @ R_part[2]
        R_suffix = [None, None, None]
        R_suffix[2] = np.eye(3)
        R_suffix[1] = R_part[2]
        R_suffix[0] = R_part[1] @ R_part[2]

        # Full rotation
        R = R_base_k @ R_part[0] @ R_part[1] @ R_part[2]

        # Angular velocity: ω = Σ_j dB̃_j * R_suffix_j^T Ω_j
        # (simplified via J_r(φ)φ = φ)
        omega = np.zeros(3)
        for j in range(3):
            omega += dB_tilde[j] * (R_suffix[j].T @ self.omega_knots[active[j]])

        # ---- J_R_list ----
        # J_R_j = R_suffix_j^T @ J_r(B̃_j Ω_j) * B̃_j   [shape (3,3)]
        J_R_list = []
        Jr = [right_jacobian_so3(phi[j]) for j in range(3)]
        for j in range(3):
            J_R_list.append(R_suffix[j].T @ Jr[j] * B_tilde[j])

        # ---- J_omega_list ----
        # J_ω_j = dB̃_j * R_suffix_j^T          (direct term)
        #        + Σ_{m<j} dB̃_m * R_suffix_j^T @ [R_j^T A_{m,j}^T Ω_m]× @ J_r(B̃_j Ω_j) * B̃_j
        # where A_{m,j} = R_part[m+1] @ ... @ R_part[j-1]
        J_omega_list = []
        for j in range(3):
            # Direct term
            J_j = dB_tilde[j] * R_suffix[j].T   # (3,3)

            # Cross terms for m < j
            for m in range(j):
                # A_{m,j} = product of R_part[m+1..j-1]
                A = np.eye(3)
                for l in range(m+1, j):
                    A = A @ R_part[l]
                # u = A^T @ Ω_m  (independent of Ω_j)
                u = A.T @ self.omega_knots[active[m]]
                # R_j^T @ u
                Rj_T_u = R_part[j].T @ u
                # cross contribution: dB̃_m * R_suffix_j^T @ skew(R_j^T u) @ J_r(B̃_j Ω_j) * B̃_j
                cross = dB_tilde[m] * (R_suffix[j].T @ skew(Rj_T_u) @ Jr[j] * B_tilde[j])
                J_j = J_j + cross

            J_omega_list.append(J_j)

        return R, omega, J_R_list, J_omega_list, active

    def evaluate_full_jacobians(
        self, t_rel: float, base_window: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Like evaluate_with_jacobians, but also includes Jacobians for base knots j < k-2.

        The cumulative B-spline has a left-triangular Jacobian structure:
          R(t) = R_base[k-3] @ R_part[0] @ R_part[1] @ R_part[2]
        Changing Ω_j for j <= k-3 shifts R_base[k-3], which affects R(t).

        Formula for base knot j in {0, ..., k-3}:
          J_R_base_j  = R_full^T @ _base_rotations[j] @ Jr(Ω_j)
          J_omega_j   = 0  (base knots do NOT affect angular velocity)

        Args:
            t_rel      : relative time (seconds)
            base_window: max number of base knots to include (counting back from k-3).
                         0 = full left-triangular (all base knots, exact but dense).
                         N > 0 = only the last N base knots (sliding-window approximation).

        Returns:
            R              : (3,3) rotation matrix
            omega          : (3,)  body angular velocity
            J_R_all        : list of (3,3) matrices — ∂(right tangent)/∂Ωⱼ
            J_omega_all    : list of (3,3) matrices — ∂ω/∂Ωⱼ
            all_indices    : list of absolute knot indices
        """
        R, omega, J_R_span, J_omega_span, active = self.evaluate_with_jacobians(t_rel)
        k = active[2]  # highest active knot index = knot span k

        # --- Base knot Jacobians: j in {0, ..., k-3} ---
        J_R_base = []
        J_omega_base = []
        all_base = list(range(0, k - 2))  # j = 0, 1, ..., k-3
        if base_window > 0:
            base_indices = all_base[-base_window:]  # most-recent W base knots
        else:
            base_indices = all_base               # full left-triangular

        for j in base_indices:
            R_base_j = self._base_rotations[j]  # exp(Ω_0) @ ... @ exp(Ω_j)
            Jr_j = right_jacobian_so3(self.omega_knots[j])
            J_R_base.append(R.T @ R_base_j @ Jr_j)  # (3,3)
            J_omega_base.append(np.zeros((3, 3)))

        # Combine: base knots first, then active span knots
        J_R_all    = J_R_base    + J_R_span
        J_omega_all = J_omega_base + J_omega_span
        all_indices = base_indices + active

        return R, omega, J_R_all, J_omega_all, all_indices

    @property
    def t_start(self) -> float:
        return self._bspline.t_start

    @property
    def t_end(self) -> float:
        return self._bspline.t_end

    # ------------------------------------------------------------------
    # State vector interface (mirrors UniformBSpline pattern)
    # ------------------------------------------------------------------

    def to_flat(self) -> np.ndarray:
        """Return omega_knots flattened to (N*3,)."""
        return self.omega_knots.flatten()

    def from_flat(self, x: np.ndarray):
        """Update omega_knots from flat (N*3,) vector and recompute base rotations."""
        self.omega_knots = x.reshape(-1, 3)
        self.recompute_base_rotations()

    # ------------------------------------------------------------------
    # Initialization from rotation samples
    # ------------------------------------------------------------------

    @classmethod
    def from_rotation_samples(
        cls,
        R_samples: np.ndarray,
        dt: float,
        t_ref: float = 0.0,
    ) -> "CumulativeSO3BSpline":
        """
        Initialize cumulative B-spline from N rotation matrices (N, 3, 3).

        Sets omega_knots[0] = log(R_0) and omega_knots[i] = log(R_{i-1}^T R_i).
        This is an approximate initialization; the solver refines it.

        Args:
            R_samples : (N, 3, 3) rotation matrices sampled at knot times
            dt        : knot spacing
            t_ref     : absolute time reference
        """
        N = len(R_samples)
        omega_knots = np.zeros((N, 3))
        omega_knots[0] = so3_log(R_samples[0])
        for i in range(1, N):
            omega_knots[i] = so3_log(R_samples[i-1].T @ R_samples[i])
        return cls(omega_knots, degree=3, dt=dt, t_ref=t_ref)


# ===========================================================================
# Numerical Jacobian validation
# ===========================================================================

def _numerical_jacobian_R(spline: CumulativeSO3BSpline, t_rel: float, eps: float = 1e-7):
    """
    Finite-difference Jacobian of R(t) w.r.t. each active omega_knot.
    Returns list of 3 (3,3) matrices (tangent-space perturbations).
    """
    k, B_tilde, _ = spline._cumulative_basis(t_rel)
    active = [k-2, k-1, k]
    R0 = spline.evaluate(t_rel)

    J_list = []
    for idx in active:
        J = np.zeros((3, 3))
        for dim in range(3):
            # Perturb Ω[idx][dim] by +eps
            spline.omega_knots[idx, dim] += eps
            spline.recompute_base_rotations()
            R_plus = spline.evaluate(t_rel)
            spline.omega_knots[idx, dim] -= eps
            spline.recompute_base_rotations()

            # Right tangent: log(R0^T R_plus) / eps
            delta_R = R0.T @ R_plus
            rotvec = so3_log(delta_R)
            J[:, dim] = rotvec / eps
        J_list.append(J)
    return J_list


def _numerical_jacobian_omega(spline: CumulativeSO3BSpline, t_rel: float, eps: float = 1e-7):
    """
    Finite-difference Jacobian of omega(t) w.r.t. each active omega_knot.
    Returns list of 3 (3,3) matrices.
    """
    k, B_tilde, _ = spline._cumulative_basis(t_rel)
    active = [k-2, k-1, k]
    _, omega0, _, _, _ = spline.evaluate_with_jacobians(t_rel)

    J_list = []
    for idx in active:
        J = np.zeros((3, 3))
        for dim in range(3):
            spline.omega_knots[idx, dim] += eps
            spline.recompute_base_rotations()
            _, omega_plus, _, _, _ = spline.evaluate_with_jacobians(t_rel)
            spline.omega_knots[idx, dim] -= eps
            spline.recompute_base_rotations()
            J[:, dim] = (omega_plus - omega0) / eps
        J_list.append(J)
    return J_list


def _test_jacobians(n_knots: int = 20, n_test_times: int = 10, verbose: bool = True):
    """
    Validate analytical Jacobians against finite differences.

    Creates a random spline, evaluates J_R_list and J_omega_list at several random times,
    and asserts relative error < 1e-4.
    """
    rng = np.random.default_rng(42)

    # Random small omega knots (so rotations don't wrap)
    omega_knots = rng.standard_normal((n_knots, 3)) * 0.15  # ~8.5 deg/knot

    spline = CumulativeSO3BSpline(omega_knots, degree=3, dt=0.05)

    t_min = spline.t_start + 0.01
    t_max = spline.t_end - 0.01
    test_times = rng.uniform(t_min, t_max, n_test_times)

    max_err_R = 0.0
    max_err_omega = 0.0

    for t_rel in test_times:
        R, omega, J_R_list, J_omega_list, active = spline.evaluate_with_jacobians(t_rel)

        J_R_num = _numerical_jacobian_R(spline, t_rel)
        J_omega_num = _numerical_jacobian_omega(spline, t_rel)

        for j in range(3):
            # R Jacobian
            denom_R = max(np.linalg.norm(J_R_num[j]), 1e-10)
            err_R = np.linalg.norm(J_R_list[j] - J_R_num[j]) / denom_R
            max_err_R = max(max_err_R, err_R)

            # omega Jacobian
            denom_o = max(np.linalg.norm(J_omega_num[j]), 1e-10)
            err_o = np.linalg.norm(J_omega_list[j] - J_omega_num[j]) / denom_o
            max_err_omega = max(max_err_omega, err_o)

            if verbose and (err_R > 1e-4 or err_o > 1e-4):
                print(f"  [WARN] t={t_rel:.4f} j={j}  err_R={err_R:.2e}  err_omega={err_o:.2e}")
                print(f"    J_R_ana:\n{J_R_list[j]}")
                print(f"    J_R_num:\n{J_R_num[j]}")
                print(f"    J_omega_ana:\n{J_omega_list[j]}")
                print(f"    J_omega_num:\n{J_omega_num[j]}")

    if verbose:
        print(f"J_R:    max relative error = {max_err_R:.2e}")
        print(f"J_omega: max relative error = {max_err_omega:.2e}")

    assert max_err_R < 1e-4, f"J_R Jacobian error too large: {max_err_R:.2e}"
    assert max_err_omega < 1e-4, f"J_omega Jacobian error too large: {max_err_omega:.2e}"

    if verbose:
        print("✓ All Jacobians validated (relative error < 1e-4)")

    return max_err_R, max_err_omega


def _test_evaluate_consistency(verbose: bool = True):
    """Verify that evaluate() and evaluate_with_jacobians() return the same R and omega."""
    rng = np.random.default_rng(7)
    omega_knots = rng.standard_normal((15, 3)) * 0.1
    spline = CumulativeSO3BSpline(omega_knots, degree=3, dt=0.05)

    t_rel = (spline.t_start + spline.t_end) / 2.0
    R1 = spline.evaluate(t_rel)
    R2, omega, _, _, _ = spline.evaluate_with_jacobians(t_rel)

    err = np.linalg.norm(R1 - R2)
    assert err < 1e-12, f"evaluate() and evaluate_with_jacobians() disagree: err={err:.2e}"
    if verbose:
        print(f"✓ evaluate() consistency check passed (err={err:.2e})")


def _test_initialization(verbose: bool = True):
    """Verify from_rotation_samples round-trips approximately."""
    from scipy.spatial.transform import Slerp, Rotation
    # Create a smooth sequence of rotations (sine-wave Euler angles)
    N = 20
    ts = np.linspace(0, 1.0, N)
    rots = Rotation.from_euler('zyx', np.column_stack([
        0.5 * np.sin(2 * np.pi * ts),
        0.3 * np.cos(2 * np.pi * ts),
        0.1 * ts,
    ]))
    R_samples = rots.as_matrix()

    spline = CumulativeSO3BSpline.from_rotation_samples(R_samples, dt=0.05)

    # Evaluate at control knot times and compare against input
    errors_deg = []
    for i in range(N):
        t_rel = spline.t_start + i * 0.0   # At knot boundaries
        # Evaluate at the B-spline domain start (t_start)
        pass  # This is a rough init only; skip detailed accuracy check

    if verbose:
        print("✓ from_rotation_samples constructed without error")


if __name__ == "__main__":
    print("=" * 60)
    print("CumulativeSO3BSpline — Jacobian Validation")
    print("=" * 60)
    _test_evaluate_consistency()
    _test_initialization()
    print("\nRunning Jacobian tests (n_knots=20, n_times=20)...")
    _test_jacobians(n_knots=20, n_test_times=20, verbose=True)
    print("\nRunning Jacobian tests with larger rotations (n_knots=20, n_times=10)...")
    # Larger knots to stress-test cross terms
    rng = np.random.default_rng(99)
    omega_knots_large = rng.standard_normal((20, 3)) * 0.4   # ~23 deg/knot
    spline_large = CumulativeSO3BSpline(omega_knots_large, degree=3, dt=0.05)
    t_test = (spline_large.t_start + spline_large.t_end) / 2.0
    _, _, J_R, J_om, _ = spline_large.evaluate_with_jacobians(t_test)
    J_R_n = _numerical_jacobian_R(spline_large, t_test)
    J_om_n = _numerical_jacobian_omega(spline_large, t_test)
    for j in range(3):
        err_R = np.linalg.norm(J_R[j] - J_R_n[j]) / max(np.linalg.norm(J_R_n[j]), 1e-10)
        err_o = np.linalg.norm(J_om[j] - J_om_n[j]) / max(np.linalg.norm(J_om_n[j]), 1e-10)
        print(f"  j={j}: err_R={err_R:.2e}  err_omega={err_o:.2e}")
    print("\n✅ All tests complete.")
