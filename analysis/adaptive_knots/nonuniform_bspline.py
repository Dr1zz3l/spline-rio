"""Non-uniform cumulative SO(3) B-spline — Python reference implementation.

This is the Phase-0 oracle for the adaptive-knot work (see documentation/ROADMAP.md
and the adaptive-knots plan). It will later serve as the golden reference for the
C++ NonUniformSO3Basis.

Conventions (match basalt / rio_solver_cpp):
- Order N=4 (cubic). One knot per control point: control point j lives at time
  knot_times[j]; basis B_j is supported on [tau_j, tau_{j+4}).
- Evaluation at t in [tau_k, tau_{k+1}) uses control points k-3 .. k
  (basalt: ori_index() -> i0 = k - (N_ORI-1)).
- The knot vector is extended by 3 "ghost" knots on each side, extrapolating the
  first/last real interval. With uniform spacing this reproduces basalt's uniform
  basis exactly everywhere, including the boundary spans.
- Cumulative basis over the active window, lam[0] = 1:
      lam[i] = sum_{l=i..3} B_{k-3+l}(t),   i = 0..3
  matching basalt's  coeff = cumulative_blending_matrix_ * [1,u,u^2,u^3]^T.
- Rotation (cumulative product, left-to-right over the active window):
      R(t) = R_{k-3} * prod_{j=1..3} Exp(lam[j] * Log(R_{k-4+j}^{-1} R_{k-3+j}))
- Body angular velocity (forward accumulation, mirrors basalt evaluate_lie):
      omega <- Exp(lam[j] delta_j)^T omega + dlam[j] * delta_j
"""

import numpy as np

N_ORDER = 4  # spline order (cubic)
DEG = N_ORDER - 1


# ---------------------------------------------------------------------------
# SO(3) helpers (self-contained; no dependency on analysis/lib)
# ---------------------------------------------------------------------------

def so3_hat(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def so3_exp(v):
    th = np.linalg.norm(v)
    K = so3_hat(v)
    if th < 1e-10:
        return np.eye(3) + K + 0.5 * (K @ K)
    return (np.eye(3) + np.sin(th) / th * K
            + (1.0 - np.cos(th)) / th**2 * (K @ K))


def so3_log(R):
    cos_th = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = np.arccos(cos_th)
    if th < 1e-10:
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return 0.5 * w
    if th > np.pi - 1e-6:
        # near-pi fallback via the symmetric part
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        # fix signs from off-diagonals
        if axis[0] > 0:
            axis[1] = np.copysign(axis[1], A[0, 1])
            axis[2] = np.copysign(axis[2], A[0, 2])
        elif axis[1] > 0:
            axis[2] = np.copysign(axis[2], A[1, 2])
        return th * axis / max(np.linalg.norm(axis), 1e-12)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2.0 * np.sin(th)) * w


# ---------------------------------------------------------------------------
# Knot vector handling
# ---------------------------------------------------------------------------

def extend_knots(knot_times):
    """Add DEG ghost knots on each side, extrapolating the boundary interval.

    Returns ext (len K + 2*DEG) with ext[j + DEG] = knot_times[j].
    """
    kt = np.asarray(knot_times, dtype=float)
    if kt.ndim != 1 or len(kt) < 2:
        raise ValueError("need >= 2 knot times")
    if np.any(np.diff(kt) <= 0):
        raise ValueError("knot times must be strictly increasing")
    dt0 = kt[1] - kt[0]
    dt1 = kt[-1] - kt[-2]
    head = kt[0] + dt0 * np.arange(-DEG, 0)
    tail = kt[-1] + dt1 * np.arange(1, DEG + 1)
    return np.concatenate([head, kt, tail])


def find_span(knot_times, t):
    """Span k such that knot_times[k] <= t < knot_times[k+1].

    Clamped to [DEG, K-1] like basalt's ori_index (valid eval domain is
    [knot_times[DEG], knot_times[K-1]]).
    """
    kt = np.asarray(knot_times)
    k = int(np.searchsorted(kt, t, side='right')) - 1
    return max(DEG, min(k, len(kt) - 1))


# ---------------------------------------------------------------------------
# Non-uniform basis (Cox-de Boor) — values and first derivative
# ---------------------------------------------------------------------------

def _basis_row(ext, j_ext, p, t):
    """B_{j,p}(t) by direct Cox-de Boor recursion on extended knots (0-based
    extended indexing: basis j_ext uses knots ext[j_ext .. j_ext+p+1])."""
    if p == 0:
        return 1.0 if (ext[j_ext] <= t < ext[j_ext + 1]) else 0.0
    out = 0.0
    d1 = ext[j_ext + p] - ext[j_ext]
    if d1 > 0:
        out += (t - ext[j_ext]) / d1 * _basis_row(ext, j_ext, p - 1, t)
    d2 = ext[j_ext + p + 1] - ext[j_ext + 1]
    if d2 > 0:
        out += (ext[j_ext + p + 1] - t) / d2 * _basis_row(ext, j_ext + 1, p - 1, t)
    return out


def basis_active(knot_times, ext, k, t):
    """Values and 1st derivatives of the 4 active cubic basis functions
    B_{k-3..k} at t (t in span [tau_k, tau_{k+1})).

    Returns (B, dB), each shape (4,), index l -> control point k-3+l.
    """
    B = np.zeros(4)
    dB = np.zeros(4)
    # clamp t into the half-open span for the degree-0 indicator logic
    t_eval = t
    if t >= knot_times[min(k + 1, len(knot_times) - 1)] or t >= knot_times[-1]:
        t_eval = np.nextafter(knot_times[min(k + 1, len(knot_times) - 1)], -np.inf)
    for l in range(4):
        j = k - 3 + l            # control point index (real indexing)
        j_ext = j + DEG          # extended indexing
        B[l] = _basis_row(ext, j_ext, 3, t_eval)
        # derivative: B'_{j,3} = 3 [ B_{j,2}/(tau_{j+3}-tau_j) - B_{j+1,2}/(tau_{j+4}-tau_{j+1}) ]
        d1 = ext[j_ext + 3] - ext[j_ext]
        d2 = ext[j_ext + 4] - ext[j_ext + 1]
        term = 0.0
        if d1 > 0:
            term += _basis_row(ext, j_ext, 2, t_eval) / d1
        if d2 > 0:
            term -= _basis_row(ext, j_ext + 1, 2, t_eval) / d2
        dB[l] = 3.0 * term
    return B, dB


def cumulative_coeffs(knot_times, ext, k, t):
    """Cumulative basis lam, dlam (each (4,)) over the active window at t.

    lam[i] = sum_{l>=i} B[l];  lam[0] == 1 (partition of unity).
    """
    B, dB = basis_active(knot_times, ext, k, t)
    lam = np.cumsum(B[::-1])[::-1].copy()
    dlam = np.cumsum(dB[::-1])[::-1].copy()
    lam[0] = 1.0   # exact by partition of unity
    dlam[0] = 0.0
    return lam, dlam


# ---------------------------------------------------------------------------
# Non-uniform cumulative SO(3) spline
# ---------------------------------------------------------------------------

class NonUniformSO3Spline:
    """Cumulative SO(3) B-spline with non-uniform knots.

    knot_times: (K,) strictly increasing; R_knots: (K,3,3) rotation matrices
    (absolute orientations, like rio's quaternion knots).
    """

    def __init__(self, knot_times, R_knots):
        self.kt = np.asarray(knot_times, dtype=float)
        self.R = np.asarray(R_knots, dtype=float)
        if len(self.kt) != len(self.R):
            raise ValueError("knot_times and R_knots length mismatch")
        if len(self.kt) < N_ORDER:
            raise ValueError("need >= 4 knots")
        self.ext = extend_knots(self.kt)
        # cache inter-knot increments delta_j = Log(R_{j-1}^T R_j), j=1..K-1
        self.delta = np.zeros((len(self.kt), 3))
        for j in range(1, len(self.kt)):
            self.delta[j] = so3_log(self.R[j - 1].T @ self.R[j])

    @property
    def t_start(self):
        return self.kt[DEG]

    @property
    def t_end(self):
        return self.kt[-1]

    def evaluate(self, t):
        """Returns (R(t), omega_body(t))."""
        k = find_span(self.kt, t)
        lam, dlam = cumulative_coeffs(self.kt, self.ext, k, t)
        i0 = k - 3
        R = self.R[i0].copy()
        omega = np.zeros(3)
        for j in range(1, 4):
            d = self.delta[i0 + j]
            e = so3_exp(lam[j] * d)
            R = R @ e
            omega = e.T @ omega + dlam[j] * d
        return R, omega
