"""
gtsam factor-graph scaffold for the continuous-time B-spline RIO (Phase 0a).

Variables (ABSOLUTE-knot, basalt convention -> strictly local support):
  - orientation knot j  -> gtsam.Rot3            key  R(j)
  - position CP i        -> gtsam Vector3 (Point3) key P(i)
  - IMU bias (6: ba,bg)  -> gtsam Vector6          key B(k)   (k=0 for constant-bias)

Spline evaluation reuses the basalt-exact primitives:
  - orientation: closed-form uniform cubic cumulative coeffs, mirrors
    NonUniformSO3Spline.evaluate() (adaptive_knots/nonuniform_bspline.py)
  - position: UniformBSpline.basis_functions (lib/bspline_utils.py)
Residual MODELS reuse codegen/generated_jacobians.py (same SymForce derivation as
the C++ analytic factors), so radar/accel/gyro conventions match the deployed solver.

Factor Jacobians are computed by finite difference using the SAME retraction gtsam
uses (Rot3: R*Exp(xi) right-perturbation; vectors: +), so they are correct w.r.t.
the gtsam variables by construction.  (Phase 1 will replace FD with the analytic
chain for the C++ port; the spike only needs correct Jacobians, not fast ones.)
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))
for p in (ANALYSIS, os.path.join(ANALYSIS, 'lib'),
          os.path.join(ANALYSIS, 'adaptive_knots'),
          os.path.join(ANALYSIS, 'codegen')):
    if p not in sys.path:
        sys.path.insert(0, p)

import gtsam                                          # noqa: E402
from gtsam import CustomFactor, Rot3                  # noqa: E402
from gtsam.symbol_shorthand import P, R, B           # noqa: E402  P=pos CP, R=ori knot, B=bias
import generated_jacobians as gj                      # noqa: E402
from bspline_utils import UniformBSpline              # noqa: E402
from radar_velocity_utils import rotation_matrix_from_euler  # noqa: E402


# Fast pure-numpy SO(3) exp/log (Rodrigues) — the spike evaluates these a LOT
# under finite differencing; avoids scipy object overhead (~10x faster).
def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(w):
    th2 = float(w @ w)
    if th2 < 1e-24:
        return np.eye(3) + _skew(w)
    th = np.sqrt(th2)
    K = _skew(w)
    return np.eye(3) + (np.sin(th) / th) * K + ((1.0 - np.cos(th)) / th2) * (K @ K)


def so3_log(R):
    c = (np.trace(R) - 1.0) * 0.5
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-7:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(th)
    return (th / (2.0 * s)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


EPS = 1e-9
G_WORLD = np.array([0.0, 0.0, -9.81])
_FD = 1e-6            # finite-difference step (tangent units)
N_ORI = 4
N_POS = 6


# ----------------------------------------------------------------------------
# Spline primitives
# ----------------------------------------------------------------------------
def cumulative_cubic(u):
    """Uniform cubic cumulative basis lam(u) and d lam/du (each length 4).
    lam[0]=1 (partition of unity).  Matches basalt cumulative_blending_matrix_."""
    u2, u3 = u * u, u * u * u
    lam = np.array([1.0,
                    (5.0 + 3.0 * u - 3.0 * u2 + u3) / 6.0,
                    (1.0 + 3.0 * u + 3.0 * u2 - 2.0 * u3) / 6.0,
                    u3 / 6.0])
    dlam_du = np.array([0.0,
                        (3.0 - 6.0 * u + 3.0 * u2) / 6.0,
                        (3.0 + 6.0 * u - 6.0 * u2) / 6.0,
                        (3.0 * u2) / 6.0])
    return lam, dlam_du


def eval_ori_local(R4, u, inv_dt):
    """R(t), omega_body(t) from 4 active absolute rotation matrices.
    Mirrors NonUniformSO3Spline.evaluate (basalt evaluate_lie, uniform grid)."""
    lam, dlam_du = cumulative_cubic(u)
    dlam = dlam_du * inv_dt
    Rcur = R4[0].copy()
    omega = np.zeros(3)
    for j in range(1, 4):
        d = so3_log(R4[j - 1].T @ R4[j])
        e = so3_exp(lam[j] * d)
        Rcur = Rcur @ e
        omega = e.T @ omega + dlam[j] * d
    return Rcur, omega


class Problem:
    """Holds the captured batch problem and the spline index/basis machinery."""

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.d = d
        self.t_ref = float(d['t_ref'])
        self.dt_ori = float(d['ori_dt'])
        self.dt_pos = float(d['pos_dt'])
        self.pos_degree = int(d['pos_degree'])
        self.init_ori_quats = d['init_ori_quats']        # (Nori,4) xyzw
        self.init_pos_cps = d['init_pos_cps']            # (Npos,3)
        self.init_biases = d['init_biases']              # (6,)
        self.n_ori = len(self.init_ori_quats)
        self.n_pos = len(self.init_pos_cps)
        self.cfg = d['solver_cfg'].item()
        # extrinsics: use the C++-SOLVED pitch (lock extrinsics in the spike)
        self.ext_euler_deg = np.asarray(d['cpp_ext_euler_deg'], float)
        self.t_bs = np.asarray(d['ext_trans_m'], float)
        self.R_bs = rotation_matrix_from_euler(*np.deg2rad(self.ext_euler_deg))
        self.R_bs_rot = gj.Rot3.from_rotation_matrix(self.R_bs)   # shim quat for generated code
        # UniformBSpline for position basis only (knot vector + basis_functions)
        self._pos = UniformBSpline(self.init_pos_cps, self.pos_degree, self.dt_pos)
        self.inv_dt_ori = 1.0 / self.dt_ori

    # --- index/basis helpers (match C++ trajectory.h pos_index / ori_index) ---
    def ori_active(self, t_abs):
        t_rel = t_abs - self.t_ref
        k = int(t_rel / self.dt_ori)
        k = max(N_ORI - 1, min(k, self.n_ori - 1))
        u = (t_rel - k * self.dt_ori) / self.dt_ori
        u = min(max(u, 0.0), 1.0 - 1e-10)
        return k, u                       # active knots [k-3 .. k]

    def pos_active(self, t_abs):
        t_rel = t_abs - self.t_ref
        k = self._pos.find_knot_span(t_rel)
        N0 = self._pos.basis_functions(t_rel, k, 0)
        N1 = self._pos.basis_functions(t_rel, k, 1)
        N2 = self._pos.basis_functions(t_rel, k, 2)
        return k, N0, N1, N2              # active CPs [k-degree .. k]

    def in_domain(self, t_abs):
        t_rel = t_abs - self.t_ref
        ok_ori = (N_ORI - 1) * self.dt_ori <= t_rel <= (self.n_ori - 1) * self.dt_ori
        ok_pos = self.pos_degree * self.dt_pos <= t_rel <= (self.n_pos - 1) * self.dt_pos
        return ok_ori and ok_pos

    # --- initial gtsam Values ---
    def initial_values(self, ori_idx=None, pos_idx=None, single_bias=True):
        v = gtsam.Values()
        ori_idx = range(self.n_ori) if ori_idx is None else ori_idx
        pos_idx = range(self.n_pos) if pos_idx is None else pos_idx
        for j in ori_idx:
            q = self.init_ori_quats[j]    # xyzw
            v.insert(R(j), Rot3(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
        for i in pos_idx:
            v.insert(P(i), np.asarray(self.init_pos_cps[i], float))
        if single_bias:
            v.insert(B(0), np.asarray(self.init_biases, float))
        return v


# ----------------------------------------------------------------------------
# Noise models (weight lambda -> Sigma = 1/sqrt(lambda); precision = lambda)
# ----------------------------------------------------------------------------
def _iso(dim, lam):
    return gtsam.noiseModel.Isotropic.Sigma(dim, 1.0 / np.sqrt(max(lam, 1e-30)))


# ----------------------------------------------------------------------------
# Factor builders (CustomFactor with retract-consistent FD Jacobians)
# ----------------------------------------------------------------------------
def _fd_fill(H, residual_fn, R4, P6, bias, has_pos, has_bias):
    """Fill H (list) with FD Jacobians.  Block order: ori(4), pos(6?), bias(1?)."""
    r0 = residual_fn(R4, P6, bias)
    m = r0.shape[0]
    slot = 0
    for idx in range(len(R4)):
        J = np.zeros((m, 3))
        for a in range(3):
            dax = np.zeros(3); dax[a] = _FD
            R4p = list(R4); R4p[idx] = R4[idx] @ so3_exp(dax)      # gtsam right-retract
            J[:, a] = (residual_fn(R4p, P6, bias) - r0) / _FD
        if H is not None:
            H[slot] = J
        slot += 1
    if has_pos:
        for idx in range(len(P6)):
            J = np.zeros((m, 3))
            for a in range(3):
                dP = np.zeros(3); dP[a] = _FD
                P6p = list(P6); P6p[idx] = P6[idx] + dP
                J[:, a] = (residual_fn(R4, P6p, bias) - r0) / _FD
            if H is not None:
                H[slot] = J
            slot += 1
    if has_bias:
        J = np.zeros((m, 6))
        for a in range(6):
            db = np.zeros(6); db[a] = _FD
            J[:, a] = (residual_fn(R4, P6, bias + db) - r0) / _FD
        if H is not None:
            H[slot] = J
        slot += 1
    return r0


def make_radar_factor(prob, t, u_sensor, v_meas, noise):
    ko, uo = prob.ori_active(t)
    kp, N0, N1, N2 = prob.pos_active(t)
    ori_keys = [R(ko - 3 + l) for l in range(N_ORI)]
    pos_keys = [P(kp - prob.pos_degree + l) for l in range(N_POS)]
    keys = ori_keys + pos_keys
    R_bs, t_bs = prob.R_bs, prob.t_bs
    inv_dt = prob.inv_dt_ori

    def residual_fn(R4, P6, bias):
        Rm, omega = eval_ori_local(R4, uo, inv_dt)
        v_world = np.zeros(3)
        for l in range(N_POS):
            v_world = v_world + N1[l] * P6[l]
        out = gj.radar_residual_with_jacobians(
            v_world, gj.Rot3.from_rotation_matrix(Rm), np.zeros(3), omega,
            u_sensor, t_bs, prob.R_bs_rot, float(v_meas), EPS)
        return np.atleast_1d(np.asarray(out[0], float)).reshape(-1)

    def err(this, values, H):
        R4 = [values.atRot3(k).matrix() for k in ori_keys]
        P6 = [np.asarray(values.atVector(k), float) for k in pos_keys]
        return _fd_fill(H, residual_fn, R4, P6, None, has_pos=True, has_bias=False)

    return CustomFactor(noise, keys, err)


def make_accel_factor(prob, t, z_acc, noise, bias_key=None):
    bias_key = B(0) if bias_key is None else bias_key
    ko, uo = prob.ori_active(t)
    kp, N0, N1, N2 = prob.pos_active(t)
    ori_keys = [R(ko - 3 + l) for l in range(N_ORI)]
    pos_keys = [P(kp - prob.pos_degree + l) for l in range(N_POS)]
    keys = ori_keys + pos_keys + [bias_key]
    inv_dt = prob.inv_dt_ori
    z_acc = np.asarray(z_acc, float)

    def residual_fn(R4, P6, bias):
        Rm, _ = eval_ori_local(R4, uo, inv_dt)
        a_world = np.zeros(3)
        for l in range(N_POS):
            a_world = a_world + N2[l] * P6[l]
        out = gj.accel_residual_with_jacobians(
            a_world, gj.Rot3.from_rotation_matrix(Rm), np.zeros(3),
            G_WORLD, z_acc, bias[:3], EPS)
        return np.asarray(out[0], float).reshape(-1)

    def err(this, values, H):
        R4 = [values.atRot3(k).matrix() for k in ori_keys]
        P6 = [np.asarray(values.atVector(k), float) for k in pos_keys]
        bias = np.asarray(values.atVector(bias_key), float)
        return _fd_fill(H, residual_fn, R4, P6, bias, has_pos=True, has_bias=True)

    return CustomFactor(noise, keys, err)


def make_gyro_factor(prob, t, z_gyro, noise, bias_key=None):
    bias_key = B(0) if bias_key is None else bias_key
    ko, uo = prob.ori_active(t)
    ori_keys = [R(ko - 3 + l) for l in range(N_ORI)]
    keys = ori_keys + [bias_key]
    inv_dt = prob.inv_dt_ori
    z_gyro = np.asarray(z_gyro, float)

    def residual_fn(R4, P6, bias):
        _, omega = eval_ori_local(R4, uo, inv_dt)
        out = gj.gyro_residual_with_jacobians(
            omega, np.zeros(3), np.zeros(3), z_gyro, bias[3:], EPS)
        return np.asarray(out[0], float).reshape(-1)

    def err(this, values, H):
        R4 = [values.atRot3(k).matrix() for k in ori_keys]
        bias = np.asarray(values.atVector(bias_key), float)
        return _fd_fill(H, residual_fn, R4, None, bias, has_pos=False, has_bias=True)

    return CustomFactor(noise, keys, err)


def make_bias_between(key0, key1, noise):
    """Random-walk bias link: r = bias1 - bias0 (drives the per-stride bias)."""
    def err(this, values, H):
        b0 = np.asarray(values.atVector(key0), float)
        b1 = np.asarray(values.atVector(key1), float)
        if H is not None:
            H[0] = -np.eye(6); H[1] = np.eye(6)
        return b1 - b0
    return CustomFactor(noise, [key0, key1], err)


# ----------------------------------------------------------------------------
# Regularizers + priors (match solver.cpp / regularization.h)
# ----------------------------------------------------------------------------
def make_minsnap_factor(prob, seg, noise):
    """Min-snap: r = p^(4) at u=0.5 of position segment `seg` (CPs [seg..seg+5])."""
    pos_keys = [P(seg + l) for l in range(N_POS)]
    t_rel = (seg + (N_POS - 1) + 0.5) * prob.dt_pos
    k = prob._pos.find_knot_span(t_rel)
    N4 = prob._pos.basis_functions(t_rel, k, 4)        # 4th-derivative basis (6,)

    def err(this, values, H):
        P6 = [np.asarray(values.atVector(key), float) for key in pos_keys]
        snap = np.zeros(3)
        for l in range(N_POS):
            snap = snap + N4[l] * P6[l]
        if H is not None:
            for l in range(N_POS):
                H[l] = N4[l] * np.eye(3)               # linear in CPs -> exact Jacobian
        return snap

    return CustomFactor(noise, pos_keys, err)


def make_angaccel_factor(prob, i, noise):
    """Angular accel: r = log(q1^-1 q2) - log(q0^-1 q1) over knots [i,i+1,i+2]."""
    ori_keys = [R(i), R(i + 1), R(i + 2)]

    def residual(R3):
        op = so3_log(R3[0].T @ R3[1])
        on = so3_log(R3[1].T @ R3[2])
        return on - op

    def err(this, values, H):
        R3 = [values.atRot3(k).matrix() for k in ori_keys]
        r0 = residual(R3)
        if H is not None:
            for idx in range(3):
                J = np.zeros((3, 3))
                for a in range(3):
                    d = np.zeros(3); d[a] = _FD
                    R3p = list(R3); R3p[idx] = R3[idx] @ so3_exp(d)
                    J[:, a] = (residual(R3p) - r0) / _FD
                H[idx] = J
        return r0

    return CustomFactor(noise, ori_keys, err)


def make_bias_prior(prob, b0, noise, bias_key=None):
    """r = bias - b0 (per-component weights live in the noise model)."""
    bias_key = B(0) if bias_key is None else bias_key
    b0 = np.asarray(b0, float)

    def err(this, values, H):
        b = np.asarray(values.atVector(bias_key), float)
        if H is not None:
            H[0] = np.eye(6)
        return b - b0
    return CustomFactor(noise, [bias_key], err)


def make_vec_prior(key, ref, noise):
    """Strong anchor on a vector variable (boundary CP / bias pinning)."""
    ref = np.asarray(ref, float)
    n = ref.shape[0]

    def err(this, values, H):
        x = np.asarray(values.atVector(key), float)
        if H is not None:
            H[0] = np.eye(n)
        return x - ref
    return CustomFactor(noise, [key], err)


def make_rot_prior(key, R_ref_mat, noise):
    """Strong anchor on a Rot3 variable: r = Log(R_ref^T R) (right-tangent)."""
    R_ref = np.asarray(R_ref_mat, float)

    def residual(Rm):
        return so3_log(R_ref.T @ Rm)

    def err(this, values, H):
        Rm = values.atRot3(key).matrix()
        r0 = residual(Rm)
        if H is not None:
            J = np.zeros((3, 3))
            for a in range(3):
                d = np.zeros(3); d[a] = _FD
                J[:, a] = (residual(Rm @ so3_exp(d)) - r0) / _FD
            H[0] = J
        return r0
    return CustomFactor(noise, [key], err)
