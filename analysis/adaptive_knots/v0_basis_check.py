"""V0 — validate the non-uniform cumulative B-spline basis math BEFORE any C++ work.

Checks (all must pass):
  1. scipy oracle: basis values + 1st derivatives on random NON-uniform knot
     vectors match scipy.interpolate.BSpline to ~1e-10.
  2. basalt uniform parity: with a uniform knot vector, the cumulative coeffs
     (lam, dlam) reproduce basalt's cumulative_blending_matrix_ formula
     (ported exactly from spline_common.h) to machine precision.
  3. Continuity: R(t) and omega(t) of a non-uniform cumulative SO(3) spline are
     numerically C1/C0-continuous across density-transition knots, and omega is
     consistent with the finite-difference of R (the actual gyro-residual path).

Run:  cd analysis && ../.venv/bin/python3 adaptive_knots/v0_basis_check.py
"""

import sys
from math import comb, factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nonuniform_bspline import (DEG, N_ORDER, NonUniformSO3Spline,
                                basis_active, cumulative_coeffs, extend_knots,
                                find_span, so3_exp, so3_log)

rng = np.random.default_rng(42)
FAIL = []


def check(name, value, tol):
    ok = value < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.3e}  (tol {tol:.0e})")
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# 1. scipy oracle on random non-uniform knot vectors
# ---------------------------------------------------------------------------
print("== Test 1: non-uniform basis vs scipy ==")
from scipy.interpolate import BSpline

max_err_B, max_err_dB, max_partition = 0.0, 0.0, 0.0
for trial in range(20):
    K = rng.integers(10, 40)
    # random spacing spanning 2 orders of magnitude (mimics 4ms..32ms tiers + jitter)
    dts = rng.uniform(0.002, 0.04, size=K - 1)
    kt = np.concatenate([[0.0], np.cumsum(dts)])
    ext = extend_knots(kt)

    for _ in range(50):
        t = rng.uniform(kt[DEG], kt[-1] - 1e-9)
        k = find_span(kt, t)
        B, dB = basis_active(kt, ext, k, t)
        max_partition = max(max_partition, abs(B.sum() - 1.0))
        for l in range(4):
            j = k - 3 + l                      # control point (real indexing)
            c = np.zeros(len(ext) - N_ORDER)   # scipy: n = len(t) - k - 1
            c[j + DEG] = 1.0                   # scipy basis index = j + DEG
            s = BSpline(ext, c, DEG)
            max_err_B = max(max_err_B, abs(s(t) - B[l]))
            max_err_dB = max(max_err_dB, abs(s.derivative()(t) - dB[l]))

check("basis value vs scipy (non-uniform)", max_err_B, 1e-10)
check("basis 1st derivative vs scipy (non-uniform)", max_err_dB, 1e-8)
check("partition of unity", max_partition, 1e-12)

# ---------------------------------------------------------------------------
# 2. basalt uniform parity (exact port of computeBlendingMatrix, spline_common.h)
# ---------------------------------------------------------------------------
print("== Test 2: uniform parity vs basalt cumulative blending matrix ==")


def basalt_blending_matrix(N, cumulative):
    m = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ssum = 0.0
            for s in range(j, N):
                ssum += (-1.0) ** (s - j) * comb(N, s - j) * (N - s - 1.0) ** (N - 1.0 - i)
            m[j, i] = comb(N - 1, N - 1 - i) * ssum
    if cumulative:
        for i in range(N):
            for jj in range(i + 1, N):
                m[i, :] += m[jj, :]
    return m / factorial(N - 1)


M_cum = basalt_blending_matrix(N_ORDER, cumulative=True)

max_err_lam, max_err_dlam = 0.0, 0.0
for dt in (0.0008, 0.008, 0.04):
    K = 60
    kt = dt * np.arange(K)
    ext = extend_knots(kt)
    for _ in range(200):
        t = rng.uniform(kt[DEG], kt[-1] - 1e-9)
        k = find_span(kt, t)
        u = (t - kt[k]) / dt
        p = np.array([1.0, u, u**2, u**3])
        dp = np.array([0.0, 1.0, 2 * u, 3 * u**2])
        lam_b = M_cum @ p
        dlam_b = (M_cum @ dp) / dt
        lam, dlam = cumulative_coeffs(kt, ext, k, t)
        max_err_lam = max(max_err_lam, np.abs(lam - lam_b).max())
        max_err_dlam = max(max_err_dlam, np.abs(dlam - dlam_b).max() * dt)  # relative to 1/dt scale

check("cumulative coeffs vs basalt (uniform)", max_err_lam, 1e-12)
check("cumulative dcoeffs vs basalt (uniform, scaled)", max_err_dlam, 1e-11)

# boundary spans (ghost-knot handling): first and last valid span explicitly
max_err_bnd = 0.0
kt = 0.008 * np.arange(12)
ext = extend_knots(kt)
for t in (kt[DEG] + 1e-6, kt[-1] - 1e-6):
    k = find_span(kt, t)
    u = (t - kt[k]) / 0.008
    lam_b = M_cum @ np.array([1.0, u, u**2, u**3])
    lam, _ = cumulative_coeffs(kt, ext, k, t)
    max_err_bnd = max(max_err_bnd, np.abs(lam - lam_b).max())
check("boundary spans with ghost knots (uniform)", max_err_bnd, 1e-12)

# ---------------------------------------------------------------------------
# 3. Continuity of R(t), omega(t) across density transitions + FD consistency
# ---------------------------------------------------------------------------
print("== Test 3: SO(3) spline continuity across density transitions ==")

# knot vector with hard tier transitions: 32ms -> 4ms -> 16ms
kt = [0.0]
for dt, t_end in ((0.032, 0.5), (0.004, 1.0), (0.016, 1.5)):
    while kt[-1] < t_end - 1e-12:
        kt.append(kt[-1] + dt)
kt = np.array(kt)

# smooth synthetic rotation: integrate omega(t) = moderately aggressive sinusoids
def omega_true(t):
    return np.array([6.0 * np.sin(2 * np.pi * 1.3 * t),
                     5.0 * np.cos(2 * np.pi * 0.9 * t + 0.4),
                     3.0 * np.sin(2 * np.pi * 2.1 * t + 1.1)])

R_knots = [np.eye(3)]
h_int = 1e-4
t_acc = kt[0]
R_acc = np.eye(3)
for j in range(1, len(kt)):
    while t_acc < kt[j] - 1e-12:
        h = min(h_int, kt[j] - t_acc)
        R_acc = R_acc @ so3_exp(omega_true(t_acc + 0.5 * h) * h)
        t_acc += h
    R_knots.append(R_acc.copy())
spline = NonUniformSO3Spline(kt, np.array(R_knots))

h = 1e-6
max_fd_err, max_jump_R = 0.0, 0.0
# probe AT every interior knot (where continuity could break), esp. transitions.
# NOTE (V0 finding): omega is C0 but its slope (angular accel) transiently reaches
# ~1/dt_dense^2 * |delta_coarse| at HARD density transitions (~13000 rad/s^2 for
# 32->4 ms) because second-derivative basis terms multiply increments delta_j of
# very different magnitudes. A geometric ramp (32->16->8->4) cuts this ~7x.
# So the correct C1 test is h-SCALING of the omega difference, not a fixed jump tol.
alpha_eff = {}
for tau in kt[DEG + 1:-1]:
    R_m, w_m = spline.evaluate(tau - h)
    R_p, w_p = spline.evaluate(tau + h)
    alpha_eff[tau] = np.linalg.norm(w_p - w_m) / (2 * h)
    max_jump_R = max(max_jump_R, np.linalg.norm(so3_log(R_m.T @ R_p)))

tau_worst = max(alpha_eff, key=alpha_eff.get)
ratios = []
for hh in (1e-6, 1e-7, 1e-8):
    _, w_m = spline.evaluate(tau_worst - hh)
    _, w_p = spline.evaluate(tau_worst + hh)
    ratios.append(np.linalg.norm(w_p - w_m) / (2 * hh))
# C1 <=> difference quotient converges to a finite alpha (ratios constant in h)
c1_dev = abs(ratios[-1] - ratios[0]) / ratios[-1]
check("omega C1 (h-scaling of difference quotient)", c1_dev, 0.05)
print(f"  [info] max effective |alpha| at knots: {max(alpha_eff.values()):.0f} rad/s^2 "
      f"at t={tau_worst:.3f} (hard 32->4ms transition; ramped grids ~7x lower)")

# FD consistency of omega vs R everywhere (random + near transitions)
ts = np.concatenate([rng.uniform(spline.t_start, spline.t_end - 1e-6, 300),
                     np.repeat([0.5, 1.0], 20) + rng.uniform(-0.01, 0.01, 40)])
ts = ts[(ts > spline.t_start + 1e-3) & (ts < spline.t_end - 1e-3)]
for t in ts:
    R0, _ = spline.evaluate(t - h)
    R1, w_mid = spline.evaluate(t)
    R2, _ = spline.evaluate(t + h)
    w_fd = so3_log(R0.T @ R2) / (2 * h)
    max_fd_err = max(max_fd_err, np.linalg.norm(w_fd - w_mid))

# R jump ~ |w|*2h ~ 1e-5 expected
check("R(t) continuity at knots (rad)", max_jump_R, 1e-4)
check("omega vs FD of R (rad/s)", max_fd_err, 1e-3)

# representation sanity: CPs sampled at GREVILLE abscissae
# xi_j = (tau_{j+1}+tau_{j+2}+tau_{j+3})/3  (ext indexing offset +3).
# V0 finding: sampling CPs at tau_j instead introduces a ~2*dt time LAG of the
# whole curve (10.5 deg mean on this signal!); Greville sampling -> 0.18 deg.
# Phase-1 init code MUST sample the dead-reckoned R(t) at Greville times.
ext = spline.ext


def truth_R(t):
    R_gt = np.eye(3)
    tt = 0.0
    while tt < t - 1e-12:
        hh = min(1e-4, t - tt)
        R_gt = R_gt @ so3_exp(omega_true(tt + 0.5 * hh) * hh)
        tt += hh
    return R_gt


# cache truth on fine grid
ts_f = np.arange(0.0, kt[-1] + 0.1, 1e-4)
Rs_f = [np.eye(3)]
for i in range(1, len(ts_f)):
    Rs_f.append(Rs_f[-1] @ so3_exp(omega_true(ts_f[i - 1] + 5e-5) * 1e-4))


def truth(t):
    return Rs_f[min(int(round(t / 1e-4)), len(Rs_f) - 1)]


R_grev = []
for j in range(len(kt)):
    xi = (ext[j + 4] + ext[j + 5] + ext[j + 6]) / 3.0
    R_grev.append(truth(np.clip(xi, 0.0, kt[-1])))
spline_g = NonUniformSO3Spline(kt, np.array(R_grev))

errs = []
for t in np.arange(spline_g.t_start, spline_g.t_end - 1e-9, 0.003):
    R_s, _ = spline_g.evaluate(t)
    errs.append(np.degrees(np.linalg.norm(so3_log(R_s.T @ truth(t)))))
errs = np.array(errs)
print(f"  [info] representation error, Greville-sampled CPs (deg): "
      f"mean {errs.mean():.3f}, max {errs.max():.3f}")
check("Greville-sampled representation error mean (deg)", float(errs.mean()), 1.0)

print()
if FAIL:
    print(f"V0 FAILED: {FAIL}")
    sys.exit(1)
print("V0 PASSED — non-uniform basis math validated.")
