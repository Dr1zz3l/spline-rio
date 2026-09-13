"""Knot placement policies for the adaptive orientation spline (V2).

Input: raw gyro measurements (timestamps + angular rates, bias roughly removed
or not — |omega| >> |bias| where it matters). Output: strictly increasing knot
times covering [t_start, t_end].

Two policies:
  - inverse_cdf: knot density proportional to clip(|omega|/omega_ref, 1, r_max),
    placed by inverse-CDF sampling -> exact knot budget, naturally ramped
    (smoothing of |omega| bounds adjacent-interval ratios). Preferred (V0
    finding: hard tier steps create large angular-accel transients).
  - tiered: quantized dt levels with hysteresis + explicit geometric ramping.
    Kept for ablation.

All policies are CAUSAL up to the smoothing half-window (~25 ms lookahead),
which is well below the SW stride (0.3 s) — suitable for live placement.
"""

import numpy as np


def smooth_omega_norm(t_gyro, gyro_xyz, win_sec=0.05):
    """|omega|(t) smoothed with a moving average of win_sec."""
    om = np.linalg.norm(np.asarray(gyro_xyz, dtype=float), axis=1)
    t = np.asarray(t_gyro, dtype=float)
    dt_med = np.median(np.diff(t))
    w = max(1, int(round(win_sec / dt_med)))
    return t, np.convolve(om, np.ones(w) / w, mode='same')


def place_inverse_cdf(t_om, om_smooth, t_start, t_end, dt_base=0.008,
                      dt_min=0.002, dt_max=0.032, omega_ref=1.5,
                      n_knots=None):
    """Inverse-CDF placement.

    Density rho(t) = (1/dt_base) * clip(|omega|/omega_ref, dt_base/dt_max, dt_base/dt_min).
    If n_knots is None the budget follows from integrating rho; otherwise the
    density shape is kept and the budget forced to n_knots (equal-budget mode).
    """
    sel = (t_om >= t_start) & (t_om <= t_end)
    tt, om = t_om[sel], om_smooth[sel]
    if len(tt) < 2:
        raise ValueError("no omega samples in window")
    rel = np.clip(om / omega_ref, dt_base / dt_max, dt_base / dt_min)
    cdf = np.concatenate([[0.0], np.cumsum(rel[:-1] * np.diff(tt))])
    if n_knots is None:
        n_knots = max(4, int(round(cdf[-1] / dt_base)) + 1)
    cdf /= cdf[-1]
    kt = np.interp(np.linspace(0.0, 1.0, n_knots), cdf, tt)
    kt[0], kt[-1] = t_start, t_end
    for i in range(1, len(kt)):           # strict monotonicity
        if kt[i] <= kt[i - 1]:
            kt[i] = kt[i - 1] + 1e-5
    return kt


def place_tiered(t_om, om_smooth, t_start, t_end, tiers=((1.5, 0.016), (4.0, 0.008), (np.inf, 0.004)),
                 hysteresis=0.25, ramp_ratio=2.0):
    """Quantized-tier placement with hysteresis and geometric ramping.

    tiers: ((omega_thresh, dt), ...) — dt for |omega| below thresh (ascending).
    hysteresis: relative band on the thresholds for switching down (coarser).
    ramp_ratio: max ratio between adjacent knot intervals (enforced by inserting
    intermediate knots near tier transitions).
    """
    def tier_dt(om, cur_dt):
        for k, (th, dt) in enumerate(tiers):
            if om < th * (1.0 + (hysteresis if dt > cur_dt else 0.0)):
                return dt
        return tiers[-1][1]

    kt = [t_start]
    cur_dt = tiers[0][1]
    while kt[-1] < t_end - 1e-9:
        om_here = float(np.interp(kt[-1], t_om, om_smooth))
        want = tier_dt(om_here, cur_dt)
        # geometric ramp toward the wanted dt
        if want > cur_dt * ramp_ratio:
            cur_dt = cur_dt * ramp_ratio
        elif want < cur_dt / ramp_ratio:
            cur_dt = cur_dt / ramp_ratio
        else:
            cur_dt = want
        kt.append(min(kt[-1] + cur_dt, t_end))
    if kt[-1] < t_end:
        kt.append(t_end)
    kt = np.array(kt)
    for i in range(1, len(kt)):
        if kt[i] <= kt[i - 1]:
            kt[i] = kt[i - 1] + 1e-5
    return kt


def grid_stats(kt):
    d = np.diff(kt)
    return (f"n={len(kt)}, dt [{d.min()*1e3:.1f}, {d.max()*1e3:.1f}] ms, "
            f"median {np.median(d)*1e3:.1f} ms, "
            f"max adjacent ratio {max((d[1:]/d[:-1]).max(), (d[:-1]/d[1:]).max()):.2f}")
