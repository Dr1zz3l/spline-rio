#!/usr/bin/env python3
"""rio_frontend: the causal radar/IMU front end shared by the live ROS nodes
(rio_gp_node, rio_spline_node).  Moved verbatim out of rio_gp_node.py on
2026-10-13; behaviour unchanged (GP replay bit-identical).  numpy only.

Ports of the validated offline helpers in analysis/validate_live_solver.py:
IMU-anchored per-frame Doppler pre-unwrap with the WLS reset loop, the
aliased-bearing repair (three AOP lobes), the per-return weight channel
(bearing law + alias down-weight), and the live-window re-unwrap + per-stride
RANSAC (GP-22).  The bearing law is per node (set_bearing_law)."""
import numpy as np

# ---------------------------------------------------------------------------
# Minimal rotation helpers (no scipy dependency in the container)
# ---------------------------------------------------------------------------

def so3_exp(w):
    th = np.linalg.norm(w)
    W = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-10:
        return np.eye(3) + W
    return (np.eye(3) + np.sin(th) / th * W
            + (1.0 - np.cos(th)) / (th * th) * (W @ W))


def quat_to_R(q):
    """q = [x, y, z, w] -> R (body->world for a body-attitude quaternion)."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def R_to_quat(R):
    """R -> q = [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def euler_deg_to_R_bs(euler_deg):
    """R_bs = Rz(yaw) Ry(pitch) Rx(roll) — matches the solver's extrinsic."""
    r, p, y = np.radians(np.asarray(euler_deg, float))
    cz, sz, cy, sy, cx, sx = np.cos(y), np.sin(y), np.cos(p), np.sin(p), \
        np.cos(r), np.sin(r)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


G_W = np.array([0.0, 0.0, -9.81])


def wls_ego_velocity(P, v, intens, min_intensity=5.0, min_range=0.2,
                     min_points=5):
    """u . v_wls = v_meas (sensor frame): EXACT port of the harness
    solve_ego_velocity_weighted default path (plain intensity-weighted LSQ,
    min_intensity floor, no Huber) — the WLS-reset loop of the pre-unwrap
    is sensitive to this (a robustified variant diverged on the
    alias-heavy fast bag in replay testing)."""
    r = np.linalg.norm(P, axis=1)
    w_all = np.asarray(intens, float)
    ok = (r >= min_range) & (w_all >= min_intensity)
    if ok.sum() < min_points:
        return None
    U = P[ok] / r[ok, None]
    z = v[ok]
    w = w_all[ok]
    lhs = U.T @ (U * w[:, None])
    rhs = U.T @ (w * z)
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None


# ---------------------------------------------------------------------------
# Causal front end (ports of the validated offline helpers)
# ---------------------------------------------------------------------------

# Per-return radar weight channel (2026-09-10, the harness's deployed model
# carried into the node): the measured per-return law for clean returns
# (main's bearing law, DIRCOS form 'b': s0 = the boresight floor, coef = the
# transverse-speed slope) and the measured alias down-weight for returns the
# unwrapper moved (sigma_al = 0.6 m/s error floor).  Column 5 of the radar
# array = the weight the solver applies (RadarPoint.w, sigma_j =
# sigma_radar / sqrt(w)); columns 6-8 carry the clean weight, the frame's
# alias weight and the raw Doppler so the live-window re-unwrap can
# recompose column 5 after moving a return.  Constants = validate_live_solver
# _BW_CONST['b'] (the --gp branch), _BW_COS_FLOOR, _ALIAS_SIGMA0 / _OMEGA0 / _SIGMA_AL.
# 2026-10-03: the law was still the pre-2026-09-11 (0.0827, 0.03291); now the GP
# harness's (0.0829, 0.03392), overridable via the `bearing_law` parameter.
BW_S0, BW_COEF, BW_COS_FLOOR = 0.0829, 0.03392, 0.15
AL_SIGMA0, AL_OMEGA0, AL_SIGMA_AL = 0.095, 4.5, 0.6


def set_bearing_law(s0, coef):
    """Per-node law: the GP keeps (0.0829, 0.03392), the spline node uses its
    own frame model's fit (0.0572, 0.02328; validate_live_solver _BW_CONST['b'])."""
    global BW_S0, BW_COEF
    BW_S0, BW_COEF = float(s0), float(coef)


def bearing_weights(P, v_wls):
    """Clean-return weights of the per-return law from the frame's WLS
    consensus velocity (sensor frame; the law is quadratic in it)."""
    n = len(P)
    if v_wls is None:
        return np.ones(n)
    rho = np.maximum(np.linalg.norm(P, axis=1), 1e-6)
    s = P / rho[:, None]
    cphi = np.maximum(s[:, 0], BW_COS_FLOOR)
    X = np.sqrt((v_wls[1] - s[:, 1] * v_wls[0] / cphi) ** 2
                + (v_wls[2] - s[:, 2] * v_wls[0] / cphi) ** 2)
    return 1.0 / (1.0 + (BW_COEF * X / BW_S0) ** 2)


def alias_frame_weight(omega_norm):
    """Measured alias down-weight of a frame at body rate |omega|."""
    s_cl2 = AL_SIGMA0 ** 2 * (1.0 + (omega_norm / AL_OMEGA0) ** 2)
    return float(np.clip(s_cl2 / AL_SIGMA_AL ** 2, 0.01, 1.0))


def compose_point_weights(v_final, w_clean, w_al, v_raw):
    """Column 5: alias weight where the ingested Doppler differs from the raw
    one, the clean-return law weight otherwise."""
    aliased = np.abs(v_final - v_raw) > 1e-6
    return np.where(aliased, w_al, w_clean)



# ---------------------------------------------------------------------------
# Aliased-bearing repair (2026-09-16/17, worklog/2026-09-16_alias-three-lobes.md,
# 2026-09-16_alias-repair-frontend.md, 2026-09-17_overnight.md): with the wrapped
# Doppler used for the on-chip TDM compensation, an aliased return of order k
# is reported at the true bearing plus k times ONE of three fixed direction-
# cosine offsets of the AOP virtual array (measured from the chip's own lobe
# multiplets, 0.002); the chip also emits the competing lobes as extra points
# at the same range and Doppler.  Multiplets -> the distinct-lobe assignment
# whose corrected bearings coincide (velocity-free), one point kept; singletons
# -> the best of {k = 0 reported bearing, three lobes} against a reference
# velocity (the frame's clean-return WLS consensus or the IMU prediction,
# whichever has the smaller leave-one-out residual on the clean returns).
# Repaired returns become clean (raw := unwrapped -> clean law weight); the
# original bearing and raw Doppler ride along in columns 9-12 so the causal
# re-unwrap can restore or re-pick a return whose alias order it changes.
# Sensor frame: c1 = -u_y - k Dx, c2 = u_z + k Dz.  Opt-in (~alias_repair).
ALIAS_LOBES = np.array([(-0.271, 0.271), (-0.177, -0.448), (0.448, 0.177)])


def _lobe_candidates(u, k):
    c1 = -u[1] - k * ALIAS_LOBES[:, 0]; c2 = u[2] + k * ALIAS_LOBES[:, 1]
    c1 = (c1 + 1.0) % 2.0 - 1.0; c2 = (c2 + 1.0) % 2.0 - 1.0
    s2 = c1 ** 2 + c2 ** 2
    U = np.stack([np.sqrt(np.clip(1.0 - s2, 0.0, None)), -c1, c2], axis=1)
    U[s2 > 1.0] = np.nan
    return U


def _law_sigma(u_raw, vp):
    cphi = max(float(u_raw[0]), BW_COS_FLOOR)
    X = np.sqrt((vp[1] - u_raw[1] * vp[0] / cphi) ** 2 + (vp[2] - u_raw[2] * vp[0] / cphi) ** 2)
    return np.sqrt(BW_S0 ** 2 + (BW_COEF * X) ** 2)


def alias_pick(U, vp, v_unw, u_raw, v_raw, margin=False):
    """Index of the best lobe, -1 for the k = 0 hypothesis, None when
    `margin` is set and the runner-up is within 2 sigma of the law."""
    vp = np.asarray(vp, float); ok = np.isfinite(U[:, 0])
    if not ok.any():
        return None
    err = np.abs(U @ vp - v_unw); err[~ok] = np.inf
    cand = np.concatenate([[abs(float(u_raw @ vp) - v_raw)], err])
    order = np.argsort(cand)
    if margin and len(order) > 1 and cand[order[1]] - cand[order[0]] < 2.0 * _law_sigma(u_raw, vp):
        return None
    return -1 if order[0] == 0 else int(order[0] - 1)


def alias_repair_frame(P, v_new, v_raw, intens, v_max, v_imu_pred, min_range=0.2, min_points=5):
    """Returns (P2, v2, keep) for one frame (pre-unwrap stage)."""
    import itertools
    n = len(P); rng = np.linalg.norm(P, axis=1)
    k = np.round((v_new - v_raw) / (2.0 * v_max)).astype(int)
    keep = np.ones(n, bool); rep = np.zeros(n, bool); P2 = P.copy(); v2 = v_new.copy()
    al = [i for i in range(n) if k[i] != 0 and rng[i] >= min_range]
    if not al:
        return P2, v2, keep
    u = P / np.maximum(rng, 1e-9)[:, None]
    C = {i: _lobe_candidates(u[i], k[i]) for i in al}
    done = np.zeros(n, bool)
    for a in al:                                   # multiplets, distinct lobes
        if done[a]:
            continue
        grp = [b for b in al if not done[b] and abs(rng[b] - rng[a]) < 0.03
               and abs(v_raw[b] - v_raw[a]) < 0.01 and k[b] == k[a]]
        if len(grp) < 2:
            continue
        best = None
        for combo in itertools.product(range(3), repeat=len(grp)):
            if len(set(combo)) < len(combo):
                continue
            V = np.array([C[g][c] for g, c in zip(grp, combo)])
            if np.isnan(V).any():
                continue
            spread = max(np.linalg.norm(V[p] - V[q]) for p in range(len(grp)) for q in range(p + 1, len(grp)))
            if best is None or spread < best[0]:
                best = (spread, V.mean(axis=0))
        if best is not None and best[0] < 0.15:
            um = best[1] / np.linalg.norm(best[1])
            P2[grp[0]] = rng[grp[0]] * um; rep[grp[0]] = True
            for g in grp[1:]:
                keep[g] = False
            for g in grp:
                done[g] = True
    # reference for the singletons: clean consensus vs IMU prediction, leave-one-out validated
    cl = (v_new == v_raw) & (rng >= min_range)
    vref = v_imu_pred
    if cl.sum() >= 5:
        vw = wls_ego_velocity(P[cl], v_new[cl], intens[cl], min_range=min_range, min_points=min_points)
        if vw is not None:
            Pc, vc, Ic = P[cl], v_new[cl], intens[cl]; Uc = Pc / np.linalg.norm(Pc, axis=1, keepdims=True)
            loo = []
            for j in range(len(vc)):
                m = np.ones(len(vc), bool); m[j] = False
                vj = wls_ego_velocity(Pc[m], vc[m], Ic[m], min_range=min_range, min_points=4)
                loo.append((Uc[j] @ vj - vc[j]) if vj is not None else np.nan)
            loo = np.array(loo, float)
            e_w = np.sqrt(np.nanmean(loo ** 2)) if np.isfinite(loo).sum() >= 4 else np.inf
            e_i = np.sqrt(np.mean((Uc @ v_imu_pred - vc) ** 2)) if v_imu_pred is not None else np.inf
            if e_w < e_i:
                vref = vw
    for a in al:                                   # singletons
        if done[a]:
            continue
        done[a] = True
        if vref is None:
            continue
        j = alias_pick(C[a], vref, v_new[a], u[a], v_raw[a])
        if j is None:
            continue
        if j == -1:
            v2[a] = v_raw[a]
            continue
        P2[a] = rng[a] * C[a][j]; rep[a] = True
    return P2, v2, keep


class PreUnwrapper:
    """Incremental IMU-anchored Doppler de-aliasing (preunwrap_radar_frames
    port): world velocity integrated from accel between frames, alias of each
    return picked nearest the prediction, WLS fit resets the velocity."""

    def __init__(self, R_bs, v_max, ba, bg, R_wb0, t0, min_range=0.2, alias_repair=False):
        self.R_bs, self.v_max = R_bs, v_max
        self.alias_repair = bool(alias_repair)
        self.ba, self.bg = np.asarray(ba, float), np.asarray(bg, float)
        self.R_wb = np.asarray(R_wb0, float).copy()   # gyro-integrated chain
        self.t_att = float(t0)
        self.v_imu = np.zeros(3)
        self.t_vel = float(t0)
        self.v_prev_wls = None
        self.om_last = 0.0          # |gyro| of the latest sample (alias weight)
        self.last_speed = 0.0       # |consensus velocity| of the last frame (spline sigma_c(v))

    def advance_imu(self, imu_blk):
        """Advance the gyro attitude chain + accel velocity with samples
        (t, ax..az, gx..gz), strictly increasing."""
        for s in imu_blk:
            t = float(s[0])
            dt = t - self.t_att
            if 0.0 < dt < 0.1:
                self.R_wb = self.R_wb @ so3_exp((s[4:7] - self.bg) * dt)
                dv = (self.R_wb @ (s[1:4] - self.ba) + G_W) * dt
                if 0.0 < t - self.t_vel < 0.5:
                    self.v_imu += dv
            self.t_att = t
            self.t_vel = max(self.t_vel, t)
            self.om_last = float(np.linalg.norm(s[4:7]))

    def unwrap_frame(self, pts, min_range=0.2, min_points=5):
        """De-alias one frame (n,5); returns (n,9) with the weight channel
        (columns 5-8: composed weight, clean weight, frame alias weight, raw
        Doppler)."""
        P, v = pts[:, :3], pts[:, 3].astype(float).copy()
        v_raw = pts[:, 3].astype(float).copy()
        r = np.linalg.norm(P, axis=1)
        ok = r >= min_range
        if self.v_max is not None and self.v_max > 0 and ok.any():
            v_pred = -(self.R_bs.T @ (self.R_wb.T @ self.v_imu))
            U = P[ok] / r[ok, None]
            pr = U @ v_pred
            v0 = v[ok]
            best = v0.copy()
            for k in (-1.0, 1.0):
                cand = v0 + k * 2.0 * self.v_max
                bet = np.abs(cand - pr) < np.abs(best - pr)
                best = np.where(bet, cand, best)
            v[ok] = best
        P_orig = np.asarray(P, float).copy(); v_orig = v_raw.copy(); intens = np.asarray(pts[:, 4], float)
        if self.alias_repair and self.v_max is not None and self.v_max > 0:
            v_pred_r = -(self.R_bs.T @ (self.R_wb.T @ self.v_imu))
            P2, v2, keep = alias_repair_frame(P_orig, v, v_raw, intens, self.v_max, v_pred_r, min_range, min_points)
            repd = np.any(P2 != P_orig, axis=1)
            v_raw = np.where(repd, v2, v_raw)          # repaired -> clean (raw := unwrapped)
            P, v, v_raw, intens = P2[keep], v2[keep], v_raw[keep], intens[keep]
            P_orig, v_orig = P_orig[keep], v_orig[keep]
            pts = pts[keep]
        v_wls = wls_ego_velocity(P, v, intens, min_range, min_points)
        if v_wls is not None:
            self.v_prev_wls = v_wls
            self.v_imu = self.R_wb @ (self.R_bs @ (-v_wls))
        # the harness's _SPEED_FRAME_V: this frame's consensus, else the last one
        _vc = v_wls if v_wls is not None else self.v_prev_wls
        self.last_speed = float(np.linalg.norm(_vc)) if _vc is not None else 0.0
        out = np.zeros((len(pts), 13 if self.alias_repair else 9))
        out[:, :5] = pts[:, :5]
        out[:, :3] = P
        out[:, 3] = v
        if self.alias_repair:
            out[:, 9:12] = P_orig; out[:, 12] = v_orig
        w_clean = bearing_weights(P, v_wls if v_wls is not None else self.v_prev_wls)
        w_al = alias_frame_weight(self.om_last)
        out[:, 6] = w_clean
        out[:, 7] = w_al
        out[:, 8] = v_raw
        out[:, 5] = compose_point_weights(v, w_clean, w_al, v_raw)
        return out


def _so3_exp(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def causal_reunwrap_stride(radar_list, pred, R_bs, t_bs, v_max, thr, rng,
                           reunwrap=True, min_range=0.2):
    """Live-window re-unwrap + per-stride RANSAC (GP-22, verbatim port of
    _gp_causal_unwrap_stride). pred = (v_world, omega, R_wb) at the previous
    stride's live edge; flip strides pass reunwrap=False and keep the
    IMU-anchored baseline. Returns (new_radar_list, n_reunwrapped)."""
    step = 2.0 * v_max if v_max else 0.0
    # pred may carry (t_pred, imu (N,7), b_a, b_g): the prediction is then
    # PROPAGATED with the IMU to each frame's own stamp (2026-09-27 port of the
    # harness fix: a stale live-edge velocity flips self-consistent frames in
    # pull-ups; worklog/2026-09-27_spline-vertical-drift.md)
    prop = reunwrap and pred is not None and len(pred) >= 7

    def v_pred_at(t_f):
        v_world, omega, R_wb = pred[:3]
        if prop:
            t_p, imu, b_a, b_g = pred[3:7]
            R = np.array(R_wb, float); v = np.array(v_world, float); tc = float(t_p)
            m = (imu[:, 0] > t_p) & (imu[:, 0] <= t_f)
            for row in imu[m]:
                dt = row[0] - tc
                if 0.0 < dt < 0.05:
                    v = v + (R @ (row[1:4] - b_a) + np.array([0.0, 0.0, -9.81])) * dt
                    R = R @ _so3_exp((row[4:7] - b_g) * dt)
                tc = row[0]
            if m.any():
                omega = imu[m][-1, 4:7] - b_g
            v_world, R_wb = v, R
        v_ant = R_wb.T @ v_world + np.cross(omega, t_bs)
        return -(R_bs.T @ v_ant)
    if reunwrap and pred is not None:
        v_pred = v_pred_at(pred[3] if prop else 0.0)
    out, n_re = [], 0
    for (t, pts) in radar_list:
        if prop:
            v_pred = v_pred_at(t)
        P = pts[:, :3]
        v_new = pts[:, 3].astype(float).copy()
        r = np.linalg.norm(P, axis=1)
        ok = r >= min_range
        if reunwrap and pred is not None and ok.any() and step > 0:
            U = P[ok] / r[ok, None]
            pr = U @ v_pred
            v0 = v_new[ok]
            best = v0.copy()
            for cand in (v0 - step, v0 + step):
                bet = np.abs(cand - pr) < np.abs(best - pr)
                best = np.where(bet, cand, best)
            n_re += int((best != v0).sum())
            v_new[ok] = best
        if pts.shape[1] >= 13 and step > 0:          # alias repair on: redo the CHANGED returns from the originals
            v0_all = pts[:, 3].astype(float)
            ch = np.where(np.abs(v_new - v0_all) > 1e-6)[0]
            if len(ch):
                pts = pts.copy(); vp_ = v_pred if (reunwrap and pred is not None) else None
                for i in ch:
                    op, ov = pts[i, 9:12], float(pts[i, 12])
                    kk = int(np.round((v_new[i] - ov) / step))
                    if kk == 0:
                        pts[i, :3] = op; pts[i, 8] = ov
                        continue
                    r_i = np.linalg.norm(op); uo = op / max(r_i, 1e-9)
                    if vp_ is None:
                        continue
                    U = _lobe_candidates(uo, kk)
                    jj = alias_pick(U, vp_, v_new[i], uo, ov, margin=True)
                    if jj is None:
                        pts[i, :3] = op; pts[i, 8] = ov
                    elif jj == -1:
                        v_new[i] = ov; pts[i, :3] = op; pts[i, 8] = ov
                    else:
                        pts[i, :3] = r_i * U[jj]; pts[i, 8] = v_new[i]
                P = pts[:, :3]
        if thr > 0 and len(v_new) >= 5:
            Hn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
            bestm = None
            for _ in range(150):
                idx = rng.choice(len(v_new), 3, replace=False)
                try:
                    vc = np.linalg.solve(Hn[idx], v_new[idx])
                except np.linalg.LinAlgError:
                    continue
                inl = np.abs(Hn @ vc - v_new) < thr
                if bestm is None or inl.sum() > bestm.sum():
                    bestm = inl
            mask = bestm if bestm is not None and bestm.sum() >= 5 \
                else np.ones(len(v_new), bool)
        else:
            mask = np.ones(len(v_new), bool)
        pts2 = pts[mask].copy()
        pts2[:, 3] = v_new[mask]
        if pts2.shape[1] >= 9:   # weight channel: recompose after the move
            pts2[:, 5] = compose_point_weights(pts2[:, 3], pts2[:, 6],
                                               pts2[:, 7], pts2[:, 8])
        out.append((t, pts2))
    return out, n_re
