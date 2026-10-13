#!/usr/bin/env python3
"""Unit test of the SlidingWindowSolver live-operation API (2026-10-13):
set_init_tail / trim_before / sample.  Run with the python the module was
built for:  python3 tests/test_sw_live_api.py [build_dir]"""
import sys
from pathlib import Path
import numpy as np

build = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'build_release'
sys.path.insert(0, str(build))
import rio_solver


def so3_exp(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def R_to_q(R):
    t = np.trace(R)
    s = np.sqrt(t + 1.0) * 2
    return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])


def q_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


cfg = rio_solver.SolverConfig()
cfg.dt_pos, cfg.dt_ori = 0.04, 0.016
ext = rio_solver.ExtrinsicConfig()
sw = rio_solver.SlidingWindowSolver(cfg, ext)
t_ref = 1773246000.0
n_p, n_o = 60, 150
P = np.stack([np.sin(0.3 * np.arange(n_p)), np.cos(0.2 * np.arange(n_p)), 0.01 * np.arange(n_p)], 1)
Q = np.array([R_to_q(so3_exp(np.array([0.1, -0.05, 0.2]) * k * 0.016 * 5)) for k in range(n_o)])
sw.initialize(P, Q, np.zeros(6), t_ref)
fails = 0

# 1. sample: velocity = d/dt position, omega = body rate of R(t)
ts = list(t_ref + np.linspace(0.5, 1.5, 7))
q, p, v, w = sw.sample(ts)
h = 1e-3   # absolute epoch times: a smaller step is rounding-limited
_, p_p, _, _ = sw.sample([t + h for t in ts]); _, p_m, _, _ = sw.sample([t - h for t in ts])
dv = np.abs((p_p - p_m) / (2 * h) - v).max()
q_p, _, _, _ = sw.sample([t + h for t in ts])
w_fd = []
for a, b in zip(q, q_p):
    D = q_to_R(a).T @ q_to_R(b)
    w_fd.append(np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2 * h))
dw = np.abs(np.array(w_fd) - w).max()
print(f"sample: |v - dp/dt| {dv:.2e}, |omega - body rate(FD)| {dw:.2e}")
fails += (dv > 1e-3) + (dw > 1e-3)

# 2. set_init_tail: append past the end, overwrite a tail
tail_p = np.tile(P[-1], (10, 1)) + 0.1
tail_q = np.tile(Q[-1], (20, 1))
sw.set_init_tail(n_p - 2, tail_p, n_o, tail_q)
ok = sw.n_pos_cps == n_p - 2 + 10 and sw.n_ori_knots == n_o + 20
print(f"set_init_tail: n_pos {sw.n_pos_cps} (exp {n_p + 8}), n_ori {sw.n_ori_knots} (exp {n_o + 20}) -> {'OK' if ok else 'FAIL'}")
fails += (not ok)
try:
    sw.set_init_tail(sw.n_pos_cps + 1, tail_p, 0, tail_q[:0])
    print("set_init_tail gap: NOT rejected -> FAIL"); fails += 1
except Exception:
    print("set_init_tail gap: rejected OK")

# 3. trim: identical samples at and after t_keep, t_ref advanced by whole 0.08 s periods
q0, p0, v0, w0 = sw.sample(ts)
t_new = sw.trim_before(ts[0])
q1, p1, v1, w1 = sw.sample(ts)
per = (t_new - t_ref) / 0.08
d = max(np.abs(p1 - p0).max(), np.abs(v1 - v0).max(), np.abs(w1 - w0).max(), np.abs(np.abs((q1 * q0).sum(1)) - 1).max())
# absolute epoch stamps: re-basing t_ref changes t_rel by ~1 ulp (2.4e-7 s) -> ~1e-6 in the samples
ok = abs(per - round(per)) < 1e-6 and per >= 1 and d < 1e-5
print(f"trim: t_ref +{t_new - t_ref:.3f} s ({per:.0f} periods), n_pos {sw.n_pos_cps}, n_ori {sw.n_ori_knots}, max sample change {d:.1e} -> {'OK' if ok else 'FAIL'}")
fails += (not ok)
print("ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
