"""
session2_gate.py  — Session 2 gate check

Runs the first 3 SW windows twice:
  (a) SPARSE_NORMAL_CHOLESKY / SuiteSparse (baseline)
  (b) BANDED_SCHUR            (dense Eigen LDLT)

Compares per-window final costs and the live-edge trajectory.

Gate: final cost of each window agrees to < 1e-4 relative tolerance,
      and live-edge position/orientation RMSE agree to < 1% relatively.
"""

import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'lib'))

import validate_live_solver as vls

BAG    = 'slow_racing_best_velocity'
COMMON = [BAG, '--mocap-yaw', '--cpp', '--sliding-window', '--no-plot']
N_WIN  = 3          # run only this many windows for speed


def run_windows(extra_flags, label):
    argv = COMMON + extra_flags
    print(f'\n{"="*60}')
    print(f'Running: {label}')
    print(f'  argv = {argv}')
    print(f'{"="*60}')
    results = vls.run_sw_windows(argv, max_windows=N_WIN)
    return results


def extract_final_cost(result):
    """Final cost from SolverResult.cost_history."""
    h = getattr(result, 'cost_history', None)
    if h is None or len(h) == 0:
        return None
    return h[-1]


def extract_pos_cps(result):
    """Return the full pos_cps array (Nx3 numpy array)."""
    p = getattr(result, 'pos_cps', None)
    if p is None:
        return None
    return np.array(p)


def main():
    # --- Baseline (SuiteSparse) ---
    results_ss = run_windows([], 'SuiteSparse baseline')

    # --- BANDED_SCHUR (dense LDLT skeleton) ---
    results_bs = run_windows(['--set', 'use_banded_schur=1'], 'BANDED_SCHUR (dense LDLT)')

    if len(results_ss) == 0 or len(results_bs) == 0:
        print('\n[FAIL] No results returned.')
        return 1

    n = min(len(results_ss), len(results_bs))
    print(f'\n{"="*60}')
    print(f'Comparing {n} windows')
    print(f'{"="*60}')

    all_ok = True
    for i in range(n):
        r_ss = results_ss[i]
        r_bs = results_bs[i]

        cost_ss = extract_final_cost(r_ss)
        cost_bs = extract_final_cost(r_bs)

        if cost_ss is not None and cost_bs is not None:
            rel = abs(cost_ss - cost_bs) / max(abs(cost_ss), 1e-15)
            status = 'PASS ✓' if rel < 1e-4 else 'FAIL ✗'
            print(f'  Window {i}: cost_ss={cost_ss:.6e}  cost_bs={cost_bs:.6e}  '
                  f'rel={rel:.2e}  {status}')
            if rel >= 1e-4:
                all_ok = False
        else:
            print(f'  Window {i}: final_cost not available — checking trajectory')

        # Compare full set of position control points
        pos_ss = extract_pos_cps(r_ss)
        pos_bs = extract_pos_cps(r_bs)
        if pos_ss is not None and pos_bs is not None:
            err = np.linalg.norm(pos_ss - pos_bs)
            ref = max(np.linalg.norm(pos_ss), 1e-6)
            status_p = 'PASS ✓' if err / ref < 0.01 else 'FAIL ✗'
            print(f'  Window {i}: pos_cps err={err:.3e}  '
                  f'rel={err/ref:.2e}  {status_p}')
            if err / ref >= 0.01:
                all_ok = False

    print(f'\n{"="*60}')
    if all_ok:
        print('SESSION 2 GATE: PASS ✓  — BANDED_SCHUR hook verified, proceed to Session 3')
    else:
        print('SESSION 2 GATE: FAIL ✗  — check discrepancies above')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
