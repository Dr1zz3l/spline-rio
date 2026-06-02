"""
dump_linear_system.py — Session 1.2

Drives the C++ solver with dump_system=True and saves the linearized system
snapshots (J, r, grad, block_map) for use by prototype_banded_solver.py.

Three scenarios are dumped:
  (a) batch racing   — 7-DoF arrowhead (bias 6 + pitch 1), no marg prior
  (b) SW racing      — 7-DoF + 30-dim marg boundary block = 37-DoF corner
  (c) SW backflips   — 6-DoF (locked extrinsics) + 30-dim marg = 36-DoF corner

Usage (from analysis/):
    python dump_linear_system.py

Output: data/linear_system_dumps/  (created if absent)
  {bag}_{mode}_{snapshot}.npz  — sparse J + metadata (r, grad, block_map)
"""

import sys, os, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'lib'))

# ---------------------------------------------------------------------------
# Import the existing live-solver infrastructure
# ---------------------------------------------------------------------------
from config_loader import load_configs
from lib.rosbag_loader.loader import load_bag_topics
import validate_live_solver as vls

OUT_DIR = Path(__file__).parent.parent / 'data' / 'linear_system_dumps'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dump scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    # (bag_alias, mode, extra_flags)
    ('slow_racing_best_velocity', 'batch',  ['--mocap-yaw', '--cpp']),
    ('slow_racing_best_velocity', 'sw',     ['--mocap-yaw', '--cpp', '--sliding-window']),
    ('fast_racing_best_velocity', 'sw',     ['--mocap-yaw', '--cpp', '--sliding-window']),
    ('backflips_best_velocity',   'sw',     ['--mocap-yaw', '--cpp', '--sliding-window']),
]

# Number of windows to dump per SW scenario (first N windows, covering
# pre-prior, first-prior, and steady-state).
N_SW_WINDOWS = 3


def save_dump(dump, tag: str):
    """Save a SystemDump to OUT_DIR/{tag}_{snapshot}.npz."""
    if not dump.valid:
        print(f'  [SKIP] {tag}: dump not valid')
        return

    # Build the scipy CSR matrix
    J = sp.csr_matrix(
        (np.array(dump.jac_values),
         np.array(dump.jac_cols),
         np.array(dump.jac_row_ptr)),
        shape=(dump.jac_num_rows, dump.jac_num_cols)
    )

    # Serialise block_map as a list of dicts (JSON-able)
    bmap = [
        {'type_id': e.type_id, 'index': e.index,
         'col_offset': e.col_offset, 'tangent_size': e.tangent_size}
        for e in dump.block_map
    ]

    path_npz = OUT_DIR / f'{tag}.npz'
    np.savez(str(path_npz),
             jac_values  = np.array(dump.jac_values),
             jac_cols    = np.array(dump.jac_cols),
             jac_row_ptr = np.array(dump.jac_row_ptr),
             jac_shape   = np.array([dump.jac_num_rows, dump.jac_num_cols]),
             residuals   = np.array(dump.residuals),
             gradient    = np.array(dump.gradient),
             block_map_json = np.array([json.dumps(bmap)]))

    nnz = len(dump.jac_values)
    density = nnz / (dump.jac_num_rows * dump.jac_num_cols) if dump.jac_num_cols > 0 else 0
    print(f'  Saved {path_npz.name}  '
          f'J={dump.jac_num_rows}×{dump.jac_num_cols}  nnz={nnz}  '
          f'density={density:.2e}  |r|={np.linalg.norm(dump.residuals):.3f}')


def build_cpp_solver_args(bag: str, mode: str, extra_flags: list) -> list:
    """Build a sys.argv-style list for validate_live_solver.main()."""
    args = [bag] + extra_flags + ['--set', 'dump_system=1', '--no-plot']
    return args


def run_batch_dump(bag: str, extra_flags: list):
    """Run one batch solve and dump pre+post snapshots."""
    print(f'\n=== Batch: {bag} ===')
    argv = [bag] + extra_flags + ['--set', 'dump_system=1', '--no-plot']

    # We call into validate_live_solver by monkey-patching sys.argv and
    # intercepting the returned result.  The --cpp flag triggers the C++ path
    # which returns a SolverResult; we retrieve it from the module's last_result.
    try:
        result = vls.run_once(argv)
    except Exception as e:
        print(f'  [ERROR] {e}')
        return

    tag_pre  = f'{bag}_batch_pre'
    tag_post = f'{bag}_batch_post'
    save_dump(result.dump_pre,  tag_pre)
    save_dump(result.dump_post, tag_post)


def run_sw_dump(bag: str, extra_flags: list, n_windows: int):
    """Run SW and dump the first n_windows windows."""
    print(f'\n=== SW: {bag} (first {n_windows} windows) ===')
    argv = [bag] + extra_flags + ['--set', 'dump_system=1', '--no-plot']

    try:
        window_results = vls.run_sw_windows(argv, max_windows=n_windows)
    except Exception as e:
        print(f'  [ERROR] {e}')
        return

    for wi, result in enumerate(window_results):
        tag_pre  = f'{bag}_sw_w{wi:02d}_pre'
        tag_post = f'{bag}_sw_w{wi:02d}_post'
        save_dump(result.dump_pre,  tag_pre)
        save_dump(result.dump_post, tag_post)


# ---------------------------------------------------------------------------
# Check that validate_live_solver exposes the needed hooks
# ---------------------------------------------------------------------------
def check_hooks():
    needed = ['run_once', 'run_sw_windows']
    missing = [f for f in needed if not hasattr(vls, f)]
    if missing:
        print(f'[WARN] validate_live_solver is missing hooks: {missing}')
        print('       Add run_once() and run_sw_windows() to validate_live_solver.py')
        print('       (see implementation plan Session 1.2)')
        return False
    return True


if __name__ == '__main__':
    print(f'Dump output dir: {OUT_DIR}')
    if not check_hooks():
        sys.exit(1)

    for bag, mode, flags in SCENARIOS:
        if mode == 'batch':
            run_batch_dump(bag, flags)
        else:
            run_sw_dump(bag, flags, N_SW_WINDOWS)

    print(f'\nDone. Files written to {OUT_DIR}')
    print('Next: run analysis/prototype_banded_solver.py')
