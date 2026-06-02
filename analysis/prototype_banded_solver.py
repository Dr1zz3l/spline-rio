"""
prototype_banded_solver.py — Session 1.3

Reads .npz files from dump_linear_system.py and verifies that banded-Cholesky
+ arrowhead-Schur produces exactly the same update step δx as a reference
direct solve (scipy sparse for large n, dense for small n ≤ 5000).

Algorithm
---------
Given the tangent-space normal-equation system H δx = g  (H = JᵀJ, g = -Jᵀr):

1. Partition columns by type from block_map:
     globals c  = bias [+ pitch_delta]   (6 or 7 DoF — the arrowhead)
     knots   k  = ori_knot + pos_cp      (everything else — the band)

2. Reorder knot columns with Reverse Cuthill-McKee (RCM) to expose the band.

3. Banded Cholesky of the reordered knot sub-matrix H_kk.

4. Arrowhead Schur complement (knots INTO globals — the correct direction):
     Y    = H_kk⁻¹ H_kc          (n_c banded back-subs)
     y_g  = H_kk⁻¹ g_k           (1 banded back-sub)
     S_c  = H_cc − H_kcᵀ Y       (n_c × n_c dense)
     δx_c = S_c⁻¹(g_c − H_kcᵀ y_g)
     δx_k = y_g − Y δx_c  (un-permute)

5. Compare δx to reference:
     n ≤ DENSE_THRESHOLD → np.linalg.solve(H_dense + damping, g)
     n >  DENSE_THRESHOLD → scipy.sparse.linalg.spsolve(H_sparse + damping, g)

Gate: ‖δx_banded − δx_ref‖ / ‖δx_ref‖ < 1e-8 (damped), < 1e-5 (undamped)

Usage (from analysis/):
    python prototype_banded_solver.py [--dump-dir PATH] [--mu 1e-4] [--verbose]
    python prototype_banded_solver.py --file my_dump_post.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla
from scipy.linalg import solveh_banded

DENSE_THRESHOLD = 5000   # use dense reference below this; sparse above

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_dump(path: Path) -> dict:
    """Load a .npz dump.  Returns dict with scipy CSR J + metadata."""
    data = np.load(str(path), allow_pickle=True)
    block_map = json.loads(str(data['block_map_json'][0]))
    J = sp.csr_matrix(
        (data['jac_values'], data['jac_cols'], data['jac_row_ptr']),
        shape=tuple(data['jac_shape']))
    return {
        'J':         J,
        'r':         data['residuals'],
        'grad':      data['gradient'],
        'block_map': block_map,
        'path':      path,
    }


# ---------------------------------------------------------------------------
# Normal equations (sparse-friendly)
# ---------------------------------------------------------------------------

def build_H_g_sparse(dump: dict):
    """Return (H_sparse, g) without densifying."""
    J = dump['J']
    r = dump['r']
    H = J.T @ J          # stays sparse
    g = -(J.T @ r)
    return H, np.asarray(g).ravel()


def partition_cols(block_map: list) -> tuple[np.ndarray, np.ndarray]:
    """Split tangent columns into (knot_cols, global_cols)."""
    GLOBAL_TYPES = {'bias', 'pitch_delta'}
    knot_cols, global_cols = [], []
    for entry in block_map:
        cols = list(range(entry['col_offset'],
                          entry['col_offset'] + entry['tangent_size']))
        if entry['type_id'] in GLOBAL_TYPES:
            global_cols.extend(cols)
        else:
            knot_cols.extend(cols)
    return np.array(knot_cols, dtype=int), np.array(global_cols, dtype=int)


def extract_blocks_sparse(H_sp, g: np.ndarray,
                          knot_cols: np.ndarray,
                          global_cols: np.ndarray):
    """Extract H_kk (sparse), H_kc (dense), H_cc (dense), g_k, g_c."""
    # scipy CSC for efficient column slicing
    H_csc = H_sp.tocsc()
    H_kk = H_csc[knot_cols, :][:, knot_cols]   # sparse band
    H_kc = np.asarray(H_csc[knot_cols, :][:, global_cols].todense())  # n_k × n_c
    H_cc = np.asarray(H_csc[global_cols, :][:, global_cols].todense())  # n_c × n_c
    g_k  = g[knot_cols]
    g_c  = g[global_cols]
    return H_kk, H_kc, H_cc, g_k, g_c


# ---------------------------------------------------------------------------
# RCM ordering
# ---------------------------------------------------------------------------

def rcm_order_bandwidth(H_kk_sp) -> tuple[np.ndarray, int]:
    """RCM permutation of knot sub-matrix; returns (perm, half_bw)."""
    # Work on binary adjacency
    H_bin = (H_kk_sp.astype(bool)).tocsr()
    perm = csgraph.reverse_cuthill_mckee(H_bin, symmetric_mode=True)

    H_perm = H_kk_sp[np.ix_(perm, perm)].tocsr()
    nz_r, nz_c = H_perm.nonzero()
    half_bw = int(np.max(np.abs(nz_r - nz_c))) if len(nz_r) > 0 else 0
    return perm, half_bw


# ---------------------------------------------------------------------------
# Banded storage for scipy solveh_banded (lower form)
# ---------------------------------------------------------------------------

def sparse_to_banded_lower(H_kk_sp, perm: np.ndarray,
                            half_bw: int) -> np.ndarray:
    """Convert reordered sparse H_kk to LAPACK lower banded storage.

    ab[i, j] = A[j+i, j]  for i=0..half_bw, j=0..n-half_bw-1
    Shape: (half_bw+1, n).
    """
    n = H_kk_sp.shape[0]
    H_perm = H_kk_sp[perm, :][:, perm].toarray()   # dense only after RCM (still banded)
    ab = np.zeros((half_bw + 1, n))
    for i in range(half_bw + 1):
        ab[i, :n - i] = np.diag(H_perm, -i)
    return ab


# ---------------------------------------------------------------------------
# Main banded-Schur solver
# ---------------------------------------------------------------------------

def banded_schur_solve(H_sp, g: np.ndarray, block_map: list,
                       mu: float = 0.0, verbose: bool = False) -> np.ndarray:
    """Banded-Cholesky + arrowhead-Schur solve on sparse H.

    mu: LM diagonal damping (added before factorization)
    """
    n = H_sp.shape[0]
    knot_cols, global_cols = partition_cols(block_map)
    n_k, n_c = len(knot_cols), len(global_cols)

    # Apply damping
    H_damp = H_sp + mu * sp.eye(n, format='csr') if mu > 0 else H_sp

    # Extract blocks
    H_kk, H_kc, H_cc, g_k, g_c = extract_blocks_sparse(
        H_damp, g, knot_cols, global_cols)

    if verbose:
        print(f'    n_k={n_k}, n_c={n_c}, n={n}')

    # RCM reorder the knot block
    perm, half_bw = rcm_order_bandwidth(H_kk)
    perm_inv = np.argsort(perm)

    H_kc_p = H_kc[perm, :]
    g_k_p  = g_k[perm]

    if verbose:
        print(f'    half_bw={half_bw}  (theory max: 5 from quintic pos spline)')

    # Banded Cholesky factorization
    ab = sparse_to_banded_lower(H_kk, perm, half_bw)

    # Arrowhead Schur: reduce knots INTO globals
    Y = np.zeros((n_k, n_c))
    for col in range(n_c):
        Y[:, col] = solveh_banded(ab, H_kc_p[:, col], lower=True)
    y_g = solveh_banded(ab, g_k_p, lower=True)

    S_c   = H_cc - H_kc_p.T @ Y
    rhs_c = g_c - H_kc_p.T @ y_g
    dx_c  = np.linalg.solve(S_c, rhs_c)

    dx_k_p = y_g - Y @ dx_c
    dx_k   = dx_k_p[perm_inv]

    dx = np.empty(n)
    dx[knot_cols]   = dx_k
    dx[global_cols] = dx_c
    return dx


# ---------------------------------------------------------------------------
# Reference solve (sparse-aware)
# ---------------------------------------------------------------------------

def reference_solve(H_sp, g: np.ndarray, mu: float = 0.0) -> np.ndarray:
    """Reference solve: dense for small n, sparse direct for large n."""
    n = H_sp.shape[0]
    H_damp = H_sp + mu * sp.eye(n, format='csr') if mu > 0 else H_sp

    if n <= DENSE_THRESHOLD:
        return np.linalg.solve(H_damp.toarray(), g)
    else:
        # SuperLU / UMFPACK — also direct, should agree to machine precision
        return spla.spsolve(H_damp.tocsr(), g)


# ---------------------------------------------------------------------------
# Sparsity analysis
# ---------------------------------------------------------------------------

def analyse_sparsity(H_sp, block_map: list, label: str):
    """Print band-width and fill-in statistics."""
    knot_cols, global_cols = partition_cols(block_map)
    n_k = len(knot_cols)
    H_kk = H_sp.tocsc()[knot_cols, :][:, knot_cols]
    perm, half_bw = rcm_order_bandwidth(H_kk)

    nnz_Hkk = H_kk.nnz
    banded_nnz = (2 * half_bw + 1) * n_k

    print(f'  [{label}] Sparsity analysis:')
    print(f'    n={H_sp.shape[0]}  n_k={n_k}  n_c={len(global_cols)}')
    print(f'    half_bw={half_bw}   H_kk nnz={nnz_Hkk}   '
          f'banded nnz≈{banded_nnz}   ratio={banded_nnz/max(nnz_Hkk,1):.2f}')
    fill_suite = half_bw ** 2 * n_k
    fill_banded = half_bw * n_k
    print(f'    CHOLMOD fill-in est: O(bw²n)≈{fill_suite:,}   '
          f'banded fill-in: O(bw·n)≈{fill_banded:,}')


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------

def validate_one(dump: dict, mu_frac: float, verbose: bool) -> bool:
    label = dump['path'].name
    bmap  = dump['block_map']
    H_sp, g = build_H_g_sparse(dump)
    n = H_sp.shape[0]

    global_types = {e['type_id'] for e in bmap
                    if e['type_id'] in ('bias', 'pitch_delta')}
    n_c = sum(e['tangent_size'] for e in bmap
              if e['type_id'] in ('bias', 'pitch_delta'))
    n_k = n - n_c

    print(f'\n  {label}')
    print(f'    n={n}  n_k={n_k}  n_c={n_c}  globals={sorted(global_types)}')
    analyse_sparsity(H_sp, bmap, label)

    all_ok = True
    max_diag = float(H_sp.diagonal().max())

    for tag, mu in [('undamped', 0.0),
                    (f'damped(mu={mu_frac:.0e})', mu_frac * max_diag)]:
        if mu_frac == 0.0 and 'damped' in tag:
            continue

        try:
            dx_ref = reference_solve(H_sp, g, mu)
        except Exception as e:
            print(f'    [{tag}] reference solve failed: {e}  (skip)')
            continue

        try:
            dx_b = banded_schur_solve(H_sp, g, bmap, mu=mu, verbose=verbose)
        except Exception as e:
            print(f'    [{tag}] banded solve FAILED: {e}')
            all_ok = False
            continue

        ref_norm  = float(np.linalg.norm(dx_ref))
        rel_err   = float(np.linalg.norm(dx_b - dx_ref) / max(ref_norm, 1e-15))

        # Gate: cond ≈ 5.5e10 undamped → floor ≈ 1e-6; damped: 1e-8
        gate = 1e-5 if 'undamped' in tag else 1e-8
        status = 'PASS ✓' if rel_err < gate else 'FAIL ✗'
        ref_method = 'dense' if n <= DENSE_THRESHOLD else 'spsolve'
        print(f'    [{tag}] rel_err={rel_err:.2e}  '
              f'gate={gate:.0e}  ref={ref_method}  {status}')
        if rel_err >= gate:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dump-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data' / 'linear_system_dumps')
    parser.add_argument('--mu', type=float, default=1e-4,
                        help='LM damping fraction × max_diag(H)  (0 = undamped only)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--file', type=str, default=None,
                        help='Test a single file basename')
    args = parser.parse_args()

    dump_dir = args.dump_dir
    if not dump_dir.exists():
        print(f'Dump dir not found: {dump_dir}')
        print('Run dump_linear_system.py first, or pass --file with a direct path.')
        sys.exit(1)

    files = [dump_dir / args.file] if args.file else sorted(dump_dir.glob('*.npz'))
    if not files:
        print(f'No .npz files in {dump_dir}')
        sys.exit(1)

    print(f'Testing {len(files)} dump(s) in {dump_dir}')
    print('=' * 70)

    all_ok = True
    for path in files:
        try:
            dump = load_dump(path)
        except Exception as e:
            print(f'\n  ERROR loading {path.name}: {e}')
            all_ok = False
            continue
        ok = validate_one(dump, args.mu, args.verbose)
        all_ok = all_ok and ok

    print('\n' + '=' * 70)
    if all_ok:
        print('SESSION 1 GATE: ALL PASS ✓  — math is correct, proceed to Session 2')
    else:
        print('SESSION 1 GATE: FAILURES DETECTED ✗  — fix before Session 2')
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Inline test: call from another script with a SolverResult.SystemDump
# ---------------------------------------------------------------------------

def validate_from_dump_obj(dump_obj, bag_label: str = '', mu_frac: float = 1e-4) -> bool:
    """Test a SystemDump object directly (no file I/O).

    Useful for calling from dump_linear_system.py without saving to disk.
    Returns True if the gate passes.
    """
    if not dump_obj.valid:
        print(f'  [{bag_label}] dump not valid — skip')
        return True   # not a failure

    import json
    bmap_list = [
        {'type_id': e.type_id, 'index': e.index,
         'col_offset': e.col_offset, 'tangent_size': e.tangent_size}
        for e in dump_obj.block_map
    ]
    J = sp.csr_matrix(
        (np.array(dump_obj.jac_values),
         np.array(dump_obj.jac_cols),
         np.array(dump_obj.jac_row_ptr)),
        shape=(dump_obj.jac_num_rows, dump_obj.jac_num_cols))
    r = np.array(dump_obj.residuals)

    fake_dump = {
        'J': J, 'r': r,
        'grad': np.array(dump_obj.gradient),
        'block_map': bmap_list,
        'path': type('P', (), {'name': bag_label})(),
    }
    return validate_one(fake_dump, mu_frac, verbose=False)


if __name__ == '__main__':
    sys.exit(main())
