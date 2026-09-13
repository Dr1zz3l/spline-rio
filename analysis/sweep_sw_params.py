"""
sweep_sw_params.py — Sweep max_iterations and/or window_duration for the C++ sliding window solver.

Characterises the accuracy–vs–speed trade-off curve to find the elbow before committing to
structural speedups (iSAM2, analytic Jacobians).  Extends eval_bags.py with per-window timing
extraction and a sweeping outer loop.

Usage:
    python sweep_sw_params.py --mode iter    # sweep max_iterations (window=3.0s)
    python sweep_sw_params.py --mode window  # sweep window_duration (max_iter=40)
    python sweep_sw_params.py --mode grid    # full 2D sweep (~1–3 h)

    # Quick smoke test (2 data points, 1 bag):
    python sweep_sw_params.py --mode iter --iter-vals 8 40 --bags slow_racing_best_velocity
"""

import sys
import re
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_bags import parse_rmse_from_output, RESULTS_DIR, DEFAULT_BAGS, PYTHON, SOLVER_SCRIPT

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------
ITER_GRID   = [8, 12, 16, 20, 28, 40]
WINDOW_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]

BASE_FLAGS = '--cpp --sliding-window --mocap-yaw --no-plot'

# Reference numbers from CLAUDE.md (max_iter=40, window=3.0s)
BASELINE = {
    'slow_racing_best_velocity': {'pos': 0.393, 'vel': 0.391, 'ori': 2.21, 'dt': 2.1},
    'fast_racing_best_velocity': {'pos': 0.877, 'vel': 0.478, 'ori': 4.16, 'dt': 1.7},
}

# ---------------------------------------------------------------------------
# Per-window timing parser
# ---------------------------------------------------------------------------
_SW_LINE_RE = re.compile(r'\[sw\s+\d+\].*?iter=(\d+).*?dt=([\d.]+)s')


def parse_window_timing(stdout: str) -> dict:
    """Extract per-window stats from [sw NNN] diagnostic lines."""
    iters, dts = [], []
    for m in _SW_LINE_RE.finditer(stdout):
        iters.append(int(m.group(1)))
        dts.append(float(m.group(2)))
    if not dts:
        return {}
    dts_s  = sorted(dts)
    iters_s = sorted(iters)
    n = len(dts_s)
    return {
        'median_dt_s':  dts_s[n // 2],
        'p90_dt_s':     dts_s[min(n - 1, int(n * 0.9))],
        'median_iter':  iters_s[n // 2],
        'n_windows':    n,
    }


# ---------------------------------------------------------------------------
# Runner (own subprocess so we keep full stdout for timing)
# ---------------------------------------------------------------------------
def run_point(bag: str, extra_flags: str, timeout: int = 1200) -> dict:
    """Run validate_live_solver.py for (bag, extra_flags), return metrics dict."""
    cmd = [PYTHON, str(SOLVER_SCRIPT), bag] + extra_flags.split()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(HERE))
        wall = time.time() - t0
        stdout = proc.stdout
        if proc.returncode != 0:
            tail = (proc.stderr or '')[-600:]
            print(f"    [FAILED rc={proc.returncode}]\n{tail}")
            return {'returncode': proc.returncode, 'metrics': {}, 'wall_time_s': round(wall, 1)}
        metrics = parse_rmse_from_output(stdout)
        metrics.update(parse_window_timing(stdout))
        metrics['wall_time_s'] = round(wall, 1)
        return {'returncode': 0, 'metrics': metrics}
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT after {timeout}s]")
        return {'returncode': -1, 'metrics': {}, 'wall_time_s': timeout}


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------
def sweep_1d(param: str, values: list, bags: list, timeout: int) -> list:
    rows = []
    for v in values:
        flags = f'{BASE_FLAGS} --set {param}={v}'
        row = {'param': param, 'value': v, 'bags': {}}
        for bag in bags:
            print(f"  [{param}={v}]  {bag} ...", flush=True)
            r = run_point(bag, flags, timeout=timeout)
            m = r.get('metrics', {})
            row['bags'][bag] = r
            pos = m.get('pos_rmse_m', float('nan'))
            vel = m.get('vel_rmse_mps', float('nan'))
            ori = m.get('ori_rmse_deg', float('nan'))
            dt  = m.get('median_dt_s', float('nan'))
            print(f"    pos={pos:.3f}m  vel={vel:.3f}m/s  ori={ori:.2f}°  dt/win={dt:.2f}s")
        rows.append(row)
    return rows


def sweep_grid(iter_vals: list, window_vals: list, bags: list, timeout: int) -> list:
    rows = []
    for w in window_vals:
        for n in iter_vals:
            flags = f'{BASE_FLAGS} --set window_duration={w} --set max_iterations={n}'
            row = {'window_duration': w, 'max_iterations': n, 'bags': {}}
            for bag in bags:
                print(f"  [window={w}s, iter={n}]  {bag} ...", flush=True)
                r = run_point(bag, flags, timeout=timeout)
                m = r.get('metrics', {})
                row['bags'][bag] = r
                pos = m.get('pos_rmse_m', float('nan'))
                dt  = m.get('median_dt_s', float('nan'))
                print(f"    pos={pos:.3f}m  dt/win={dt:.2f}s")
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------
_BAG_SHORT = {
    'slow_racing_best_velocity': 'slow',
    'fast_racing_best_velocity': 'fast',
}
_COL = 8


def _cell(metrics: dict, key: str, prec: int) -> str:
    v = metrics.get(key)
    return f'{v:.{prec}f}' if v is not None else '  N/A '


def print_table(mode: str, rows: list, bags: list):
    print(f'\n{"="*110}')
    print(f'Sweep: {mode}')

    # Column headers
    param_col = 'max_iter' if mode == 'iter' else ('win(s)' if mode == 'window' else 'w / iter')
    hdr = f'  {param_col:<11}'
    for bag in bags:
        b = _BAG_SHORT.get(bag, bag[:6])
        hdr += f'  {"pos(m)":>{_COL}} {"vel(m/s)":>{_COL}} {"ori(°)":>{_COL}} {"dt/w(s)":>{_COL}}'
        hdr += f'  # [{b}]'
    print(hdr)
    print(f'  {"-"*11}' + f'  {"-"*_COL} {"-"*_COL} {"-"*_COL} {"-"*_COL}' * len(bags))

    for row in rows:
        if mode == 'iter':
            label = str(row['value'])
        elif mode == 'window':
            label = str(row['value'])
        else:
            label = f'{row["window_duration"]}s/{row["max_iterations"]}'
        line = f'  {label:<11}'
        for bag in bags:
            m = row['bags'].get(bag, {}).get('metrics', {})
            line += (f'  {_cell(m, "pos_rmse_m",   3):>{_COL}}'
                     f' {_cell(m, "vel_rmse_mps", 3):>{_COL}}'
                     f' {_cell(m, "ori_rmse_deg", 2):>{_COL}}'
                     f' {_cell(m, "median_dt_s",  2):>{_COL}}')
        print(line)

    # Baseline reference row
    line = f'  {"baseline*":<11}'
    for bag in bags:
        b = BASELINE.get(bag, {})
        line += (f'  {b.get("pos", 0):.3f}   {b.get("vel", 0):.3f}   '
                 f'{b.get("ori", 0):.2f}   {b.get("dt", 0):.2f}')
        # pad to column width
    print(line)
    print(f'  * CLAUDE.md reference: max_iter=40, window=3.0s')
    print(f'{"="*110}\n')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Sweep max_iterations / window_duration for C++ sliding window solver')
    parser.add_argument('--mode', choices=['iter', 'window', 'grid'], default='iter')
    parser.add_argument('--bags', nargs='+', default=DEFAULT_BAGS)
    parser.add_argument('--iter-vals',   nargs='+', type=int,   default=None,
                        help=f'Override iteration grid (default: {ITER_GRID})')
    parser.add_argument('--window-vals', nargs='+', type=float, default=None,
                        help=f'Override window grid (default: {WINDOW_GRID})')
    parser.add_argument('--timeout', type=int, default=1200, help='Per-bag timeout in seconds')
    args = parser.parse_args()

    iter_vals   = args.iter_vals   or ITER_GRID
    window_vals = args.window_vals or WINDOW_GRID

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file  = RESULTS_DIR / f'sweep_{args.mode}_{timestamp}.json'

    print(f'\n{"="*60}')
    print(f'SW param sweep  mode={args.mode}  bags={args.bags}')
    if args.mode in ('iter', 'grid'):
        print(f'  max_iterations: {iter_vals}')
    if args.mode in ('window', 'grid'):
        print(f'  window_duration: {window_vals}')
    print(f'  timeout: {args.timeout}s per bag')
    print(f'{"="*60}\n')

    t_total = time.time()

    if args.mode == 'iter':
        rows = sweep_1d('max_iterations', iter_vals, args.bags, args.timeout)
    elif args.mode == 'window':
        rows = sweep_1d('window_duration', window_vals, args.bags, args.timeout)
    else:
        rows = sweep_grid(iter_vals, window_vals, args.bags, args.timeout)

    print_table(args.mode, rows, args.bags)
    print(f'Total sweep time: {time.time() - t_total:.0f}s')

    payload = {
        'mode':         args.mode,
        'timestamp':    timestamp,
        'bags':         args.bags,
        'iter_vals':    iter_vals   if args.mode in ('iter',   'grid') else None,
        'window_vals':  window_vals if args.mode in ('window', 'grid') else None,
        'rows':         rows,
    }
    with open(out_file, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Results saved to: {out_file}\n')


if __name__ == '__main__':
    main()
