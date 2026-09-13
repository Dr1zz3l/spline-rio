"""
Multi-bag evaluation script for validate_live_solver.py.

Runs the live solver on a set of bags, parses RMSE metrics from stdout,
and saves structured results to eval_results/<label>_<timestamp>.json.

Usage:
    python eval_bags.py [--bags bag1 bag2 ...] [--label name] [--flags "extra flags"]

Examples:
    # Baseline with mocap-yaw + converged biases
    python eval_bags.py --label baseline --flags "--mocap-yaw --bias converged"

    # With preintegration
    python eval_bags.py --label preintegration --flags "--mocap-yaw --bias converged --preintegrate"

    # Sensor-only (no MoCap)
    python eval_bags.py --label sensor_only --flags ""

Default bags: slow_racing_best_velocity fast_racing_best_velocity
"""

import sys
import os
import re
import json
import subprocess
import argparse
import statistics
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / 'eval_results'
SOLVER_SCRIPT = HERE / 'validate_live_solver.py'

# Default bags to evaluate
DEFAULT_BAGS = [
    'slow_racing_best_velocity',
    'fast_racing_best_velocity',
]

# Python executable (use venv if available, else sys.executable)
_VENV_PYTHON = Path(sys.executable).parent / 'python3'
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def parse_rmse_from_output(stdout: str) -> dict:
    """Parse RMSE metrics from validate_live_solver.py stdout.

    The solver prints a two-column settled/live table (see the RESULTS block in
    validate_live_solver.py).  The LIVE column is the deployment metric and the
    one every gate battery is judged on, so both are captured and `*_live_*`
    keys are the ones to compare.

    (Until 2026-08-06 these patterns matched `Position RMSE (aligned):` and
    friends, a format the solver stopped printing long ago, so this returned
    N/A for position, velocity and orientation without complaining.  That is
    why every recorded battery was run by shell loop and grep instead.)
    """
    metrics = {}

    # settled-only rows
    single = {
        'angvel_rmse_rads': r'Angular vel RMSE \(rad/s\)\s+([\d.]+)',
        'accel_rmse_mps2':  r'Acceleration RMSE \(m/s.\)\s+([\d.]+)',
        'pos_horiz_rmse_m': r'horizontal \(xy\) RMSE \(m\)\s+([\d.]+)',
        'pos_vert_rmse_m':  r'vertical \(z\) RMSE \(m\)\s+([\d.]+)',
    }
    for key, pat in single.items():
        m = re.search(pat, stdout)
        if m:
            metrics[key] = float(m.group(1))

    # settled | live pairs.  The live column is absent in batch mode.
    paired = {
        'pos_rmse_m':   r'Position RMSE \(m\)\s+([\d.]+)(?:\s+([\d.]+))?',
        'vel_rmse_mps': r'Velocity RMSE \(m/s\)\s+([\d.]+)(?:\s+([\d.]+))?',
        'ori_rmse_deg': r'Orientation RMSE \(deg\)\s+([\d.]+)(?:\s+([\d.]+))?',
        'pos_drift_pct': r'Position drift \(%\)\s+([\d.]+)%(?:\s+([\d.]+)%)?',
    }
    for key, pat in paired.items():
        m = re.search(pat, stdout)
        if not m:
            continue
        metrics[key] = float(m.group(1))
        if m.group(2) is not None:
            metrics[key.replace('_', '_live_', 1)] = float(m.group(2))

    # whole-trajectory alignment (ICINS runs use --whole-traj-align)
    m = re.search(r'\[whole-traj align\] pos RMSE ([\d.]+) m \(drift ([\d.]+)%\)',
                  stdout)
    if m:
        metrics['whole_traj_ate_m'] = float(m.group(1))
        metrics['whole_traj_drift_pct'] = float(m.group(2))

    # per-point weighting coverage, when --bearing-weight is active: a run with
    # 0% coverage is a silent no-op and must not be read as a null result
    m = re.search(r'\[--bearing-weight (\w+)\].*?\((\d+\.\d+)%\), '
                  r'mean clean-return weight ([\d.]+)', stdout)
    if m:
        metrics['bearing_weight'] = m.group(1)
        metrics['bearing_weight_coverage_pct'] = float(m.group(2))
        metrics['bearing_weight_mean'] = float(m.group(3))

    # median per-window solve time, from the [sw NNN] ... dt=X.XXs lines
    dts = [float(x) for x in re.findall(r'\bdt=([\d.]+)s', stdout)]
    if dts:
        metrics['t_win_median_s'] = round(statistics.median(dts), 3)
        metrics['n_windows'] = len(dts)

    # Per-axis orientation RMSE
    m = re.search(r'Per-axis ori RMSE:\s*roll=([\d.]+)\s+pitch=([\d.]+)\s+yaw=([\d.]+)', stdout)
    if m:
        metrics['ori_rmse_roll_deg']  = float(m.group(1))
        metrics['ori_rmse_pitch_deg'] = float(m.group(2))
        metrics['ori_rmse_yaw_deg']   = float(m.group(3))

    # Solver iterations and timing
    m = re.search(r'Iterations?:\s*(\d+)', stdout, re.IGNORECASE)
    if m:
        metrics['lm_iterations'] = int(m.group(1))

    m = re.search(r'Total time:\s*([\d.]+)\s*s', stdout, re.IGNORECASE)
    if m:
        metrics['total_time_s'] = float(m.group(1))

    # IMU residual count (per-sample path)
    m = re.search(r'IMU after downsampling.*?:\s*(\d+)', stdout)
    if m:
        metrics['imu_samples_used'] = int(m.group(1))

    # Preintegrated factor count (preintegration path)
    m = re.search(r'Preintegrated IMU factors:\s*(\d+)', stdout)
    if m:
        metrics['preintegrated_factors'] = int(m.group(1))

    # Radar frame count
    m = re.search(r'Radar frames.*?:\s*(\d+)', stdout)
    if m:
        metrics['radar_frames'] = int(m.group(1))

    return metrics


def run_bag(bag_key: str, extra_flags: str, timeout: int = 600) -> dict:
    """Run validate_live_solver.py on one bag, return parsed result."""
    cmd = [PYTHON, str(SOLVER_SCRIPT), bag_key, '--no-plot'] + extra_flags.split()
    print(f"\n{'='*70}")
    print(f"Running: {bag_key}  flags: {extra_flags or '(none)'}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    t_start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(HERE),
        )
        elapsed = time.time() - t_start
        stdout = proc.stdout
        stderr = proc.stderr

        # Print live output summary (last 60 lines to not flood terminal)
        output_lines = stdout.splitlines()
        if len(output_lines) > 60:
            print(f"  [... {len(output_lines)-60} lines omitted ...]")
            print('\n'.join(output_lines[-60:]))
        else:
            print(stdout)

        if proc.returncode != 0:
            print(f"\n[STDERR]\n{stderr[-2000:] if len(stderr) > 2000 else stderr}")

        metrics = parse_rmse_from_output(stdout)
        metrics['wall_time_s'] = round(elapsed, 1)

        return {
            'bag': bag_key,
            'flags': extra_flags,
            'returncode': proc.returncode,
            'metrics': metrics,
            'stdout_tail': '\n'.join(output_lines[-100:]),
            'stderr_tail': stderr[-1000:] if stderr else '',
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        print(f"[TIMEOUT] {bag_key} exceeded {timeout}s")
        return {
            'bag': bag_key,
            'flags': extra_flags,
            'returncode': -1,
            'error': f'timeout after {timeout}s',
            'metrics': {},
            'stdout_tail': '',
            'stderr_tail': '',
        }
    except Exception as e:
        return {
            'bag': bag_key,
            'flags': extra_flags,
            'returncode': -1,
            'error': str(e),
            'metrics': {},
            'stdout_tail': '',
            'stderr_tail': '',
        }


def print_summary(results: list):
    """Compact table in the LIVE-edge metrics, which is what a gate battery is
    judged on (falls back to the settled column in batch mode)."""
    print(f"\n{'='*94}")
    print(f"{'BAG':<34} {'vel':>7} {'ori°':>7} {'drift%':>8} {'pos m':>8} "
          f"{'t_win':>7} {'meanW':>7} {'wall':>7}")
    print(f"{'-'*94}")
    for r in results:
        m = r['metrics']

        def col(key, fmt='.3f'):
            v = m.get(key.replace('_', '_live_', 1), m.get(key))
            return f'{v:{fmt}}' if v is not None else '  N/A'

        flags = []
        if r['returncode'] != 0:
            flags.append('FAILED')
        # a 0%-coverage bearing-weight run is a silent no-op, not a null result
        if 0.0 <= m.get('bearing_weight_coverage_pct', 100.0) < 99.0:
            flags.append(f"COVERAGE {m['bearing_weight_coverage_pct']:.0f}%")
        print(f"  {r['bag']:<32} {col('vel_rmse_mps'):>7} "
              f"{col('ori_rmse_deg'):>7} {col('pos_drift_pct', '.2f'):>8} "
              f"{col('pos_rmse_m', '.4f'):>8} "
              f"{m.get('t_win_median_s', float('nan')):>7.2f} "
              f"{m.get('bearing_weight_mean', float('nan')):>7.3f} "
              f"{m.get('wall_time_s', 0):>6.0f}s"
              + ('  [' + ', '.join(flags) + ']' if flags else ''))
    print(f"{'='*94}\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-bag RIO evaluation')
    parser.add_argument('--bags', nargs='+', default=DEFAULT_BAGS,
                        help='Bag keys to evaluate')
    parser.add_argument('--label', default='run',
                        help='Label for the output file (e.g. baseline, preintegration)')
    parser.add_argument('--flags', default='',
                        help='Extra flags to pass to validate_live_solver.py')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Per-bag timeout in seconds')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f"{args.label}_{timestamp}.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    run_meta = {
        'label': args.label,
        'timestamp': timestamp,
        'flags': args.flags,
        'bags': args.bags,
        'python': PYTHON,
    }

    print(f"\nEval: {args.label}")
    print(f"Bags: {args.bags}")
    print(f"Flags: '{args.flags}'")
    print(f"Output: {out_file}")

    results = []
    for bag in args.bags:
        result = run_bag(bag, args.flags, timeout=args.timeout)
        results.append(result)

    print_summary(results)

    output = {
        'meta': run_meta,
        'results': results,
    }

    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {out_file}\n")
    return out_file


if __name__ == '__main__':
    main()
