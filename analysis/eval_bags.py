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
    """Parse RMSE metrics from validate_live_solver.py stdout."""
    metrics = {}

    patterns = {
        'pos_rmse_m':       r'Position RMSE \(aligned\):\s*([\d.]+)\s*m',
        'pos_max_m':        r'Position RMSE \(aligned\):.*max:\s*([\d.]+)\s*m',
        'vel_rmse_mps':     r'Velocity RMSE:\s*([\d.]+)\s*m/s',
        'angvel_rmse_rads': r'Angular vel RMSE:\s*([\d.]+)\s*rad/s',
        'accel_rmse_mps2':  r'Acceleration RMSE:\s*([\d.]+)\s*m/s',
        'ori_rmse_deg':     r'Orientation RMSE:\s*([\d.]+)\s*deg\s+\(max:',
        'ori_max_deg':      r'Orientation RMSE:.*\(max:\s*([\d.]+)\s*deg\)',
    }

    for key, pat in patterns.items():
        m = re.search(pat, stdout)
        if m:
            metrics[key] = float(m.group(1))

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
    """Print a compact RMSE table."""
    print(f"\n{'='*80}")
    print(f"{'BAG':<35} {'Pos(m)':>8} {'Vel(m/s)':>10} {'Ori(°)':>8} {'Time(s)':>9}")
    print(f"{'-'*80}")
    for r in results:
        m = r['metrics']
        pos  = f"{m['pos_rmse_m']:.4f}"  if 'pos_rmse_m'     in m else '  N/A  '
        vel  = f"{m['vel_rmse_mps']:.4f}" if 'vel_rmse_mps'   in m else '  N/A  '
        ori  = f"{m['ori_rmse_deg']:.3f}" if 'ori_rmse_deg'   in m else '  N/A  '
        wall = f"{m.get('wall_time_s', 0):.0f}"
        status = '' if r['returncode'] == 0 else ' [FAILED]'
        print(f"  {r['bag']:<33} {pos:>8} {vel:>10} {ori:>8} {wall:>8}s{status}")
    print(f"{'='*80}\n")


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
