# `analysis/` — Directory Structure

## Layout

```
analysis/
│
├── config/                          # Single source of truth — all hardcoded values
│   ├── bags.yaml                    # bag aliases → paths, flipped set, timing [start_offset, duration]
│   ├── extrinsics.yaml              # rotation [180,30,0] deg, translation [0.08,0.02,-0.01] m
│   └── solver.yaml                  # LM hyperparameters, B-spline config, loss weights
│
├── lib/                             # Shared libraries — imported by everything
│   ├── radar_velocity_utils.py      # Forward model, WLS ego-velocity solver, transforms
│   ├── bspline_utils.py             # Uniform B-splines, derivatives, min-snap regularization
│   ├── cumulative_so3_bspline.py    # Cumulative SO(3) B-spline on Lie groups
│   ├── imu_preintegration.py        # Forster TRO-2017 on-manifold preintegration
│   └── rosbag_loader/               # ROS bag loading (load_bag_topics + structures)
│
├── codegen/                         # SymForce code generation — run-once
│   ├── derive_jacobians_symforce.py # Run to regenerate generated_jacobians.py
│   └── generated_jacobians.py       # Auto-generated — DO NOT EDIT BY HAND
│
├── viz/                             # Visualization tools
│   ├── plot_radar_map.py            # Interactive 3D radar point cloud (Open3D)
│   └── plot_extrinsics.py           # Coordinate frame visualizer
│
├── diagnostics/                     # Investigative scripts (diagnose_*, check_*)
│
├── eval_results/                    # JSON output from eval_bags.py runs
│
├── validate_live_solver.py          # Live RIO: MoCap-free init + LM solver (main entry point)
├── validate_nonlinear_solver.py     # Batch solver: full LM with MoCap init (shared solver core)
├── validate_linear_solver.py        # Phase 2: linear LS baseline
├── validate_forward_model.py        # Phase 1: forward model validation
├── validate_physics.py              # Physics/kinematics check (no optimization)
├── eval_bags.py                     # Multi-bag evaluation harness → eval_results/*.json
│
├── config_loader.py                 # Loads all config/ YAML files into a dict-of-dicts
├── notebooks/                       # Jupyter notebooks
└── requirements.txt
```

## Entry Points

Run everything from `analysis/` using the repo venv:

```bash
cd analysis/

# Live RIO solver (MoCap-free init, MoCap used only for eval)
../.venv/bin/python3 validate_live_solver.py slow_racing_best_velocity --mocap-yaw
../.venv/bin/python3 validate_live_solver.py fast_racing_best_velocity --mocap-yaw

# Multi-bag eval → eval_results/<label>_<timestamp>.json
../.venv/bin/python3 eval_bags.py --label baseline --flags "--mocap-yaw"

# Batch solver (MoCap-initialized, Phase 3)
../.venv/bin/python3 validate_nonlinear_solver.py circle_fwd

# Phase 1 & 2
../.venv/bin/python3 validate_physics.py original
../.venv/bin/python3 validate_linear_solver.py circle

# Diagnostics
../.venv/bin/python3 diagnostics/diagnose_doppler.py circle
../.venv/bin/python3 diagnostics/diagnose_gyro.py circle

# Visualization
../.venv/bin/python3 viz/plot_radar_map.py circle_fwd
../.venv/bin/python3 viz/plot_extrinsics.py

# Regenerate Jacobians (requires SymForce — root venv, not analysis venv)
cd ..
source .venv/bin/activate
python analysis/codegen/derive_jacobians_symforce.py
```

## Config Files

| File | Content |
|------|---------|
| `config/bags.yaml` | `bags: {alias: path}`, `flipped: [...]`, `timing: {bag: [start_offset, duration]}` |
| `config/extrinsics.yaml` | `rotation_euler_deg`, `translation_body_m`, time offsets |
| `config/solver.yaml` | `huber_delta`, `lambda_accel`, `bspline_degree`, all LM hyperparameters |

**Canonical extrinsics** (confirmed correct):
- Rotation: `[roll=180°, pitch=30°, yaw=0°]`
- Translation: `[0.08, +0.02, -0.01]` m in body frame

## Documentation

| File | Content |
|------|---------|
| `analysis/LIVE_RIO_SUMMARY.md` | Current state: benchmarks, architecture, known limitations |
| `analysis/IMPROVEMENTS.md` | Roadmap: done items + remaining (GNC, C++ migration) |
| `analysis/FINDINGS_PREINTEGRATION.md` | Detailed investigation: preintegration, pre-unwrapping, root cause analysis |
| `documentation/FINDINGS.md` | Foundational findings: extrinsics, body frame, IMU offset, Doppler sign |
| `documentation/Forward Model.md` | Math: forward measurement model |
| `documentation/Backward Model.md` | Math: estimation / optimization formulation |

## Import Convention

Root scripts add `lib/` to `sys.path` and use bare imports:
```python
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import rotation_matrix_from_euler
```

Scripts in subdirectories add both `analysis/` and `analysis/lib/`:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
```
