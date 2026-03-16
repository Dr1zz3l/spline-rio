# `analysis/` — Directory Structure

## Layout

```
analysis/
│
├── config/                          # Single source of truth — all hardcoded values
│   ├── bags.yaml                    # bag aliases → paths, flipped bag set
│   ├── extrinsics.yaml              # rotation [180,30,0] deg, translation [0,0.02,-0.01] m
│   └── solver.yaml                  # LM hyperparameters, B-spline config
│
├── lib/                             # Shared libraries — imported by everything
│   ├── radar_velocity_utils.py      # Forward model, WLS ego-velocity, transforms
│   ├── bspline_utils.py             # Uniform B-splines, derivatives, min-snap reg
│   └── rosbag_loader/               # ROS bag loading (load_bag_topics + structures)
│
├── codegen/                         # SymForce code generation — run-once
│   ├── derive_jacobians_symforce.py # Run to regenerate generated_jacobians.py
│   └── generated_jacobians.py       # Auto-generated — DO NOT EDIT BY HAND
│
├── viz/                             # Visualization tools
│   ├── plot_radar_map.py            # Interactive 3D radar point cloud (Open3D)
│   └── plot_extrinsics.py           # Coordinate frame visualizer (confirms extrinsics)
│
├── diagnostics/                     # Investigative one-off scripts (check_*, diagnose_*, etc.)
│
├── tests/                           # Unit tests (runnable with pytest)
│   ├── test_bspline*.py             # B-spline correctness tests
│   ├── test_generated.py            # Smoke-test SymForce-generated functions
│   └── test_symforce_extrinsic.py   # Extrinsic perturbation test
│
├── validate_nonlinear_solver.py     # Phase 3: full LM solver (main entry point)
├── validate_linear_solver.py        # Phase 2: linear LS baseline
├── validate_forward_model.py        # Phase 1: forward model validation
├── validate_physics.py              # Physics/kinematics check (no optimization)
│
├── config_loader.py                 # Loads all config/ YAML files into a dict-of-dicts
├── notebooks/                       # Jupyter notebooks
├── notes.md                         # Running analysis notes
└── requirements.txt
```

## Entry Points

Run everything from `analysis/`:

```bash
# Main solver
python validate_nonlinear_solver.py circle_fwd

# Phase 1 & 2
python validate_physics.py original
python validate_linear_solver.py circle

# Diagnostics
python diagnostics/diagnose_doppler.py circle
python diagnostics/diagnose_gyro.py circle

# Visualization
python viz/plot_radar_map.py circle_fwd
python viz/plot_extrinsics.py              # show extrinsics
python viz/plot_extrinsics.py circle_fwd   # with yaw-flip applied

# Regenerate Jacobians (requires SymForce in Docker)
python codegen/derive_jacobians_symforce.py
```

## Config Files

| File | Content |
|------|---------|
| `config/bags.yaml` | `bags: {alias: path}`, `flipped: [...]`, `timing: {bag: [start, dur]}` |
| `config/extrinsics.yaml` | `rotation_euler_deg`, `translation_body_m`, time offsets |
| `config/solver.yaml` | `huber_delta`, `lambda_accel`, `bspline_degree`, etc. |

**Canonical extrinsics** (confirmed correct):
- Rotation: `[roll=180°, pitch=30°, yaw=0°]`
- Translation: `[0.08, +0.02, -0.01]` m in body frame (8 cm forward, 2 cm left, 1 cm down)

## Import Convention

Root scripts add `lib/` to sys.path and use bare imports:
```python
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from rosbag_loader.loader import load_bag_topics
from radar_velocity_utils import rotation_matrix_from_euler
```

Scripts in subdirectories add both `analysis/` and `analysis/lib/` to sys.path:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
```
