"""Import + construct smoke test for the pixi build (pixi run check).

Usage: python pixi_check.py [LIBDIR]   (default: rio_solver_cpp/build_pixi/lib)
"""
import sys
from pathlib import Path

lib = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "build_pixi" / "lib"
sys.path.insert(0, str(lib))

import numpy
import rospy  # noqa: F401  (RoboStack ROS1)
import rio_solver
try:
    import rio_gp   # optional: absent in the public spline-only tree
except ImportError:
    rio_gp = None

cfg = rio_solver.SolverConfig()
rio_solver.SlidingWindowSolver(cfg, rio_solver.ExtrinsicConfig())
if rio_gp is not None:
    rio_gp.GpConfig()
print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("rospy", rospy.__file__)
print("rio_solver", rio_solver.__file__)
print("rio_gp", rio_gp.__file__ if rio_gp else "not built (spline-only tree)")
print("check OK")
