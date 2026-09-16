"""
Minimal test script for ASAN: runs 1 SW window with BANDED_SCHUR without
importing matplotlib (which conflicts with ASAN's __cxa_throw interceptor).
"""
import sys
import os
import pathlib

# Suppress matplotlib before any import can drag it in
sys.modules['matplotlib'] = type(sys)('matplotlib')
sys.modules['matplotlib.pyplot'] = type(sys)('matplotlib.pyplot')

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'lib'))

# Now safe to import the solver modules
import config_loader
import validate_live_solver as vls

BAG  = 'slow_racing_best_velocity'
argv = [BAG, '--mocap-yaw', '--cpp', '--sliding-window', '--no-plot',
        '--set', 'use_banded_schur=1']

print("asan_minimal_test: calling run_sw_windows with max_windows=1", flush=True)
results = vls.run_sw_windows(argv, max_windows=1)
print(f"asan_minimal_test: done, got {len(results)} window(s)", flush=True)
