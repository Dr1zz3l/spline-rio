#!/usr/bin/env bash
# Build the rio_gp + rio_solver pybind modules inside the pixi environment
# (RoboStack ROS1 Noetic + conda-forge toolchain), for the env's python.
#
#   pixi run build-rio            (from the repo root)
#
# Output: rio_solver_cpp/build_pixi/{rio_gp,rio_solver}.cpython-*.so, copied to
# rio_solver_cpp/build_pixi/lib/.  Never touches build_release/ (the host venv
# build) and never runs `install` (its targets write into analysis/).
#
# The host-built vendored Ceres (ceres_vendored_install/) is skipped via
# RIO_USE_VENDORED_CERES=OFF: the env's conda-forge Ceres is used instead.
# NOTE: compiled with the BUILD machine's CPU flags (CMake RIO_ARCH_FLAGS:
# -march=native on x86, -mcpu=native on the Jetson / aarch64), so the .so files
# only run on CPUs like it: build on the deployment machine (the Jetson).
# Header-only math deps are vendored in rio_solver_cpp/third_party/.
set -euo pipefail
: "${CONDA_PREFIX:?run this through pixi (pixi run build-rio)}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC="$ROOT/rio_solver_cpp"
BUILD="$SRC/build_pixi"
PY="$CONDA_PREFIX/bin/python"
# compile jobs: the Ceres/Eigen autodiff units take a few GB each; on the
# Jetson (8-16 GB shared with the GPU) keep it low
if [ "$(uname -m)" = aarch64 ]; then JOBS=${RIO_BUILD_JOBS:-2}; else JOBS=${RIO_BUILD_JOBS:-8}; fi

# shellcheck disable=SC2086  # CMAKE_ARGS (conda compiler activation) is a word list
cmake -S "$SRC" -B "$BUILD" -G Ninja ${CMAKE_ARGS:-} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_IGNORE_PREFIX_PATH="/usr;/usr/local" \
  -DRIO_USE_VENDORED_CERES=OFF \
  -DPython3_EXECUTABLE="$PY" -DPYTHON_EXECUTABLE="$PY"
TARGETS="rio_solver_py"; [ -f "$SRC/src/gp_solver.cpp" ] && TARGETS="$TARGETS rio_gp"
# shellcheck disable=SC2086
cmake --build "$BUILD" -j"$JOBS" --target $TARGETS

mkdir -p "$BUILD/lib"
cp "$BUILD"/rio_solver.cpython-*.so "$BUILD/lib/"
ls "$BUILD"/rio_gp.cpython-*.so >/dev/null 2>&1 && cp "$BUILD"/rio_gp.cpython-*.so "$BUILD/lib/"
grep -E '^(Ceres_DIR|Eigen3_DIR|pybind11_DIR|fmt_DIR)' "$BUILD/CMakeCache.txt" || true
echo "built for $PY -> $BUILD/lib"
"$PY" "$SRC/scripts/pixi_check.py" "$BUILD/lib"
