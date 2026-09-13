#!/bin/bash
# Build the rio_solver_cpp project.
# Usage: ./scripts/build.sh [--debug] [--install]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

BUILD_TYPE="Release"
INSTALL=0

for arg in "$@"; do
    case "$arg" in
        --debug)    BUILD_TYPE="Debug" ;;
        --install)  INSTALL=1 ;;
    esac
done

BUILD_DIR="$ROOT/build_${BUILD_TYPE,,}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$ROOT" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build . -- -j"$(nproc)"

echo ""
echo "Build complete: $BUILD_DIR"

if [ $INSTALL -eq 1 ]; then
    cmake --install .
    echo "Installed rio_solver.so to analysis/"
fi

echo ""
echo "Run tests:"
echo "  cd $BUILD_DIR && ctest --output-on-failure"
echo ""
echo "Run test directly:"
echo "  $BUILD_DIR/test_spline"
echo ""
echo "To install Python module:"
echo "  $0 --install"
