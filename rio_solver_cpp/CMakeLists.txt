cmake_minimum_required(VERSION 3.16)
project(rio_solver_cpp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CPU tuning for the BUILD machine (2026-10-13): x86 -march=native, ARM
# (Jetson companion computer) -mcpu=native.  Override with -DRIO_ARCH_FLAGS=...
# (e.g. "" for a portable build).  The binaries only run on CPUs like the
# build machine: build ON the Jetson.
if(NOT DEFINED RIO_ARCH_FLAGS)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(RIO_ARCH_FLAGS "-mcpu=native")
  else()
    set(RIO_ARCH_FLAGS "-march=native")
  endif()
endif()
message(STATUS "RIO_ARCH_FLAGS: ${RIO_ARCH_FLAGS} (${CMAKE_SYSTEM_PROCESSOR})")
separate_arguments(_rio_arch UNIX_COMMAND "${RIO_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${RIO_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")

# --- Dependencies ---

find_package(Eigen3 3.4 REQUIRED)

# [Session 2] Prefer the locally-patched vendored Ceres (2.1.0 + BANDED_SCHUR)
# over the system libceres-dev (2.2.0, no BANDED_SCHUR hook).
# The local install lives at ceres_vendored_install/ (built from
# lie-spline-experiments/thirdparty/ceres-solver).
set(CERES_VENDORED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ceres_vendored_install")
# RIO_USE_VENDORED_CERES=OFF skips the host-built vendored Ceres (ABI-tied to
# the host toolchain) so another environment (pixi/conda) uses its own Ceres.
option(RIO_USE_VENDORED_CERES "Prefer ceres_vendored_install/ when present" ON)
if(NOT RIO_USE_VENDORED_CERES)
  message(STATUS "RIO_USE_VENDORED_CERES=OFF: using Ceres from CMAKE_PREFIX_PATH / system")
elseif(EXISTS "${CERES_VENDORED_DIR}/lib/cmake/Ceres/CeresConfig.cmake")
  set(Ceres_DIR "${CERES_VENDORED_DIR}/lib/cmake/Ceres")
  message(STATUS "Using vendored (patched) Ceres from ${CERES_VENDORED_DIR}")
else()
  message(WARNING "Vendored Ceres not found at ${CERES_VENDORED_DIR}; "
                  "falling back to system Ceres (BANDED_SCHUR unavailable). "
                  "Run: cd lie-spline-experiments/thirdparty/ceres-solver && "
                  "mkdir build_rio && cd build_rio && cmake .. <flags> && make install")
endif()
find_package(Ceres REQUIRED)

# BANDED_SCHUR exists only in the patched vendored Ceres; guard its use.
get_target_property(_rio_ceres_inc Ceres::ceres INTERFACE_INCLUDE_DIRECTORIES)
find_file(RIO_CERES_TYPES_H ceres/types.h PATHS ${CERES_INCLUDE_DIRS} ${_rio_ceres_inc} NO_DEFAULT_PATH)
if(RIO_CERES_TYPES_H)
  file(STRINGS "${RIO_CERES_TYPES_H}" _rio_banded REGEX "BANDED_SCHUR")
endif()
if(_rio_banded)
  add_compile_definitions(RIO_HAVE_BANDED_SCHUR)
  message(STATUS "Ceres has BANDED_SCHUR (patched): use_banded_schur available")
endif()

find_package(pybind11 REQUIRED)
find_package(fmt REQUIRED)

# basalt-headers (CeresSplineHelper + Sophus)
# Header-only math deps VENDORED in third_party/ (2026-10-13, third_party/VENDORED.md):
# byte-identical copies of the lie-spline-experiments checkout, so the Jetson /
# pixi builds need nothing untracked.
set(BASALT_HEADERS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/basalt-headers"
    CACHE PATH "Path to basalt-headers directory")

set(SOPHUS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Sophus" CACHE PATH "Path to Sophus (contains sophus/)")

if(NOT EXISTS "${BASALT_HEADERS_DIR}/include/basalt/spline/ceres_spline_helper.h")
  message(FATAL_ERROR "basalt-headers not found at ${BASALT_HEADERS_DIR}. "
    "See third_party/VENDORED.md.")
endif()

# --- Library: rio_solver ---

add_library(rio_solver STATIC
    src/solver.cpp
    src/sliding_window_solver.cpp
)
set_target_properties(rio_solver PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(rio_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BASALT_HEADERS_DIR}/include
    ${SOPHUS_DIR}
)

target_link_libraries(rio_solver PUBLIC
    Eigen3::Eigen
    Ceres::ceres
    fmt::fmt
)

# --- Python extension: rio_solver_py ---

pybind11_add_module(rio_solver_py
    src/pybind_module.cpp
)

target_include_directories(rio_solver_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BASALT_HEADERS_DIR}/include
    ${SOPHUS_DIR}
)

target_link_libraries(rio_solver_py PRIVATE
    rio_solver
    Eigen3::Eigen
    Ceres::ceres
)

# Rename to rio_solver (Python import name)
set_target_properties(rio_solver_py PROPERTIES OUTPUT_NAME "rio_solver")

# Install into analysis/ so Python can import it
install(TARGETS rio_solver_py
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../analysis"
)

# --- Tests ---

enable_testing()

add_executable(test_spline tests/test_spline.cpp)
target_include_directories(test_spline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BASALT_HEADERS_DIR}/include
    ${SOPHUS_DIR}
)
target_link_libraries(test_spline PRIVATE
    Eigen3::Eigen
    Ceres::ceres
    fmt::fmt
)
add_test(NAME test_spline COMMAND test_spline)

# --- GP/WNOA backend: OPTIONAL (2026-10-13). The public spline release ships
# without src/gp_solver.cpp; the targets are skipped when it is absent.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/gp_solver.cpp")
# --- GP/WNOA sliding-window backend (branch gp-backend, Phase 3) ---
# Eigen-only port of the verified Python spike (analysis/gp_spike/); no
# GTSAM/Ceres/steam dependency (vendored steam kept for ablations only).
pybind11_add_module(rio_gp src/pybind_gp.cpp src/gp_solver.cpp)
target_compile_options(rio_gp PRIVATE ${_rio_arch} -O3)
target_include_directories(rio_gp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${BASALT_HEADERS_DIR}/include ${SOPHUS_DIR})
target_link_libraries(rio_gp PRIVATE Eigen3::Eigen fmt::fmt)
install(TARGETS rio_gp LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../analysis")
message(STATUS "Building rio_gp (GP/WNOA sliding-window backend)")

add_executable(test_gp_solver_math tests/test_gp_solver_math.cpp)
target_compile_options(test_gp_solver_math PRIVATE ${_rio_arch} -O2)
target_include_directories(test_gp_solver_math PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${BASALT_HEADERS_DIR}/include ${SOPHUS_DIR})
target_link_libraries(test_gp_solver_math PRIVATE Eigen3::Eigen)
add_test(NAME test_gp_solver_math COMMAND test_gp_solver_math)
else()
  message(STATUS "GP backend sources absent: rio_gp not built (spline-only tree)")
endif()
