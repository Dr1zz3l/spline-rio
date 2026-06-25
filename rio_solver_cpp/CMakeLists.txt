cmake_minimum_required(VERSION 3.16)
project(rio_solver_cpp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")

# --- Dependencies ---

find_package(Eigen3 3.4 REQUIRED)

# [Session 2] Prefer the locally-patched vendored Ceres (2.1.0 + BANDED_SCHUR)
# over the system libceres-dev (2.2.0, no BANDED_SCHUR hook).
# The local install lives at ceres_vendored_install/ (built from
# lie-spline-experiments/thirdparty/ceres-solver).
set(CERES_VENDORED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ceres_vendored_install")
if(EXISTS "${CERES_VENDORED_DIR}/lib/cmake/Ceres/CeresConfig.cmake")
  set(Ceres_DIR "${CERES_VENDORED_DIR}/lib/cmake/Ceres")
  message(STATUS "Using vendored (patched) Ceres from ${CERES_VENDORED_DIR}")
else()
  message(WARNING "Vendored Ceres not found at ${CERES_VENDORED_DIR}; "
                  "falling back to system Ceres (BANDED_SCHUR unavailable). "
                  "Run: cd lie-spline-experiments/thirdparty/ceres-solver && "
                  "mkdir build_rio && cd build_rio && cmake .. <flags> && make install")
endif()
find_package(Ceres REQUIRED)

find_package(pybind11 REQUIRED)
find_package(fmt REQUIRED)

# basalt-headers (CeresSplineHelper + Sophus)
set(BASALT_HEADERS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../lie-spline-experiments/thirdparty/basalt/thirdparty/basalt-headers"
    CACHE PATH "Path to basalt-headers directory")

set(SOPHUS_DIR "${BASALT_HEADERS_DIR}/thirdparty/Sophus")

if(NOT EXISTS "${BASALT_HEADERS_DIR}/include/basalt/spline/ceres_spline_helper.h")
  message(FATAL_ERROR "basalt-headers not found at ${BASALT_HEADERS_DIR}. "
    "Make sure lie-spline-experiments submodules are initialised.")
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

# GTSAM-tangent factor math (no GTSAM dependency: validates the convention/chain
# of the Phase-1 factors via numerical diff + Ceres residual parity).
add_executable(test_gtsam_factor_math tests/test_gtsam_factor_math.cpp)
target_include_directories(test_gtsam_factor_math PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BASALT_HEADERS_DIR}/include
    ${SOPHUS_DIR}
)
target_link_libraries(test_gtsam_factor_math PRIVATE
    Eigen3::Eigen
    Ceres::ceres
    fmt::fmt
)
add_test(NAME test_gtsam_factor_math COMMAND test_gtsam_factor_math)

# GTSAM factor wrappers (Phase 1) + IncrementalFixedLagSmoother backend (Phase 2).
# Prefer the VENDORED gtsam (built with -march=native + GTSAM_BUILD_UNSTABLE, so
# it provides gtsam_unstable AND matches our AVX Eigen alignment). Fall back to
# the apt libgtsam-dev (baseline x86-64, no unstable) for the factor test only.
# NOTE: Ubuntu's GTSAMConfig.cmake is broken (missing CppUnitLite.a), so we link
# libgtsam.so directly instead of find_package(GTSAM).
set(GTSAM_VENDORED "${CMAKE_CURRENT_SOURCE_DIR}/gtsam_vendored_install")
set(GTSAM_OK FALSE)
if(EXISTS "${GTSAM_VENDORED}/lib/libgtsam_unstable.so")
  set(GTSAM_INC "${GTSAM_VENDORED}/include")
  set(GTSAM_LIB "${GTSAM_VENDORED}/lib/libgtsam.so")
  set(GTSAM_UNSTABLE_LIB "${GTSAM_VENDORED}/lib/libgtsam_unstable.so")
  set(GTSAM_MARCH "-march=native")     # vendored matches our build
  set(GTSAM_OK TRUE)
  message(STATUS "Using VENDORED GTSAM (march=native + unstable): ${GTSAM_VENDORED}")
else()
  find_library(GTSAM_LIB gtsam)
  find_path(GTSAM_INC gtsam/nonlinear/NonlinearFactor.h)
  set(GTSAM_UNSTABLE_LIB "")           # apt ships no gtsam_unstable
  set(GTSAM_MARCH "-march=x86-64")     # apt is baseline; avoid Eigen ABI mismatch
  if(GTSAM_LIB AND GTSAM_INC)
    set(GTSAM_OK TRUE)
    message(STATUS "Using system GTSAM (apt, no unstable, -march=x86-64): ${GTSAM_LIB}")
  else()
    message(STATUS "GTSAM not found; skipping GTSAM targets")
  endif()
endif()

if(GTSAM_OK)
  # --- Phase 1: factor-wrapper test (works with apt or vendored) ---
  # vendored gtsam pulls libgtsam_unstable / libmetis-gtsam from its own lib dir
  get_filename_component(GTSAM_LIBDIR "${GTSAM_LIB}" DIRECTORY)
  add_executable(test_gtsam_factors tests/test_gtsam_factors.cpp)
  target_compile_options(test_gtsam_factors PRIVATE ${GTSAM_MARCH})
  target_include_directories(test_gtsam_factors PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${BASALT_HEADERS_DIR}/include ${SOPHUS_DIR} ${GTSAM_INC})
  target_link_libraries(test_gtsam_factors PRIVATE Eigen3::Eigen ${GTSAM_LIB} fmt::fmt tbb)
  set_target_properties(test_gtsam_factors PROPERTIES
      BUILD_RPATH "${GTSAM_LIBDIR}" INSTALL_RPATH "${GTSAM_LIBDIR}")
  target_link_options(test_gtsam_factors PRIVATE "-Wl,--disable-new-dtags")
  add_test(NAME test_gtsam_factors COMMAND test_gtsam_factors)
endif()

if(GTSAM_OK AND GTSAM_UNSTABLE_LIB)
  # --- Phase 2: IsamSolver backend + pybind module (needs gtsam_unstable) ---
  add_library(rio_isam_lib STATIC src/isam_sliding_window_solver.cpp)
  set_target_properties(rio_isam_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_options(rio_isam_lib PRIVATE ${GTSAM_MARCH})
  target_include_directories(rio_isam_lib PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${BASALT_HEADERS_DIR}/include ${SOPHUS_DIR} ${GTSAM_INC})
  target_link_libraries(rio_isam_lib PUBLIC
      Eigen3::Eigen ${GTSAM_LIB} ${GTSAM_UNSTABLE_LIB} fmt::fmt tbb)

  pybind11_add_module(rio_isam src/pybind_isam.cpp)
  target_compile_options(rio_isam PRIVATE ${GTSAM_MARCH})
  target_include_directories(rio_isam PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${BASALT_HEADERS_DIR}/include ${SOPHUS_DIR} ${GTSAM_INC})
  target_link_libraries(rio_isam PRIVATE rio_isam_lib ${GTSAM_LIB} ${GTSAM_UNSTABLE_LIB})
  set_target_properties(rio_isam PROPERTIES
      BUILD_RPATH "${GTSAM_LIBDIR}" INSTALL_RPATH "${GTSAM_LIBDIR}")
  target_link_options(rio_isam PRIVATE "-Wl,--disable-new-dtags")
  install(TARGETS rio_isam LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../analysis")
  message(STATUS "Building rio_isam (IncrementalFixedLagSmoother backend)")
endif()
