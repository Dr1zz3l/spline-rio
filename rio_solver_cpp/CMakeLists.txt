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
