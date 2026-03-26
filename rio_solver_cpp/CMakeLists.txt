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
