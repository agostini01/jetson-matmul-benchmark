cmake_minimum_required(VERSION 3.20)
project(matmul-benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Detect OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found")
    add_compile_definitions(USE_OPENMP)
    set(OPENMP_AVAILABLE ON)
else()
    message(STATUS "OpenMP not found, using tiled CPU fallback")
    set(OPENMP_AVAILABLE OFF)
endif()

option(MATMUL_USE_BLAS_BASELINE "Enable BLAS-backed CPU implementation" ON)
option(MATMUL_PREFER_NVPL "Prefer NVPL when searching for BLAS" ON)

set(MATMUL_ENABLE_BLAS OFF)
set(MATMUL_BLAS_PROVIDER "none")

if(MATMUL_USE_BLAS_BASELINE)
    if(MATMUL_PREFER_NVPL)
        set(_matmul_prev_bla_vendor "${BLA_VENDOR}")
        set(BLA_VENDOR NVPL)
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            set(MATMUL_ENABLE_BLAS ON)
            set(MATMUL_BLAS_PROVIDER "NVPL")
        endif()
        if(DEFINED _matmul_prev_bla_vendor)
            set(BLA_VENDOR "${_matmul_prev_bla_vendor}")
        else()
            unset(BLA_VENDOR)
        endif()
    endif()

    if(NOT MATMUL_ENABLE_BLAS)
        unset(BLAS_FOUND CACHE)
        unset(BLAS_LIBRARIES CACHE)
        unset(BLAS_LINKER_FLAGS CACHE)
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            set(MATMUL_ENABLE_BLAS ON)
            set(MATMUL_BLAS_PROVIDER "generic")
        endif()
    endif()
endif()

if(MATMUL_ENABLE_BLAS)
    message(STATUS "BLAS baseline enabled (${MATMUL_BLAS_PROVIDER}): ${BLAS_LIBRARIES}")
else()
    message(STATUS "BLAS baseline disabled (cpu_naive reference fallback)")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -Wpedantic" CACHE STRING "Release flags" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_RELEASE}")

# Benchmark parameters with defaults
set(BENCH_SIZE 1024 CACHE STRING "Matrix size for benchmarks")
set(TEST_SIZE 128 CACHE STRING "Matrix size for validation tests")
set(WARMUP_RUNS 2 CACHE STRING "Number of warmup runs per implementation")
set(TIMED_RUNS 10 CACHE STRING "Number of timed runs per implementation")
set(JSON_OUTPUT "benchmark-results.json" CACHE STRING "JSON output file path")

message(STATUS "Benchmark configuration:")
message(STATUS "  BENCH_SIZE: ${BENCH_SIZE}")
message(STATUS "  TEST_SIZE: ${TEST_SIZE}")
message(STATUS "  WARMUP_RUNS: ${WARMUP_RUNS}")
message(STATUS "  TIMED_RUNS: ${TIMED_RUNS}")
message(STATUS "  JSON_OUTPUT: ${JSON_OUTPUT}")

# Add subdirectories for modular libraries and executables
add_subdirectory(src/harness)
add_subdirectory(src/kernels)
add_subdirectory(tests)
add_subdirectory(benchmarks)

# Custom targets that mimic Makefile behavior
add_custom_target(test
    COMMAND test_matmul --size ${TEST_SIZE}
    DEPENDS test_matmul
    COMMENT "Running validation tests (size=${TEST_SIZE})..."
)

add_custom_target(benchmark
    COMMAND bench_matmul
        --size ${BENCH_SIZE}
        --test-size ${TEST_SIZE}
        --warmup ${WARMUP_RUNS}
        --runs ${TIMED_RUNS}
        --json ${JSON_OUTPUT}
    DEPENDS bench_matmul
    COMMENT "Running benchmark (size=${BENCH_SIZE}, test-size=${TEST_SIZE}, warmup=${WARMUP_RUNS}, runs=${TIMED_RUNS})..."
)
