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
