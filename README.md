# Jetson MatMul Benchmark (CPU + CUDA)

[![CI Test](https://github.com/agostini01/jetson-matmul-benchmark/actions/workflows/ci-test.yml/badge.svg?branch=main)](https://github.com/agostini01/jetson-matmul-benchmark/actions/workflows/ci-test.yml)

This project benchmarks float32 matrix multiplication on:

- CPU baseline (`cpu_naive`)
- CPU optimized (`cpu_optimized_openmp` when OpenMP is available, otherwise tiled fallback)
- CPU BLAS (`cpu_blas`) when BLAS is available
- GPU baseline CUDA (`gpu_naive_cuda`)
- GPU optimized CUDA tiled (`gpu_optimized_tiled_cuda`)

It includes:

- an object-oriented benchmarking harness in C++
- isolated implementation files for each variant
- validation tests (small matrix)
- benchmarks (default full matrix size 1024)
- JSON output for benchmark results
- GitHub Actions workflows for push validation and manual benchmarks

## Layout

```
include/matmul/
	matrix.hpp 			# Common matrix definitions and utilities
	implementation.hpp 	# Abstract base class for implementations
	harness.hpp
	registry.hpp

src/harness/ 		# Contains the benchmarking harness implementation
	harness.cpp

src/kernels/ 		# Contains all implementations
	registry.cpp 	# Registers implementations for dynamic selection
	cpu_naive.cpp
	cpu_optimized.cpp
	gpu_naive.cu
	gpu_tiled.cu

tests/
	test_main.cpp

benchmarks/
	benchmark_main.cpp # Contains the benchmark driver that runs all implementations and outputs JSON

.github/workflows/
	ci-test.yml 	# Runs on push, executes build + validation tests.
	benchmark.yml 	# Manual workflow for benchmarks, uploads JSON artifact.
```

## Devcontainer

Development can be done inside the provided devcontainer, which is configured for CUDA development on Jetson devices. See `.devcontainer/devcontainer.json` and `.devcontainer/Dockerfile` for details.

## Build

Configure and build:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

CMake auto-detects OpenMP support. If available, it builds the optimized CPU version with OpenMP.

BLAS integration is enabled by default. The project prefers NVPL when available and falls back to generic BLAS discovery.

## Validation Test

Runs correctness checks for all implementations against `cpu_naive` reference.

```
cmake -B build -DTEST_SIZE=128
cmake --build build --target test
```

Or after initial configuration:

```
cmake --build build --target test
```

## Benchmark

Default benchmark configuration:

- benchmark size: `1024`
- validation size: `128`
- warmup runs: `2`
- timed runs: `10`

Configure and run:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DBENCH_SIZE=1024 \
  -DTEST_SIZE=128 \
  -DWARMUP_RUNS=2 \
  -DTIMED_RUNS=10 \
  -DJSON_OUTPUT=benchmark-results.json

cmake --build build --target benchmark
```

Or after initial configuration, override parameters:

```
cmake -B build -DBENCH_SIZE=2048 -DTEST_SIZE=256
cmake --build build --target benchmark
```

Useful BLAS options:

```
# Default behavior: BLAS enabled, prefer NVPL first, fallback to generic BLAS
cmake -B build -DMATMUL_USE_BLAS_BASELINE=ON -DMATMUL_PREFER_NVPL=ON

# Disable BLAS integration entirely
cmake -B build -DMATMUL_USE_BLAS_BASELINE=OFF
```

Benchmark reference behavior:

- benchmark runner prefers `cpu_blas` as the validation reference when available
- if `cpu_blas` is unavailable, it falls back to `cpu_naive`
- tests still validate against `cpu_naive`

## Output

Benchmark output is written to JSON, for example:

```
benchmark-results.json
```

This JSON includes config, per-implementation validation status, run times, and summary metrics (min/max/mean/stddev, GFLOPS).

## Clean

Remove build artifacts:

```
rm -rf build
```

## Common Issues

Devcontainer on jetson required `--runtime=nvidia` to access GPU. 

As root, make sure your Docker daemon is configured with NVIDIA Container Runtime. For example, `/etc/docker/daemon.json` should include:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

Run `sudo systemctl restart docker` after making changes.

## GitHub Actions

- `ci-test.yml`: runs on push/pull request and executes build + validation tests.
- `benchmark.yml`: manual `workflow_dispatch` benchmark run from GitHub Web UI and uploads JSON artifact.

Both workflows target self-hosted ARM64 Linux runners (Jetson).

## Self-hosted Runner Setup

See `docs/runner-setup.md` for instructions to register this container/host as a GitHub Actions self-hosted runner.