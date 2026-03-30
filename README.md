# Jetson MatMul Benchmark (CPU + CUDA)

This project benchmarks float32 matrix multiplication on:

- CPU baseline (`cpu_naive`)
- CPU optimized (`cpu_optimized_openmp` when OpenMP is available, otherwise tiled fallback)
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
	matrix.hpp
	implementation.hpp
	harness.hpp
	registry.hpp

src/
	harness.cpp
	registry.cpp
	cpu_naive.cpp
	cpu_optimized.cpp
	gpu_naive.cu
	gpu_tiled.cu
	test_main.cpp
	benchmark_main.cpp

.github/workflows/
	ci-test.yml
	benchmark.yml
```

## Build

```
make all
```

The Makefile auto-detects OpenMP support. If available, it builds the optimized CPU version with OpenMP.

## Validation Test

Runs correctness checks for all implementations against `cpu_naive` reference.

```
make test TEST_SIZE=128
```

## Benchmark

Default benchmark configuration requested:

- benchmark size: `1024`
- validation size: `128`
- warmup runs: `2`
- timed runs: `10`

Run:

```
make benchmark BENCH_SIZE=1024 TEST_SIZE=128 WARMUP=2 RUNS=10 JSON=benchmark-results.json
```

You can override values from command line with the variables above.

## Output

Benchmark output is written to JSON, for example:

```
benchmark-results.json
```

This JSON includes config, per-implementation validation status, run times, and summary metrics (min/max/mean/stddev, GFLOPS).

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