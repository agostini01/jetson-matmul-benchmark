#include "matmul/implementation.hpp"
#include "utils/utils.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <memory>

using namespace utils;
namespace matmul {
namespace {

constexpr int BLOCK_THREADS = 128;
constexpr int TILE = 16;

__global__ void matmul_cub_kernel(const float *a, const float *b, float *c,
                                  int n) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;

  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float a_tile[BLOCK_THREADS];
  __shared__ float b_tile[BLOCK_THREADS];

  const int tile_row = blockIdx.y * TILE;
  const int tile_col = blockIdx.x * TILE;
  const int tile_count = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;

  for (int local_row = 0; local_row < TILE; ++local_row) {
    const int row = tile_row + local_row;

    for (int local_col = 0; local_col < TILE; ++local_col) {
      const int col = tile_col + local_col;
      const bool in_bounds = row < n && col < n;

      float acc = 0.0f;

      for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const int k = tile_idx * BLOCK_THREADS + threadIdx.x;

        a_tile[threadIdx.x] = (in_bounds && k < n) ? a[row * n + k] : 0.0f;
        b_tile[threadIdx.x] = (in_bounds && k < n) ? b[k * n + col] : 0.0f;

        __syncthreads();

        const float partial = a_tile[threadIdx.x] * b_tile[threadIdx.x];
        const float tile_sum = BlockReduce(temp_storage).Sum(partial);

        if (threadIdx.x == 0) {
          acc += tile_sum;
        }

        __syncthreads();
      }

      if (threadIdx.x == 0 && in_bounds) {
        c[row * n + col] = acc;
      }

      __syncthreads();
    }
  }
}

} // namespace

class GpuCubMatmul final : public MatmulImplementation {

public:
  const char *name() const override { return "gpu_cub_cuda"; }
  bool is_optimized() const override { return true; }

  void multiply(const Matrix &a, const Matrix &b, Matrix &c) override {
    const int n = static_cast<int>(a.size());
    const std::size_t bytes = static_cast<std::size_t>(n) *
                              static_cast<std::size_t>(n) * sizeof(float);

    std::shared_ptr<float> d_a = makeCudaShared<float>(n * n);
    std::shared_ptr<float> d_b = makeCudaShared<float>(n * n);
    std::shared_ptr<float> d_c = makeCudaShared<float>(n * n);

    checkCuda(cudaMemcpy(d_a.get(), a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy a->d_a");
    checkCuda(cudaMemcpy(d_b.get(), b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy b->d_b");

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    matmul_cub_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n);
    checkCuda(cudaGetLastError(), "launch matmul_cub_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync matmul_cub_kernel");

    checkCuda(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy d_c->c");
  }
};

std::unique_ptr<MatmulImplementation> make_gpu_cub() {
  return std::make_unique<GpuCubMatmul>();
}

} // namespace matmul