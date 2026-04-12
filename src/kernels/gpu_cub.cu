#include "matmul/implementation.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace matmul {
namespace {

inline void check_cuda(cudaError_t status, const char *context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

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

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    check_cuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy a->d_a");
    check_cuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy b->d_b");

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    matmul_cub_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError(), "launch matmul_cub_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync matmul_cub_kernel");

    check_cuda(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy d_c->c");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

std::unique_ptr<MatmulImplementation> make_gpu_cub() {
  return std::make_unique<GpuCubMatmul>();
}

} // namespace matmul