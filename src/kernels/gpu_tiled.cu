#include "matmul/implementation.hpp"

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

constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const float *a, const float *b, float *c,
                                    int n) {
  __shared__ float a_tile[TILE][TILE];
  __shared__ float b_tile[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  const int tile_count = (n + TILE - 1) / TILE;

  // Iterate over all tile pairs needed for the dot product of row and col
  // A therad will visit all tiles in the K dimension
  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int a_col = tile_idx * TILE + threadIdx.x;
    const int b_row = tile_idx * TILE + threadIdx.y;

    // Each thread loads one A element and one B element into shared memory
    a_tile[threadIdx.y][threadIdx.x] =
        (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads(); // All threads must finish loading

    // Compute a tile sized dot product for 1 set of tiles, using shared memory
    // for reads
    for (int k = 0; k < TILE; ++k) {
      acc += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
    }

    __syncthreads(); // Prevents threads from writing the next tile before
                     // others finish reading
  }

  if (row < n && col < n) {
    c[row * n + col] = acc;
  }
}

} // namespace

class GpuTiledMatmul final : public MatmulImplementation {
public:
  const char *name() const override { return "gpu_optimized_tiled_cuda"; }
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

    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError(), "launch matmul_tiled_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync matmul_tiled_kernel");

    check_cuda(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy d_c->c");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

std::unique_ptr<MatmulImplementation> make_gpu_tiled() {
  return std::make_unique<GpuTiledMatmul>();
}

} // namespace matmul
