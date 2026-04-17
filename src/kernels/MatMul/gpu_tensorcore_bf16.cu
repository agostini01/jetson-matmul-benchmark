#include "matmul/implementation.hpp"
#include "utils/utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <memory>

using namespace utils;

namespace matmul {

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void matmul_tensorcore_bf16_kernel(const float *a, const float *b,
                                              float *c, int n) {
#if __CUDA_ARCH__ >= 800
  const int lane = threadIdx.x;
  const int tile_row = blockIdx.y * WMMA_M;
  const int tile_col = blockIdx.x * WMMA_N;

  __align__(32) __shared__ __nv_bfloat16 a_tile[WMMA_M * WMMA_K];
  __align__(32) __shared__ __nv_bfloat16 b_tile[WMMA_K * WMMA_N];
  __align__(32) __shared__ float c_tile[WMMA_M * WMMA_N];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                         __nv_bfloat16, nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                         __nv_bfloat16, nvcuda::wmma::row_major>
      b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         float>
      c_frag;

  nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < n; k0 += WMMA_K) {
    for (int idx = lane; idx < WMMA_M * WMMA_K; idx += WARP_SIZE) {
      const int local_row = idx / WMMA_K;
      const int local_col = idx % WMMA_K;
      const int global_row = tile_row + local_row;
      const int global_col = k0 + local_col;
      const float value = (global_row < n && global_col < n)
                              ? a[global_row * n + global_col]
                              : 0.0f;
      a_tile[idx] = __float2bfloat16(value);
    }

    for (int idx = lane; idx < WMMA_K * WMMA_N; idx += WARP_SIZE) {
      const int local_row = idx / WMMA_N;
      const int local_col = idx % WMMA_N;
      const int global_row = k0 + local_row;
      const int global_col = tile_col + local_col;
      const float value = (global_row < n && global_col < n)
                              ? b[global_row * n + global_col]
                              : 0.0f;
      b_tile[idx] = __float2bfloat16(value);
    }

    __syncthreads();

    nvcuda::wmma::load_matrix_sync(a_frag, a_tile, WMMA_K);
    nvcuda::wmma::load_matrix_sync(b_frag, b_tile, WMMA_N);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();
  }

  nvcuda::wmma::store_matrix_sync(c_tile, c_frag, WMMA_N,
                                  nvcuda::wmma::mem_row_major);
  __syncthreads();

  for (int idx = lane; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
    const int local_row = idx / WMMA_N;
    const int local_col = idx % WMMA_N;
    const int global_row = tile_row + local_row;
    const int global_col = tile_col + local_col;
    if (global_row < n && global_col < n) {
      c[global_row * n + global_col] = c_tile[idx];
    }
  }
#else
  (void)a;
  (void)b;
  (void)c;
  (void)n;
#endif
}

} // namespace

class GpuTensorcoreBf16Matmul final : public MatmulImplementation {
public:
  const char *name() const override { return "gpu_tensorcore_bf16_cuda"; }
  bool is_optimized() const override { return true; }

  void multiply(const Matrix &a, const Matrix &b, Matrix &c) override {
    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    if (prop.major < 8) {
      throw std::runtime_error(
          "BF16 Tensor Cores require compute capability 8.0+");
    }

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

    const dim3 block(WARP_SIZE, 1, 1);
    const dim3 grid((n + WMMA_N - 1) / WMMA_N, (n + WMMA_M - 1) / WMMA_M, 1);

    matmul_tensorcore_bf16_kernel<<<grid, block>>>(d_a.get(), d_b.get(),
                                                   d_c.get(), n);
    checkCuda(cudaGetLastError(), "launch matmul_tensorcore_bf16_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync matmul_tensorcore_bf16_kernel");

    checkCuda(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy d_c->c");
  }
};

std::unique_ptr<MatmulImplementation> make_gpu_tensorcore_bf16() {
  return std::make_unique<GpuTensorcoreBf16Matmul>();
}

} // namespace matmul
