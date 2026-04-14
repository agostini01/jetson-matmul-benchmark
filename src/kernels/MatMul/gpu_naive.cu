#include "matmul/implementation.hpp"
#include "utils/utils.h"

#include <cuda_runtime.h>

#include <memory>

using namespace utils;
namespace matmul {

namespace {

__global__ void matmul_naive_kernel(const float *a, const float *b, float *c,
                                    int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < n; ++k) {
    acc += a[row * n + k] * b[k * n + col];
  }
  c[row * n + col] = acc;
}

} // namespace

class GpuNaiveMatmul final : public MatmulImplementation {
public:
  const char *name() const override { return "gpu_naive_cuda"; }
  bool is_optimized() const override { return false; }

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

    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n);
    checkCuda(cudaGetLastError(), "launch matmul_naive_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync matmul_naive_kernel");

    checkCuda(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy d_c->c");
  }
};

std::unique_ptr<MatmulImplementation> make_gpu_naive() {
  return std::make_unique<GpuNaiveMatmul>();
}

} // namespace matmul
