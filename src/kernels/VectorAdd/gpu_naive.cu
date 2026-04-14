#include "utils/utils.h"

#include <cuda_runtime.h>

#include <memory>

using namespace utils;
namespace vecadd {

namespace {

__global__ void vectoradd_naive_kernel(const float *a, const float *b, float *c,
                                       int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

} // namespace

void driver(float *a, float *b, float *c, int n) {
  std::shared_ptr<float> d_a = makeCudaShared<float>(n);
  std::shared_ptr<float> d_b = makeCudaShared<float>(n);
  std::shared_ptr<float> d_c = makeCudaShared<float>(n);
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  checkCuda(cudaMemcpy(d_a.get(), a, bytes, cudaMemcpyHostToDevice),
            "d_a to device copy");
  checkCuda(cudaMemcpy(d_b.get(), b, bytes, cudaMemcpyHostToDevice),
            "d_b to device copy");

  const int block = 256;
  int grid = (n + block - 1) / block;
  vectoradd_naive_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n);
  checkCuda(cudaGetLastError(), "launch vectoradd_naive_kernel");
  checkCuda(cudaDeviceSynchronize(), "sync vectoradd_naive_kernel");

  checkCuda(cudaMemcpy(c, d_c.get(), bytes, cudaMemcpyDeviceToHost),
            "d_c to host copy");
}
} // namespace vecadd
