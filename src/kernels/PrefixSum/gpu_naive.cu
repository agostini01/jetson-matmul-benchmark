#include "utils/utils.h"

#include <cuda_runtime.h>

#include <utility>

using namespace utils;
namespace prefixsum {

namespace {

__global__ void prefixSumNaiveKernel(const float *in, float *out, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = 2 * blockIdx.x * blockDim.x + tid;

  s[tid] = (i < n) ? in[i] : 0.f;
  s[tid + blockDim.x] = (i + blockDim.x < n) ? in[blockDim.x] : 0.f;

  int offset = 1;
  for (int d = blockDim.x; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      s[bi] += s[ai];
    }
    offset <<= 1;
  }

  if (tid == 0)
    s[2 * blockDim.x - 1] = 0;

  for (int d = 1; d < 2 * blockDim.x; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      float t = s[ai];
      s[ai] = s[bi];
      s[bi] += t;
    }
  }
  __syncthreads();

  if (i < n)
    out[i] = s[tid];
  if (i + blockDim.x < n)
    out[i + blockDim.x] = s[tid + blockDim.x];
}

} // namespace

void driver(const float *in, float *out, int n) {

  std::shared_ptr<float> d_in = makeCudaShared<float>(n);
  std::shared_ptr<float> d_out = makeCudaShared<float>(n);

  checkCuda(
      cudaMemcpy(d_in.get(), in, sizeof(float) * n, cudaMemcpyHostToDevice),
      "In to device copy");

  const int block = 256;
  const int grid = (n + block * 2 - 1) / block * 2;
  size_t shared_mem_size = 2 * block * sizeof(float);
  prefixSumNaiveKernel<<<grid, block, shared_mem_size>>>(d_in.get(),
                                                         d_out.get(), n);

  checkCuda(cudaGetLastError(), "Error during kernel launch");
  checkCuda(cudaDeviceSynchronize(), "Error during kernel execution");

  checkCuda(
      cudaMemcpy(out, d_out.get(), sizeof(float) * n, cudaMemcpyHostToDevice),
      "Device to host copy");
}
} // namespace prefixsum
