#include "utils/utils.h"

#include <cuda_runtime.h>

#include <utility>

using namespace utils;
namespace parreduction {

namespace {

__global__ void reduce_sum_naive_kernel(const float *in, float *out, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float x = 0.0f;
  if (i < n) {
    x += in[i];
  }
  if (i + blockDim.x < n) {
    x += in[i + blockDim.x];
  }
  s[tid] = x;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = s[0];
  }
}

} // namespace

float driver(float *in, int n) {
  const int block = 256;
  int curr_n = n;
  std::shared_ptr<float> d_in = makeCudaShared<float>(n);

  checkCuda(
      cudaMemcpy(d_in.get(), in, sizeof(float) * n, cudaMemcpyHostToDevice),
      "in to device copy");

  while (curr_n > 1) {
    int grid = (curr_n + block * 2 - 1) / (block * 2);
    std::shared_ptr<float> d_out = makeCudaShared<float>(grid);
    reduce_sum_naive_kernel<<<grid, block, block * sizeof(float)>>>(
        d_in.get(), d_out.get(), curr_n);
    checkCuda(cudaGetLastError(), "launch reduce_sum_naive_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync reduce_sum_naive_kernel");

    // Swap input and output for next iteration
    std::swap(d_in, d_out);
    curr_n = grid;
  }

  float result = 0.0f;
  checkCuda(
      cudaMemcpy(&result, d_in.get(), sizeof(float), cudaMemcpyDeviceToHost),
      "result to host copy");
  return result;
}
} // namespace parreduction
