#include "utils/utils.h"

#include <cuda_runtime.h>

#include <utility>

using namespace utils;
namespace histogram {

namespace {

// Histogram Size - 256 for all possible char values
constexpr int HS = 256;

__global__ void histogram_naive_kernel(const char *data, int *hist, int n) {
  __shared__ int local_hist[HS];
  const int t = threadIdx.x;

  // Reset local histogram
  for (int i = t; i < HS; i += blockDim.x)
    local_hist[i] = 0;
  __syncthreads();

  // Collect local histogram
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
    atomicAdd(&local_hist[data[tid]], 1);
  __syncthreads();

  // Merge local histograms into global histogram
  for (int i = t; i < HS; i += blockDim.x)
    atomicAdd(&hist[i], local_hist[i]);
}

} // namespace

void driver(const char *data, int *hist, int n) {
  std::shared_ptr<char> d_data = makeCudaShared<char>(n);
  std::shared_ptr<int> d_hist = makeCudaShared<int>(HS);
  checkCuda(
      cudaMemcpy(d_data.get(), data, n * sizeof(char), cudaMemcpyHostToDevice),
      "Failed to copy data to device");
  checkCuda(cudaMemset(d_hist.get(), 0, HS * sizeof(int)),
            "Failed to initialize histogram on device");

  const int block = 256;
  const int grid = (n + block - 1) / block;
  histogram_naive_kernel<<<grid, block>>>(d_data.get(), d_hist.get(), n);
  checkCuda(cudaGetLastError(), "Kernel launch failed");

  checkCuda(
      cudaMemcpy(hist, d_hist.get(), HS * sizeof(int), cudaMemcpyDeviceToHost),
      "Failed to copy histogram from device");
}
} // namespace histogram
