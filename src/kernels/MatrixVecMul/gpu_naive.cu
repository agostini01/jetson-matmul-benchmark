#include "utils/utils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>

using namespace utils;
namespace matvecmul {

namespace {

__global__ void matvecmulKernel(const float *A, const float *v, float *o,
                                int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // How many threads available in the grid
  int stride = gridDim.x * blockDim.x;

  // Calculate the dot product
  for (int row = tid; row < n; row += stride) {
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
      sum += A[n * row + j] * v[j];
    }
    o[row] = sum;
  }
}

__global__ void matvec_tiled_kernel(const float *A, const float *v, float *o,
                                    int N) {
  extern __shared__ float sx[];

  // Kernel computes a full dot product of N elements
  // grid stride allows to go beyond max num threads
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int row = tid; row < N; row += stride) {
    float acc = 0.f;

    for (int t = 0; t < (N + blockDim.x - 1) / blockDim.x; ++t) {
      int j = t * blockDim.x + threadIdx.x;

      sx[threadIdx.x] = (j < N) ? v[j] : 0.f;
      __syncthreads();
      for (int k = 0; k < blockDim.x; ++k) {
        int col = t * blockDim.x * k;
        if (col < N)
          acc += A[row * N + col] * sx[k];
      }
      __syncthreads();
    }

    o[row] = acc;
  }
}

} // namespace

// Create the driver function
void matvecmul(const float *A, const float *v, float *o, int n) {
  // Allocate device memory
  std::shared_ptr<float> d_m = makeCudaShared<float>(n * n);
  std::shared_ptr<float> d_v = makeCudaShared<float>(n);
  std::shared_ptr<float> d_o = makeCudaShared<float>(n);

  // Copy data to device
  checkCuda(
      cudaMemcpy(d_m.get(), A, n * n * sizeof(float), cudaMemcpyHostToDevice),
      "Failed to copy matrix to device");
  checkCuda(cudaMemcpy(d_v.get(), v, n * sizeof(float), cudaMemcpyHostToDevice),
            "Failed to copy vector to device");

  // Launch kernel
  int device_id;
  int num_SMs;
  checkCuda(cudaGetDevice(&device_id), "get cuda device");
  checkCuda(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount,
                                   device_id),
            "get multiprocessor count");
  int block_size = 256;
  int blocks_per_SM = 32;

  // We could use occupancy API to calculate the optimal grid size, but since we
  // are using a grid-stride loop, we can just launch enough blocks to cover all
  // rows, up to a maximum of num_SMs * blocks_per_SM

  // int max_blocks_per_SM;
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
  //                                               matvecmulKernel,
  //                                               block_size, 0);
  int blocks_needed = (n + block_size - 1) / block_size;
  // Since we are using a grid-stride we limit the total number of threads
  int grid_size = std::min(num_SMs * blocks_per_SM, blocks_needed);
  matvecmulKernel<<<grid_size, block_size>>>(d_m.get(), d_v.get(), d_o.get(),
                                             n);
  checkCuda(cudaGetLastError(), "launch kernel");
  checkCuda(cudaDeviceSynchronize(), "sync matvecmul");

  // Copy result back to host
  checkCuda(cudaMemcpy(o, d_o.get(), n * sizeof(float), cudaMemcpyDeviceToHost),
            "copy output to host");
}

} // namespace matvecmul