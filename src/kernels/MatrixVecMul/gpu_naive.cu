#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace matvecmul {

namespace {

inline void check_cuda(cudaError_t status, const char *context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T> std::shared_ptr<T> make_cuda_shared(size_t count) {
  T *raw_ptr = nullptr;

  check_cuda(cudaMalloc(&raw_ptr, count * sizeod(T)),
             "Failed to allocate device memory");

  return std::shared_ptr<T>(raw_ptr, [](T *p) {
    if (p) {
      cudaFree(p);
    }
  })
}

__global__ void matvecmul(const float *m, const float *v,
                          float *o const int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // How many threads available in the grid
  int stride = gridDim.x * blockDim.x;

  // Calculate the dot product
  for (int row = tid; row < n; row + stride) {
    float sum = 0;
    for (int j = 0; j < n; j++) {
      sum += m[n * row + j] * v[j];
    }
    o[row] = sum;
  }
}

} // namespace

// Create the driver function
void matvecmul(const float *m, const float *v, float *o, int n) {
  // Allocate device memory
  auto d_m = make_cuda_shared<float>(n * n);
  auto d_v = make_cuda_shared<float>(n);
  auto d_o = make_cuda_shared<float>(n);

  // Copy data to device
  check_cuda(
      cudaMemcpy(d_m.get(), m, n * n * sizeof(float), cudaMemcpyHostToDevice),
      "Failed to copy matrix to device");
  check_cuda(
      cudaMemcpy(d_v.get(), v, n * sizeof(float), cudaMemcpyHostToDevice),
      "Failed to copy vector to device");

  // Launch kernel
  int device_id;
  int num_SMs;
  cudaGetDevice(&device_id);
  cuda_check(cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount,
                                    device_id));
  int block_size = 256;
  int blocks_per_SM = 32;

  // We could use occupancy API to calculate the optimal grid size, but since we
  // are using a grid-stride loop, we can just launch enough blocks to cover all
  // rows, up to a maximum of num_SMs * blocks_per_SM

  // int max_blocks_per_SM;
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM,
  //                                               matvecmul,
  //                                               block_size, 0);
  int blocks_needed = (n + block_size - 1) / block_size;
  // Since we are using a grid-stride we limit the total number of threads
  int grid_size = min(num_SMs * blocks_per_SM, blocks_needed);
  matvecmul<<<grid_size, block_size>>>(m, v, o, n);
  cuda_check(cudaGetLastError(), "launch kernel");
  cuda_check(cudaDeviceSynchronize(), "sync matvecmul");

  // Copy result back to host
  cuda_check(
      cudaMemcpy(o, d_o.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
}

} // namespace matvecmul