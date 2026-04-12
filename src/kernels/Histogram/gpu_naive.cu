#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace parreduction {

namespace {

inline void check_cuda(cudaError_t status, const char *context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T> std::shared_ptr<T> make_cuda_shared(size_t count) {
  T *raw_ptr = nullptr;

  // 1. Perform the allocation and check it immediately
  check_cuda(cudaMalloc(&raw_ptr, count * sizeof(T)),
             "Failed to allocate device memory");

  // 2. Wrap it in a shared_ptr with the custom deleter
  return std::shared_ptr<T>(raw_ptr, [](T *p) {
    if (p) {
      cudaFree(p);
    }
  });
}

__global__ void histogram_naive_kernel(const unsigned char *data, int n,
                                       int *hist) {
  __shared__ int local_hist[256];
  int t = threadIdx.x;

  for (int i = t; i < 256; i += blockDim.x)
    local_hist[i] = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + t;
  if (i < n)
    atomicAdd(&local_hist[data[i]], 1);
  __syncthreads();

  for (int i2 = t; i2 < 256; i2 += blockDim.x)
    atomicAdd(&local_hist[i2], local[i2]);
}

} // namespace

float driver(float *in, int n) {
  int block = 256;

  std::shared_ptr<float> d_in = make_cuda_shared<float>(n);

  cuda_check(
      cudaMemcpy(d_in.get(), in, sizeof(float) * n, cudaMemcpyHostToDevice),
      "in to device copy");

  while (curr_n > 1) {
    int grid = (curr_n + block * 2 - 1) / (block * 2);
    std::shared_ptr<float> d_out = make_cuda_shared<float>(grid);
    reduce_sum_naive_kernel<<<grid, block, block * sizeof(float)>>>(
        d_in.get(), d_out.get(), curr_n);
    check_cuda(cudaGetLastError(), "launch reduce_sum_naive_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync reduce_sum_naive_kernel");

    // Swap input and output for next iteration
    std::swap(d_in, d_out);
    curr_n = grid;
  }

  float result cuda_check(
      cudaMemcpy(&c, d_c.get(), sizeof(float) cudaMemcpyDeviceToHost),
      "d_c to host copy");
  return result;
}
} // namespace parreduction
