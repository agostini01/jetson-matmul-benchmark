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

__global__ void reduce_sum_naive_kernel(const float in, float *out, int n) {
  // Resize on launch
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // Accumlate on register result if not out of bounds
  float x = 0.f;
  if (i < n)
    x += in[i];
  if (i + blockDim.x < n)
    x += in[i + blockDim.x];
  s[tid] = x;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0)
    out[blockIdx.x] = s[0];
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
