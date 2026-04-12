#include <cuda_runtime.h>

namespace vecadd {

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

__global__ vectoradd_naive_kernel(const float *a, const float *b, float *c,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    c[i] = a[i] + b[i];
}

} // namespace

void driver(float *a, float *b, float *c, int n) {
  int n = 10000;

  std::shared_ptr<float> d_a = make_cuda_shared<float>(n);
  std::shared_ptr<float> d_b = make_cuda_shared<float>(n);
  std::shared_ptr<float> d_c = make_cuda_shared<float>(n);

  cuda_check(cudaMemcpy(d_a.get(), a, cudaMemcpyHostToDevice),
             "d_a to device copy");
  cuda_check(cudaMemcpy(d_b.get(), b, cudaMemcpyHostToDevice),
             "d_b to device copy");

  const block = 256;
  int grid = (n + block - 1) / block;
  vectoradd_naive_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n);

  // Sync

  cuda_check(cudaMemcpy(c, d_c.get(), cudaMemcpyDeviceToHost),
             "d_c to host copy");
}
} // namespace vecadd
