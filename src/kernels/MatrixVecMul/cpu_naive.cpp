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

  // How many elements in the grid
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
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int maxThreadsPerBlock = prop.maxThreadsPerBlock;       // 1024
  int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor; // e.g., 2048
  int numSMs = prop.multiProcessorCount;                  // e.g., 128
  int block_size = 256;
  int MAXNUMTHREADS = maxThreadsPerSM * numSMs;
  int grid_size =
      min((n + block_size - 1) / block_size, MAXNUMTHREADS / block_size);
  matvecmul<<<grid_size, block_size>>>(m, v, o, n);
  cuda_check(cudaGetLastError(), "launch kernel");
  cuda_check(cudaDeviceSynchronize(), "sync matvecmul")

      cuda_check(
          cudaMemcpy(o, d_o.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
} // namespace matvecmul