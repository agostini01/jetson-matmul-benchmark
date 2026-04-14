#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace utils {

inline void checkCuda(cudaError_t status, const char *context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T> std::shared_ptr<T> makeCudaShared(std::size_t count) {
  T *rawPtr = nullptr;
  checkCuda(cudaMalloc(&rawPtr, count * sizeof(T)),
            "Failed to allocate device memory");

  return std::shared_ptr<T>(rawPtr, [](T *p) {
    if (p) {
      cudaFree(p);
    }
  });
}

} // namespace utils
