#include "matmul/implementation.hpp"
#include "utils/utils.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <memory>

using namespace utils;
namespace matmul {

namespace {

inline void check_cublas(cublasStatus_t status, const char *context) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuBLAS error");
  }
}

} // namespace

class GpuCublasMatmul final : public MatmulImplementation {
public:
  GpuCublasMatmul() : handle_(nullptr) {
    check_cublas(cublasCreate(&handle_), "Failed to create cuBLAS handle");
  }

  ~GpuCublasMatmul() {
    if (handle_) {
      cublasDestroy(handle_);
    }
  }

  const char *name() const override { return "gpu_cublas"; }
  bool is_optimized() const override { return true; }

  void multiply(const Matrix &a, const Matrix &b, Matrix &c) override {
    const int n = static_cast<int>(a.size());
    const std::size_t bytes = static_cast<std::size_t>(n) *
                              static_cast<std::size_t>(n) * sizeof(float);

    std::shared_ptr<float> d_a = makeCudaShared<float>(n * n);
    std::shared_ptr<float> d_b = makeCudaShared<float>(n * n);
    std::shared_ptr<float> d_c = makeCudaShared<float>(n * n);

    checkCuda(cudaMemcpy(d_a.get(), a.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy a->d_a");
    checkCuda(cudaMemcpy(d_b.get(), b.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy b->d_b");

    // Matrix stores row-major data, but cuBLAS interprets buffers as
    // column-major. Swapping the operands computes (B * A) in column-major,
    // which maps back to the desired row-major result A * B in the output
    // buffer.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_cublas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                             d_b.get(), n, d_a.get(), n, &beta, d_c.get(), n),
                 "cublasSgemm failed");

    checkCuda(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy d_c->c");
  }

private:
  cublasHandle_t handle_;
};

std::unique_ptr<MatmulImplementation> make_gpu_cublas() {
  return std::make_unique<GpuCublasMatmul>();
}

} // namespace matmul
