#include "matmul/implementation.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace matmul {

namespace {

inline void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

__global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        acc += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = acc;
}

}  // namespace

class GpuNaiveMatmul final : public MatmulImplementation {
public:
    const char* name() const override { return "gpu_naive_cuda"; }
    bool is_optimized() const override { return false; }

    void multiply(const Matrix& a, const Matrix& b, Matrix& c) override {
        const int n = static_cast<int>(a.size());
        const std::size_t bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(float);

        float* d_a = nullptr;
        float* d_b = nullptr;
        float* d_c = nullptr;

        check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
        check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
        check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

        check_cuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy a->d_a");
        check_cuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy b->d_b");

        const dim3 block(16, 16);
        const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        matmul_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
        check_cuda(cudaGetLastError(), "launch matmul_naive_kernel");
        check_cuda(cudaDeviceSynchronize(), "sync matmul_naive_kernel");

        check_cuda(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_c->c");

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
};

std::unique_ptr<MatmulImplementation> make_gpu_naive() {
    return std::make_unique<GpuNaiveMatmul>();
}

}  // namespace matmul
