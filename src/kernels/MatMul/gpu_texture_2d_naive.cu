#include "matmul/implementation.hpp"
#include "utils/utils.h"

#include <cuda_runtime.h>

#include <memory>

using namespace utils;

namespace matmul {

namespace {

constexpr int BLOCK = 16;

cudaTextureObject_t create_pitch2d_texture_object(const float *device_ptr,
                                                  std::size_t pitch_bytes,
                                                  int width, int height) {
  cudaResourceDesc resource_desc{};
  resource_desc.resType = cudaResourceTypePitch2D;
  resource_desc.res.pitch2D.devPtr = const_cast<float *>(device_ptr);
  resource_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  resource_desc.res.pitch2D.width = static_cast<std::size_t>(width);
  resource_desc.res.pitch2D.height = static_cast<std::size_t>(height);
  resource_desc.res.pitch2D.pitchInBytes = pitch_bytes;

  cudaTextureDesc texture_desc{};
  texture_desc.addressMode[0] = cudaAddressModeClamp;
  texture_desc.addressMode[1] = cudaAddressModeClamp;
  texture_desc.filterMode = cudaFilterModePoint;
  texture_desc.readMode = cudaReadModeElementType;
  texture_desc.normalizedCoords = 0;

  cudaTextureObject_t texture = 0;
  checkCuda(
      cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr),
      "cudaCreateTextureObject");
  return texture;
}

__global__ void matmul_texture2d_naive_kernel(cudaTextureObject_t tex_a,
                                              cudaTextureObject_t tex_b,
                                              float *c, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < n; ++k) {
    const float a_value = tex2D<float>(tex_a, static_cast<float>(k) + 0.5f,
                                       static_cast<float>(row) + 0.5f);
    const float b_value = tex2D<float>(tex_b, static_cast<float>(col) + 0.5f,
                                       static_cast<float>(k) + 0.5f);
    acc += a_value * b_value;
  }

  c[row * n + col] = acc;
}

} // namespace

class GpuTexture2DNaiveMatmul final : public MatmulImplementation {
public:
  const char *name() const override { return "gpu_texture_2d_naive_cuda"; }
  bool is_optimized() const override { return false; }

  void multiply(const Matrix &a, const Matrix &b, Matrix &c) override {
    const int n = static_cast<int>(a.size());
    const std::size_t row_bytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t bytes = row_bytes * static_cast<std::size_t>(n);

    float *raw_a = nullptr;
    float *raw_b = nullptr;
    std::size_t pitch_a = 0;
    std::size_t pitch_b = 0;

    checkCuda(cudaMallocPitch(reinterpret_cast<void **>(&raw_a), &pitch_a,
                              row_bytes, static_cast<std::size_t>(n)),
              "cudaMallocPitch d_a");
    checkCuda(cudaMallocPitch(reinterpret_cast<void **>(&raw_b), &pitch_b,
                              row_bytes, static_cast<std::size_t>(n)),
              "cudaMallocPitch d_b");

    std::shared_ptr<float> d_a(raw_a, [](float *p) {
      if (p) {
        cudaFree(p);
      }
    });
    std::shared_ptr<float> d_b(raw_b, [](float *p) {
      if (p) {
        cudaFree(p);
      }
    });
    std::shared_ptr<float> d_c = makeCudaShared<float>(n * n);

    checkCuda(cudaMemcpy2D(d_a.get(), pitch_a, a.data(), row_bytes, row_bytes,
                           static_cast<std::size_t>(n), cudaMemcpyHostToDevice),
              "cudaMemcpy2D a->d_a");
    checkCuda(cudaMemcpy2D(d_b.get(), pitch_b, b.data(), row_bytes, row_bytes,
                           static_cast<std::size_t>(n), cudaMemcpyHostToDevice),
              "cudaMemcpy2D b->d_b");

    const cudaTextureObject_t tex_a =
        create_pitch2d_texture_object(d_a.get(), pitch_a, n, n);
    const cudaTextureObject_t tex_b =
        create_pitch2d_texture_object(d_b.get(), pitch_b, n, n);

    const dim3 block(BLOCK, BLOCK);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    matmul_texture2d_naive_kernel<<<grid, block>>>(tex_a, tex_b, d_c.get(), n);
    checkCuda(cudaGetLastError(), "launch matmul_texture2d_naive_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync matmul_texture2d_naive_kernel");

    checkCuda(cudaDestroyTextureObject(tex_a),
              "cudaDestroyTextureObject tex_a");
    checkCuda(cudaDestroyTextureObject(tex_b),
              "cudaDestroyTextureObject tex_b");

    checkCuda(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy d_c->c");
  }
};

std::unique_ptr<MatmulImplementation> make_gpu_texture_2d_naive() {
  return std::make_unique<GpuTexture2DNaiveMatmul>();
}

} // namespace matmul
