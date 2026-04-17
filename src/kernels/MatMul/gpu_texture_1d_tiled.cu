#include "matmul/implementation.hpp"
#include "utils/utils.h"

#include <cuda_runtime.h>

#include <memory>

using namespace utils;

namespace matmul {

namespace {

constexpr int TILE = 16;

cudaTextureObject_t create_linear_texture_object(const float *device_ptr,
                                                 std::size_t bytes) {
  cudaResourceDesc resource_desc{};
  resource_desc.resType = cudaResourceTypeLinear;
  resource_desc.res.linear.devPtr = const_cast<float *>(device_ptr);
  resource_desc.res.linear.desc = cudaCreateChannelDesc<float>();
  resource_desc.res.linear.sizeInBytes = bytes;

  cudaTextureDesc texture_desc{};
  texture_desc.readMode = cudaReadModeElementType;

  cudaTextureObject_t texture = 0;
  checkCuda(
      cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr),
      "cudaCreateTextureObject");
  return texture;
}

__global__ void matmul_texture_kernel(cudaTextureObject_t tex_a,
                                      cudaTextureObject_t tex_b, float *c,
                                      int n) {
  __shared__ float a_tile[TILE][TILE];
  __shared__ float b_tile[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  const int tile_count = (n + TILE - 1) / TILE;

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int a_col = tile_idx * TILE + threadIdx.x;
    const int b_row = tile_idx * TILE + threadIdx.y;

    if (row < n && a_col < n) {
      const int a_idx = row * n + a_col;
      a_tile[threadIdx.y][threadIdx.x] = tex1Dfetch<float>(tex_a, a_idx);
    } else {
      a_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (b_row < n && col < n) {
      const int b_idx = b_row * n + col;
      b_tile[threadIdx.y][threadIdx.x] = tex1Dfetch<float>(tex_b, b_idx);
    } else {
      b_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = acc;
  }
}
} // namespace

class GpuTexture1DTiledMatmul final : public MatmulImplementation {
public:
  const char *name() const override { return "gpu_texture_1d_tiled_cuda"; }
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

    const cudaTextureObject_t tex_a =
        create_linear_texture_object(d_a.get(), bytes);
    const cudaTextureObject_t tex_b =
        create_linear_texture_object(d_b.get(), bytes);

    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    matmul_texture_kernel<<<grid, block>>>(tex_a, tex_b, d_c.get(), n);
    checkCuda(cudaGetLastError(), "launch matmul_texture_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync matmul_texture_kernel");

    checkCuda(cudaDestroyTextureObject(tex_a),
              "cudaDestroyTextureObject tex_a");
    checkCuda(cudaDestroyTextureObject(tex_b),
              "cudaDestroyTextureObject tex_b");

    checkCuda(cudaMemcpy(c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy d_c->c");
  }
};

std::unique_ptr<MatmulImplementation> make_gpu_texture_1d_tiled() {
  return std::make_unique<GpuTexture1DTiledMatmul>();
}

} // namespace matmul