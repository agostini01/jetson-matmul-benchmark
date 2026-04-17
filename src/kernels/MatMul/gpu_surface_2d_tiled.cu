#include "matmul/implementation.hpp"
#include "utils/utils.h"

#include <cuda_runtime.h>

#include <memory>

using namespace utils;

namespace matmul {

namespace {

constexpr int TILE = 16;

cudaArray_t create_surface_array_2d(int width, int height) {
  cudaArray_t array = nullptr;
  const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  checkCuda(cudaMallocArray(
                &array, &channel_desc, static_cast<std::size_t>(width),
                static_cast<std::size_t>(height), cudaArraySurfaceLoadStore),
            "cudaMallocArray");
  return array;
}

cudaSurfaceObject_t create_surface_object(cudaArray_t array) {
  cudaResourceDesc resource_desc{};
  resource_desc.resType = cudaResourceTypeArray;
  resource_desc.res.array.array = array;

  cudaSurfaceObject_t surface = 0;
  checkCuda(cudaCreateSurfaceObject(&surface, &resource_desc),
            "cudaCreateSurfaceObject");
  return surface;
}

__global__ void matmul_surface_2d_tiled_kernel(cudaSurfaceObject_t surf_a,
                                               cudaSurfaceObject_t surf_b,
                                               cudaSurfaceObject_t surf_c,
                                               int n) {
  __shared__ float a_tile[TILE][TILE];
  __shared__ float b_tile[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  const int tile_count = (n + TILE - 1) / TILE;

  float acc = 0.0f;

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int a_col = tile_idx * TILE + threadIdx.x;
    const int b_row = tile_idx * TILE + threadIdx.y;

    float a_value = 0.0f;
    float b_value = 0.0f;

    if (row < n && a_col < n) {
      surf2Dread(&a_value, surf_a, a_col * static_cast<int>(sizeof(float)), row,
                 cudaBoundaryModeZero);
    }

    if (b_row < n && col < n) {
      surf2Dread(&b_value, surf_b, col * static_cast<int>(sizeof(float)), b_row,
                 cudaBoundaryModeZero);
    }

    a_tile[threadIdx.y][threadIdx.x] = a_value;
    b_tile[threadIdx.y][threadIdx.x] = b_value;

    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      acc += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    surf2Dwrite(acc, surf_c, col * static_cast<int>(sizeof(float)), row,
                cudaBoundaryModeTrap);
  }
}

} // namespace

class GpuSurface2DTiledMatmul final : public MatmulImplementation {
public:
  const char *name() const override { return "gpu_surface_2d_tiled_cuda"; }
  bool is_optimized() const override { return true; }

  void multiply(const Matrix &a, const Matrix &b, Matrix &c) override {
    const int n = static_cast<int>(a.size());
    const std::size_t row_bytes = static_cast<std::size_t>(n) * sizeof(float);

    const cudaArray_t a_array = create_surface_array_2d(n, n);
    const cudaArray_t b_array = create_surface_array_2d(n, n);
    const cudaArray_t c_array = create_surface_array_2d(n, n);

    std::shared_ptr<cudaArray> a_array_handle(a_array, [](cudaArray *array) {
      if (array) {
        cudaFreeArray(array);
      }
    });
    std::shared_ptr<cudaArray> b_array_handle(b_array, [](cudaArray *array) {
      if (array) {
        cudaFreeArray(array);
      }
    });
    std::shared_ptr<cudaArray> c_array_handle(c_array, [](cudaArray *array) {
      if (array) {
        cudaFreeArray(array);
      }
    });

    checkCuda(cudaMemcpy2DToArray(a_array, 0, 0, a.data(), row_bytes, row_bytes,
                                  static_cast<std::size_t>(n),
                                  cudaMemcpyHostToDevice),
              "cudaMemcpy2DToArray a");
    checkCuda(cudaMemcpy2DToArray(b_array, 0, 0, b.data(), row_bytes, row_bytes,
                                  static_cast<std::size_t>(n),
                                  cudaMemcpyHostToDevice),
              "cudaMemcpy2DToArray b");

    const cudaSurfaceObject_t surf_a = create_surface_object(a_array);
    const cudaSurfaceObject_t surf_b = create_surface_object(b_array);
    const cudaSurfaceObject_t surf_c = create_surface_object(c_array);

    const dim3 block(TILE, TILE);
    const dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    matmul_surface_2d_tiled_kernel<<<grid, block>>>(surf_a, surf_b, surf_c, n);
    checkCuda(cudaGetLastError(), "launch matmul_surface_2d_tiled_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync matmul_surface_2d_tiled_kernel");

    checkCuda(cudaDestroySurfaceObject(surf_a),
              "cudaDestroySurfaceObject surf_a");
    checkCuda(cudaDestroySurfaceObject(surf_b),
              "cudaDestroySurfaceObject surf_b");
    checkCuda(cudaDestroySurfaceObject(surf_c),
              "cudaDestroySurfaceObject surf_c");

    checkCuda(cudaMemcpy2DFromArray(c.data(), row_bytes, c_array, 0, 0,
                                    row_bytes, static_cast<std::size_t>(n),
                                    cudaMemcpyDeviceToHost),
              "cudaMemcpy2DFromArray c");
  }
};

std::unique_ptr<MatmulImplementation> make_gpu_surface_2d_tiled() {
  return std::make_unique<GpuSurface2DTiledMatmul>();
}

} // namespace matmul
