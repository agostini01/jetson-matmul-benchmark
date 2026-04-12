#include "matmul/registry.hpp"

namespace matmul {

std::vector<std::unique_ptr<MatmulImplementation>> make_all_implementations() {
  std::vector<std::unique_ptr<MatmulImplementation>> implementations;
  implementations.push_back(make_cpu_naive());
  implementations.push_back(make_cpu_optimized());
#ifdef MATMUL_ENABLE_BLAS
  implementations.push_back(make_cpu_blas());
#endif
  implementations.push_back(make_gpu_naive());
  // implementations.push_back(make_gpu_cub());
  implementations.push_back(make_gpu_tiled());
  implementations.push_back(make_gpu_cublas());
  return implementations;
}

} // namespace matmul
