#pragma once

#include <memory>
#include <vector>

#include "matmul/implementation.hpp"

namespace matmul {

std::unique_ptr<MatmulImplementation> make_cpu_naive();
std::unique_ptr<MatmulImplementation> make_cpu_optimized();
std::unique_ptr<MatmulImplementation> make_cpu_blas();
std::unique_ptr<MatmulImplementation> make_gpu_naive();
std::unique_ptr<MatmulImplementation> make_gpu_cub();
std::unique_ptr<MatmulImplementation> make_gpu_cublas();
std::unique_ptr<MatmulImplementation> make_gpu_tiled();

std::vector<std::unique_ptr<MatmulImplementation>> make_all_implementations();

} // namespace matmul
