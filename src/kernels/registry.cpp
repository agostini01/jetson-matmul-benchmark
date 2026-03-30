#include "matmul/registry.hpp"

namespace matmul {

std::vector<std::unique_ptr<MatmulImplementation>> make_all_implementations() {
    std::vector<std::unique_ptr<MatmulImplementation>> implementations;
    implementations.push_back(make_cpu_naive());
    implementations.push_back(make_gpu_naive());
    implementations.push_back(make_gpu_tiled());
    return implementations;
}

}  // namespace matmul
