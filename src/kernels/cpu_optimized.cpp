#include "matmul/implementation.hpp"

#include <algorithm>
#include <memory>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace matmul {

class CpuOptimizedMatmul final : public MatmulImplementation {
public:
    const char* name() const override {
#ifdef USE_OPENMP
        return "cpu_optimized_openmp";
#else
        return "cpu_optimized_tiled";
#endif
    }

    bool is_optimized() const override { return true; }

    void multiply(const Matrix& a, const Matrix& b, Matrix& c) override {
        const std::size_t n = a.size();
        constexpr std::size_t tile = 32;

#ifdef USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        for (std::size_t ii = 0; ii < n; ii += tile) {
            for (std::size_t jj = 0; jj < n; jj += tile) {
                for (std::size_t kk = 0; kk < n; kk += tile) {
                    const std::size_t i_max = std::min(ii + tile, n);
                    const std::size_t j_max = std::min(jj + tile, n);
                    const std::size_t k_max = std::min(kk + tile, n);
                    for (std::size_t i = ii; i < i_max; ++i) {
                        for (std::size_t j = jj; j < j_max; ++j) {
                            float acc = (kk == 0) ? 0.0f : c(i, j);
                            for (std::size_t k = kk; k < k_max; ++k) {
                                acc += a(i, k) * b(k, j);
                            }
                            c(i, j) = acc;
                        }
                    }
                }
            }
        }
#else
        for (std::size_t ii = 0; ii < n; ii += tile) {
            for (std::size_t jj = 0; jj < n; jj += tile) {
                for (std::size_t kk = 0; kk < n; kk += tile) {
                    const std::size_t i_max = std::min(ii + tile, n);
                    const std::size_t j_max = std::min(jj + tile, n);
                    const std::size_t k_max = std::min(kk + tile, n);
                    for (std::size_t i = ii; i < i_max; ++i) {
                        for (std::size_t j = jj; j < j_max; ++j) {
                            float acc = (kk == 0) ? 0.0f : c(i, j);
                            for (std::size_t k = kk; k < k_max; ++k) {
                                acc += a(i, k) * b(k, j);
                            }
                            c(i, j) = acc;
                        }
                    }
                }
            }
        }
#endif
    }
};

std::unique_ptr<MatmulImplementation> make_cpu_optimized() {
    return std::make_unique<CpuOptimizedMatmul>();
}

}  // namespace matmul
