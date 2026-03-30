#include "matmul/implementation.hpp"

#include <memory>

namespace matmul {

class CpuNaiveMatmul final : public MatmulImplementation {
public:
    const char* name() const override { return "cpu_naive"; }
    bool is_optimized() const override { return false; }

    void multiply(const Matrix& a, const Matrix& b, Matrix& c) override {
        const std::size_t n = a.size();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < n; ++k) {
                    acc += a(i, k) * b(k, j);
                }
                c(i, j) = acc;
            }
        }
    }
};

std::unique_ptr<MatmulImplementation> make_cpu_naive() {
    return std::make_unique<CpuNaiveMatmul>();
}

}  // namespace matmul
