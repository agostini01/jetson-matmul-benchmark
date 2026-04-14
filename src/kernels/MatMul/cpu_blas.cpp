#include "matmul/implementation.hpp"

#include <memory>

extern "C" {
#include <cblas.h>
}

namespace matmul {

class CpuBlasMatmul final : public MatmulImplementation {
public:
  const char *name() const override { return "cpu_blas"; }
  bool is_optimized() const override { return true; }

  void multiply(const Matrix &a, const Matrix &b, Matrix &c) override {
    const int n = static_cast<int>(a.size());

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f,
                a.data(), n, b.data(), n, 0.0f, c.data(), n);
  }
};

std::unique_ptr<MatmulImplementation> make_cpu_blas() {
  return std::make_unique<CpuBlasMatmul>();
}

} // namespace matmul
