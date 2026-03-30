#pragma once

#include <string>

#include "matmul/matrix.hpp"

namespace matmul {

class MatmulImplementation {
public:
    virtual ~MatmulImplementation() = default;

    virtual const char* name() const = 0;
    virtual bool is_optimized() const = 0;
    virtual void multiply(const Matrix& a, const Matrix& b, Matrix& c) = 0;
};

}  // namespace matmul
