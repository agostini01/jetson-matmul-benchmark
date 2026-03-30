#pragma once

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace matmul {

class Matrix {
public:
    Matrix() = default;

    explicit Matrix(std::size_t size)
        : size_(size), data_(size * size, 0.0f) {}

    std::size_t size() const { return size_; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    float& operator()(std::size_t row, std::size_t col) {
        return data_[row * size_ + col];
    }

    float operator()(std::size_t row, std::size_t col) const {
        return data_[row * size_ + col];
    }

    void fill_random(unsigned int seed, float min_value = -1.0f, float max_value = 1.0f) {
        if (size_ == 0) {
            throw std::runtime_error("Matrix size must be greater than zero");
        }
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(min_value, max_value);
        for (float& value : data_) {
            value = dist(rng);
        }
    }

    void fill_zero() {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }

private:
    std::size_t size_ = 0;
    std::vector<float> data_;
};

}  // namespace matmul