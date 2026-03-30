#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "matmul/harness.hpp"
#include "matmul/registry.hpp"

namespace {

std::size_t parse_size_arg(int argc, char** argv, std::size_t default_size) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--size") {
            return static_cast<std::size_t>(std::strtoul(argv[i + 1], nullptr, 10));
        }
    }
    return default_size;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t size = parse_size_arg(argc, argv, 128);
    constexpr float tolerance = 1e-3f;
    constexpr unsigned int seed = 1234;

    auto implementations = matmul::make_all_implementations();
    if (implementations.empty()) {
        std::cerr << "No implementations registered" << std::endl;
        return 1;
    }

    matmul::MatmulImplementation* reference = nullptr;
    for (const auto& impl : implementations) {
        if (std::string(impl->name()) == "cpu_naive") {
            reference = impl.get();
            break;
        }
    }

    if (reference == nullptr) {
        std::cerr << "Reference implementation cpu_naive not found" << std::endl;
        return 1;
    }

    matmul::Validator validator;
    bool all_ok = true;

    std::cout << "Validation size: " << size << "x" << size << std::endl;
    for (const auto& impl : implementations) {
        auto result = validator.validate_against_reference(*impl, *reference, size, tolerance, seed);
        std::cout << "[" << (result.ok ? "PASS" : "FAIL") << "] " << result.implementation_name
                  << " max_abs_diff=" << result.max_abs_diff << std::endl;
        all_ok = all_ok && result.ok;
    }

    return all_ok ? 0 : 2;
}
