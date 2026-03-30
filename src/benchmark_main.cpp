#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "matmul/harness.hpp"
#include "matmul/registry.hpp"

namespace {

int parse_int_flag(int argc, char** argv, const std::string& name, int default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == name) {
            return std::atoi(argv[i + 1]);
        }
    }
    return default_value;
}

std::size_t parse_size_flag(int argc, char** argv, const std::string& name, std::size_t default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == name) {
            return static_cast<std::size_t>(std::strtoul(argv[i + 1], nullptr, 10));
        }
    }
    return default_value;
}

std::string parse_string_flag(int argc, char** argv, const std::string& name, const std::string& default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == name) {
            return std::string(argv[i + 1]);
        }
    }
    return default_value;
}

}  // namespace

int main(int argc, char** argv) {
    matmul::BenchmarkConfig config;
    config.benchmark_size = parse_size_flag(argc, argv, "--size", 1024);
    config.validation_size = parse_size_flag(argc, argv, "--test-size", 128);
    config.warmup_runs = parse_int_flag(argc, argv, "--warmup", 2);
    config.timed_runs = parse_int_flag(argc, argv, "--runs", 10);
    config.tolerance = 1e-3f;
    config.seed = 1234;

    const std::string output_path = parse_string_flag(argc, argv, "--json", "benchmark-results.json");

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

    matmul::BenchmarkRunner runner(config);
    std::vector<matmul::BenchmarkResult> results;
    results.reserve(implementations.size());

    std::cout << "Benchmark size: " << config.benchmark_size << "x" << config.benchmark_size << std::endl;
    std::cout << "Validation size: " << config.validation_size << "x" << config.validation_size << std::endl;
    std::cout << "Warmup runs: " << config.warmup_runs << ", timed runs: " << config.timed_runs << std::endl;

    for (const auto& impl : implementations) {
        matmul::BenchmarkResult result = runner.run(*impl, *reference);
        if (!result.validation_ok) {
            std::cerr << "[SKIP] " << result.implementation_name
                      << " failed validation (max_abs_diff=" << result.validation_max_abs_diff << ")"
                      << std::endl;
            results.push_back(result);
            continue;
        }

        std::cout << "[OK] " << result.implementation_name
                  << " mean=" << result.mean_ms << " ms"
                  << " gflops=" << result.gflops << std::endl;
        results.push_back(result);
    }

    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Unable to open output file: " << output_path << std::endl;
        return 1;
    }
    out << matmul::benchmark_results_to_json(config, results);
    std::cout << "Saved benchmark JSON to " << output_path << std::endl;

    bool optimized_valid = true;
    for (const auto& r : results) {
        if (r.optimized && !r.validation_ok) {
            optimized_valid = false;
        }
    }
    return optimized_valid ? 0 : 2;
}
