#include "matmul/harness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

#include "matmul/matrix.hpp"

namespace matmul {

ValidationResult Validator::validate_against_reference(
    MatmulImplementation& candidate,
    MatmulImplementation& reference,
    std::size_t size,
    float tolerance,
    unsigned int seed) const {
    Matrix a(size);
    Matrix b(size);
    Matrix c_reference(size);
    Matrix c_candidate(size);

    a.fill_random(seed);
    b.fill_random(seed + 1U);
    c_reference.fill_zero();
    c_candidate.fill_zero();

    reference.multiply(a, b, c_reference);
    candidate.multiply(a, b, c_candidate);

    float max_abs_diff = 0.0f;
    for (std::size_t i = 0; i < size * size; ++i) {
        const float diff = std::fabs(c_reference.data()[i] - c_candidate.data()[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
    }

    ValidationResult result;
    result.implementation_name = candidate.name();
    result.ok = max_abs_diff <= tolerance;
    result.max_abs_diff = max_abs_diff;
    result.details = result.ok ? "ok" : "max abs diff exceeds tolerance";
    return result;
}

BenchmarkResult BenchmarkRunner::run(
    MatmulImplementation& implementation,
    MatmulImplementation& reference) const {
    Validator validator;
    const ValidationResult validation = validator.validate_against_reference(
        implementation,
        reference,
        config_.validation_size,
        config_.tolerance,
        config_.seed);

    BenchmarkResult result;
    result.implementation_name = implementation.name();
    result.optimized = implementation.is_optimized();
    result.validation_ok = validation.ok;
    result.validation_max_abs_diff = validation.max_abs_diff;

    if (!validation.ok) {
        return result;
    }

    Matrix a(config_.benchmark_size);
    Matrix b(config_.benchmark_size);
    Matrix c(config_.benchmark_size);

    a.fill_random(config_.seed);
    b.fill_random(config_.seed + 1U);

    for (int i = 0; i < config_.warmup_runs; ++i) {
        c.fill_zero();
        implementation.multiply(a, b, c);
    }

    result.run_times_ms.reserve(static_cast<std::size_t>(config_.timed_runs));
    for (int i = 0; i < config_.timed_runs; ++i) {
        c.fill_zero();
        const auto start = std::chrono::high_resolution_clock::now();
        implementation.multiply(a, b, c);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        result.run_times_ms.push_back(elapsed);
    }

    result.min_ms = *std::min_element(result.run_times_ms.begin(), result.run_times_ms.end());
    result.max_ms = *std::max_element(result.run_times_ms.begin(), result.run_times_ms.end());
    result.mean_ms = std::accumulate(result.run_times_ms.begin(), result.run_times_ms.end(), 0.0) /
                     static_cast<double>(result.run_times_ms.size());

    double variance = 0.0;
    for (const double sample : result.run_times_ms) {
        const double delta = sample - result.mean_ms;
        variance += delta * delta;
    }
    variance /= static_cast<double>(result.run_times_ms.size());
    result.stddev_ms = std::sqrt(variance);

    const double n = static_cast<double>(config_.benchmark_size);
    const double ops = 2.0 * n * n * n;
    result.gflops = (ops / (result.mean_ms / 1000.0)) / 1e9;

    return result;
}

std::string benchmark_results_to_json(
    const BenchmarkConfig& config,
    const std::vector<BenchmarkResult>& results) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"benchmark_size\": " << config.benchmark_size << ",\n";
    out << "    \"validation_size\": " << config.validation_size << ",\n";
    out << "    \"warmup_runs\": " << config.warmup_runs << ",\n";
    out << "    \"timed_runs\": " << config.timed_runs << ",\n";
    out << "    \"tolerance\": " << config.tolerance << "\n";
    out << "  },\n";
    out << "  \"results\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const BenchmarkResult& r = results[i];
        out << "    {\n";
        out << "      \"name\": \"" << r.implementation_name << "\",\n";
        out << "      \"optimized\": " << (r.optimized ? "true" : "false") << ",\n";
        out << "      \"validation_ok\": " << (r.validation_ok ? "true" : "false") << ",\n";
        out << "      \"validation_max_abs_diff\": " << r.validation_max_abs_diff << ",\n";
        out << "      \"min_ms\": " << r.min_ms << ",\n";
        out << "      \"max_ms\": " << r.max_ms << ",\n";
        out << "      \"mean_ms\": " << r.mean_ms << ",\n";
        out << "      \"stddev_ms\": " << r.stddev_ms << ",\n";
        out << "      \"gflops\": " << r.gflops << ",\n";
        out << "      \"runs_ms\": [";
        for (std::size_t k = 0; k < r.run_times_ms.size(); ++k) {
            out << r.run_times_ms[k];
            if (k + 1 < r.run_times_ms.size()) {
                out << ", ";
            }
        }
        out << "]\n";
        out << "    }";
        if (i + 1 < results.size()) {
            out << ",";
        }
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
    return out.str();
}

}  // namespace matmul
