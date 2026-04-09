#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "matmul/implementation.hpp"

namespace matmul {

struct ValidationResult {
  std::string implementation_name;
  bool ok = false;
  float max_abs_diff = 0.0f;
  std::string details;
};

class Validator {
public:
  ValidationResult validate_against_reference(MatmulImplementation &candidate,
                                              MatmulImplementation &reference,
                                              std::size_t size, float tolerance,
                                              unsigned int seed) const;
};

struct BenchmarkConfig {
  std::size_t benchmark_size = 1024;
  std::size_t validation_size = 128;
  int warmup_runs = 2;
  int timed_runs = 10;
  float tolerance = 1e-3f;
  unsigned int seed = 1234;
};

struct BenchmarkResult {
  std::string implementation_name;
  bool optimized = false;
  bool validation_ok = false;
  float validation_max_abs_diff = 0.0f;
  std::vector<double> run_times_ms;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
  double gflops = 0.0;
};

class BenchmarkRunner {
public:
  explicit BenchmarkRunner(BenchmarkConfig config) : config_(config) {}

  BenchmarkResult run(MatmulImplementation &implementation,
                      MatmulImplementation &reference) const;

private:
  BenchmarkConfig config_;
};

std::string
benchmark_results_to_json(const BenchmarkConfig &config,
                          const std::vector<BenchmarkResult> &results);

} // namespace matmul
