CXX ?= g++
NVCC ?= nvcc

BUILD_DIR := build
BIN_DIR := $(BUILD_DIR)/bin
OBJ_DIR := $(BUILD_DIR)/obj

CXX_STD := -std=c++17
CXX_WARN := -Wall -Wextra -Wpedantic
CXX_OPT := -O3
INCLUDES := -Iinclude

OPENMP_AVAILABLE := $(shell cat /tmp/omp_probe.cpp 2>/dev/null >/dev/null; printf '#include <omp.h>\nint main(){return 0;}\n' > /tmp/omp_probe.cpp && $(CXX) -fopenmp /tmp/omp_probe.cpp -o /tmp/omp_probe.bin >/dev/null 2>&1 && echo 1 || echo 0)

CPPFLAGS := $(INCLUDES)
CXXFLAGS := $(CXX_STD) $(CXX_WARN) $(CXX_OPT)
NVCCFLAGS := -std=c++17 -O3 $(INCLUDES)
LDFLAGS :=
LDLIBS :=

ifeq ($(OPENMP_AVAILABLE),1)
    CPPFLAGS += -DUSE_OPENMP
    CXXFLAGS += -fopenmp
    NVCCFLAGS += -Xcompiler -fopenmp
	LDLIBS += -Xcompiler -fopenmp
endif

COMMON_CPP_SRCS := src/harness.cpp src/registry.cpp src/cpu_naive.cpp
CUDA_SRCS := src/gpu_naive.cu

TEST_SRCS := src/test_main.cpp
BENCH_SRCS := src/benchmark_main.cpp

COMMON_CPP_OBJS := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_CPP_SRCS))
CUDA_OBJS := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(CUDA_SRCS))
TEST_OBJS := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(TEST_SRCS))
BENCH_OBJS := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(BENCH_SRCS))

TEST_BIN := $(BIN_DIR)/test_matmul
BENCH_BIN := $(BIN_DIR)/bench_matmul

BENCH_SIZE ?= 1024
TEST_SIZE ?= 128
WARMUP ?= 2
RUNS ?= 10
JSON ?= benchmark-results.json

.PHONY: all info dirs clean test benchmark

all: info $(TEST_BIN) $(BENCH_BIN)

info:
	@echo "OpenMP available: $(OPENMP_AVAILABLE)"

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

$(OBJ_DIR)/%.o: src/%.cpp | dirs
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: src/%.cu | dirs
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(TEST_BIN): $(COMMON_CPP_OBJS) $(CUDA_OBJS) $(TEST_OBJS)
	$(NVCC) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BENCH_BIN): $(COMMON_CPP_OBJS) $(CUDA_OBJS) $(BENCH_OBJS)
	$(NVCC) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test: $(TEST_BIN)
	$(TEST_BIN) --size $(TEST_SIZE)

benchmark: $(BENCH_BIN)
	$(BENCH_BIN) --size $(BENCH_SIZE) --test-size $(TEST_SIZE) --warmup $(WARMUP) --runs $(RUNS) --json $(JSON)

clean:
	rm -rf $(BUILD_DIR) benchmark-results*.json /tmp/omp_probe.cpp /tmp/omp_probe.bin
