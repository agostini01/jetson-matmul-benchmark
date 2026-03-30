add_executable(bench_matmul benchmark_main.cpp)
target_include_directories(bench_matmul PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(bench_matmul PRIVATE matmul_kernels)
