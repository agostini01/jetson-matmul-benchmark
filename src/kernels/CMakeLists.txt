set(KERNELS_SOURCES
    cpu_naive.cpp
    cpu_optimized.cpp
    gpu_naive.cu
    gpu_tiled.cu
    registry.cpp
)

add_library(matmul_kernels ${KERNELS_SOURCES})
target_include_directories(matmul_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(matmul_kernels PUBLIC matmul_harness CUDA::cudart)

if(OPENMP_FOUND)
    target_link_libraries(matmul_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()
