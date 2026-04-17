set(MATMUL_SOURCES
    cpu_naive.cpp
    cpu_optimized.cpp
    gpu_naive.cu
    gpu_tiled.cu
    gpu_cub.cu
    gpu_cublas.cu
    gpu_texture_1d_naive.cu
    gpu_texture_1d_tiled.cu
    gpu_texture_2d_tiled.cu
    gpu_texture_2d_naive.cu
    gpu_surface_2d_naive.cu
    gpu_surface_2d_tiled.cu
    gpu_tensorcore_bf16.cu
    gpu_tensorcore_tf32.cu
    registry.cpp
)

if(MATMUL_ENABLE_BLAS)
    list(APPEND MATMUL_SOURCES cpu_blas.cpp)
endif()

add_library(matmul_kernels ${MATMUL_SOURCES})
target_include_directories(matmul_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(matmul_kernels PUBLIC matmul_harness CUDA::cudart CUDA::cublas)

if(MATMUL_ENABLE_BLAS)
    target_compile_definitions(matmul_kernels PUBLIC MATMUL_ENABLE_BLAS)
    target_link_libraries(matmul_kernels PUBLIC ${BLAS_LIBRARIES})
    if(BLAS_LINKER_FLAGS)
        target_link_options(matmul_kernels PUBLIC ${BLAS_LINKER_FLAGS})
    endif()
endif()

if(OPENMP_FOUND)
    target_link_libraries(matmul_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()
