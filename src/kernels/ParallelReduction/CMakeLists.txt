set(PARALLEL_REDUCTION_SOURCES
    # cpu_naive.cpp
    # cpu_optimized.cpp
    gpu_naive.cu
    # gpu_tiled.cu
    # gpu_cub.cu
    # gpu_cublas.cu
    # registry.cpp
)

# if(PARALLEL_REDUCTION_ENABLE_BLAS)
#     list(APPEND PARALLEL_REDUCTION_SOURCES cpu_blas.cpp)
# endif()

add_library(parallel_reduction_kernels ${PARALLEL_REDUCTION_SOURCES})
target_include_directories(parallel_reduction_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(parallel_reduction_kernels PUBLIC parallel_reduction_harness CUDA::cudart CUDA::cublas)

# if(PARALLEL_REDUCTION_ENABLE_BLAS)
#     target_compile_definitions(parallel_reduction_kernels PUBLIC PARALLEL_REDUCTION_ENABLE_BLAS)
#     target_link_libraries(parallel_reduction_kernels PUBLIC ${BLAS_LIBRARIES})
#     if(BLAS_LINKER_FLAGS)
#         target_link_options(parallel_reduction_kernels PUBLIC ${BLAS_LINKER_FLAGS})
#     endif()
# endif()

if(OPENMP_FOUND)
    target_link_libraries(parallel_reduction_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()
