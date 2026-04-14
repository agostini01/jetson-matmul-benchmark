set(PREFIX_SUM_SOURCES
    # cpu_naive.cpp
    # cpu_optimized.cpp
    gpu_naive.cu
    # gpu_tiled.cu
    # gpu_cub.cu
    # gpu_cublas.cu
    # registry.cpp
)

# if(PREFIX_SUM_ENABLE_BLAS)
#     list(APPEND PREFIX_SUM_SOURCES cpu_blas.cpp)
# endif()

add_library(prefix_sum_kernels ${PREFIX_SUM_SOURCES})
target_include_directories(prefix_sum_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(prefix_sum_kernels PUBLIC prefix_sum_harness CUDA::cudart CUDA::cublas)

# if(PREFIX_SUM_ENABLE_BLAS)
#     target_compile_definitions(prefix_sum_kernels PUBLIC PREFIX_SUM_ENABLE_BLAS)
#     target_link_libraries(prefix_sum_kernels PUBLIC ${BLAS_LIBRARIES})
#     if(BLAS_LINKER_FLAGS)
#         target_link_options(prefix_sum_kernels PUBLIC ${BLAS_LINKER_FLAGS})
#     endif()
# endif()

if(OPENMP_FOUND)
    target_link_libraries(prefix_sum_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()
