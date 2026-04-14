set(HISTOGRAM_SOURCES
    # cpu_naive.cpp
    # cpu_optimized.cpp
    gpu_naive.cu
    # gpu_tiled.cu
    # gpu_cub.cu
    # gpu_cublas.cu
    # registry.cpp
)

# if(HISTOGRAM_ENABLE_BLAS)
#     list(APPEND HISTOGRAM_SOURCES cpu_blas.cpp)
# endif()

add_library(histogram_kernels ${HISTOGRAM_SOURCES})
target_include_directories(histogram_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(histogram_kernels PUBLIC CUDA::cudart CUDA::cublas)

# if(HISTOGRAM_ENABLE_BLAS)
#     target_compile_definitions(histogram_kernels PUBLIC HISTOGRAM_ENABLE_BLAS)
#     target_link_libraries(histogram_kernels PUBLIC ${BLAS_LIBRARIES})
#     if(BLAS_LINKER_FLAGS)
#         target_link_options(histogram_kernels PUBLIC ${BLAS_LINKER_FLAGS})
#     endif()
# endif()

if(OPENMP_FOUND)
    target_link_libraries(histogram_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()