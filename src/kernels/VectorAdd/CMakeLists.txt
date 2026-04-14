set(VECTOR_ADD_SOURCES
    # cpu_naive.cpp
    # cpu_optimized.cpp
    gpu_naive.cu
    # gpu_tiled.cu
    # gpu_cub.cu
    # gpu_cublas.cu
    # registry.cpp
)

# if(VECTOR_ADD_ENABLE_BLAS)
#     list(APPEND VECTOR_ADD_SOURCES cpu_blas.cpp)
# endif()

add_library(vector_add_kernels ${VECTOR_ADD_SOURCES})
target_include_directories(vector_add_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(vector_add_kernels PUBLIC CUDA::cudart CUDA::cublas)

# if(VECTOR_ADD_ENABLE_BLAS)
#     target_compile_definitions(vector_add_kernels PUBLIC VECTOR_ADD_ENABLE_BLAS)
#     target_link_libraries(vector_add_kernels PUBLIC ${BLAS_LIBRARIES})
#     if(BLAS_LINKER_FLAGS)
#         target_link_options(vector_add_kernels PUBLIC ${BLAS_LINKER_FLAGS})
#     endif()
# endif()

if(OPENMP_FOUND)
    target_link_libraries(vector_add_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()