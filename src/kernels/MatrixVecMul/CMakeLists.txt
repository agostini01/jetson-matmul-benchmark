set(MATRIX_VEC_MUL_SOURCES
    # cpu_naive.cpp
    # cpu_optimized.cpp
    gpu_naive.cu
    # gpu_tiled.cu
    # gpu_cub.cu
    # gpu_cublas.cu
    # registry.cpp
)

# if(MATRIX_VEC_MUL_ENABLE_BLAS)
#     list(APPEND MATRIX_VEC_MUL_SOURCES cpu_blas.cpp)
# endif()

add_library(matrix_vec_mul_kernels ${MATRIX_VEC_MUL_SOURCES})
target_include_directories(matrix_vec_mul_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(matrix_vec_mul_kernels PUBLIC CUDA::cudart CUDA::cublas)

# if(MATRIX_VEC_MUL_ENABLE_BLAS)
#     target_compile_definitions(matrix_vec_mul_kernels PUBLIC MATRIX_VEC_MUL_ENABLE_BLAS)
#     target_link_libraries(matrix_vec_mul_kernels PUBLIC ${BLAS_LIBRARIES})
#     if(BLAS_LINKER_FLAGS)
#         target_link_options(matrix_vec_mul_kernels PUBLIC ${BLAS_LINKER_FLAGS})
#     endif()
# endif()

if(OPENMP_FOUND)
    target_link_libraries(matrix_vec_mul_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()