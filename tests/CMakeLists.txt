add_executable(test_matmul test_main.cpp)
target_include_directories(test_matmul PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_matmul PRIVATE matmul_kernels)
