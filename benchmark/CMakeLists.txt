# Copyright 2019 JD.com Inc. JD AI

add_executable(dabnn_benchmark benchmark.cu)
target_link_libraries(dabnn_benchmark
    dabnn
    benchmark_main
    ${CUDA_LIBRARIES}
)
add_executable(benchmark_single_model benchmark_single_model.cpp)
target_link_libraries(benchmark_single_model
    dabnn
    benchmark_main
    ${CUDA_LIBRARIES}
)

set_property(TARGET dabnn_benchmark PROPERTY CUDA_ARCHITECTURES OFF)