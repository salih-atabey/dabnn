# Copyright 2019 JD.com Inc. JD AI

CUDA_ADD_LIBRARY(dabnn SHARED
    allocator.cpp
    allocator.h
    mat.h
    bconv.h
    bitpack.h
    net.cu
    im2col.h
    fconv.h
    layers/FloatConv.cpp
    layers/FloatConv.h
    layers/BinConv.cu
    layers/BinConv.h
    layers/MaxPool.cu
    layers/MaxPool.h
    layers/Affine.cu
    layers/Affine.h
    layers/AvePool.cu
    layers/AvePool.h
    layers/Add.cu
    layers/Add.h
    layers/Concat.cpp
    layers/Concat.h
    layers/Relu.cpp
    layers/Relu.h
    layers/Binarize.cu
    layers/Binarize.h
    layers/Shuffle.cpp
    layers/Shuffle.h
    layers/Split.cpp
    layers/Split.h
    layers/PRelu.cpp
    layers/PRelu.h
    layer.h
    layer.cpp
    ${PROJECT_SOURCE_DIR}/common/Shaper.cpp
    ${PROJECT_SOURCE_DIR}/common/Shaper.h
    OPTIONS -arch compute_61 -code sm_61
    )
target_include_directories(dabnn
    PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}
    )
target_include_directories(dabnn
    SYSTEM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/eigen
    )
target_link_libraries(dabnn
    glog::glog
    flatbuffers
    ${CUDA_LIBRARIES}
    )

if (${BNN_BUILD_JNI})
    add_subdirectory(jni)
endif()

if (${BNN_CHECK_CONSISTENCY})
    target_compile_definitions(
        dabnn
        PRIVATE
        BNN_CHECK_CONSISTENCY
        )
endif()

treat_warnings_as_errors(dabnn)

set_property(TARGET dabnn PROPERTY CUDA_ARCHITECTURES OFF)
