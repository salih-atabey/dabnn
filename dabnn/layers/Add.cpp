// Copyright 2019 JD.com Inc. JD AI

#include "Add.h"

namespace bnn {

// Functor for adding a value from all pixels in an image
template<typename T>
struct add_functor {
    __host__ __device__
    T operator()(T a, T b) {
        return a + b;
    }
};

inline void add_inplace(bnn::Mat &a, bnn::Mat &b) {
    if (a.data_type != b.data_type) {
        throw std::invalid_argument("Mismatch datatype");
    } else if (a.data_type == DataType::Float) {
        add_functor<float> func;
        thrust::transform(a.begin<float>(), a.end<float>(), b.begin<float>(), a.begin<float>(), func);
    } else if (b.data_type == DataType::Bit) {
        add_functor<uint64_t> func;
        thrust::transform(a.begin<uint64_t>(), a.end<uint64_t>(), b.begin<uint64_t>(), a.begin<uint64_t>(), func);
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
    cudaDeviceSynchronize();
}

inline void add(bnn::Mat &a, bnn::Mat &b, bnn::Mat &c) {
    if (a.data_type != b.data_type) {
        throw std::invalid_argument("Mismatch datatype");
    } else if (a.data_type == DataType::Float) {
        add_functor<float> func;
        thrust::transform(a.begin<float>(), a.end<float>(), b.begin<float>(), c.begin<float>(), func);
    } else if (b.data_type == DataType::Bit) {
        add_functor<uint64_t> func;
        thrust::transform(a.begin<uint64_t>(), a.end<uint64_t>(), b.begin<uint64_t>(), c.begin<uint64_t>(), func);
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
}

void Add::forward_impl() const {
#ifdef BNN_CHECK_CONSISTENCY
    add(*input1_mat, *input2_mat, *output_mat);
#else
    add_inplace(*input1_mat, *input2_mat);
#endif
}

}  // namespace bnn
