// Copyright 2019 JD.com Inc. JD AI

#include "Affine.h"

namespace bnn {

template<typename T>
struct affine_functor {
    __host__ __device__
    T operator()(T input, thrust::tuple<T,T> p) {
        return input * thrust::get<0>(p) + thrust::get<1>(p);
    }
};

/**
 * per channel affine, x = a * x + b
 */
inline void affine_inplace(bnn::Mat &data, bnn::Mat &a,
                           bnn::Mat &b) {
    BNN_ASSERT(a.total() == b.total(), "");
    if (a.data_type != b.data_type || a.data_type != data.data_type) {
        throw std::invalid_argument("Mismatch datatype");
    } else if (a.data_type == DataType::Float) {
        affine_functor<float> func;
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(a.begin<float>(), b.begin<float>()));
        auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(a.end<float>(), b.end<float>()));
        auto zip_cyc_begin = cyclic_iterator<thrust::zip_iterator<thrust::tuple<float*, float*>>>(zip_begin, a.total());
        auto zip_cyc_end = cyclic_iterator<thrust::zip_iterator<thrust::tuple<float*, float*>>>(zip_end, a.total());
        thrust::transform(thrust::host, data.begin<float>(), data.end<float>(), zip_cyc_begin, data.begin<float>(), func); // _TODO: Move this function into device after net impl
    } else if (b.data_type == DataType::Bit) {
        affine_functor<uint64_t> func;
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(a.begin<uint64_t>(), b.begin<uint64_t>()));
        auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(a.end<uint64_t>(), b.end<uint64_t>()));
        auto zip_cyc_begin = cyclic_iterator<thrust::zip_iterator<thrust::tuple<uint64_t*, uint64_t*>>>(zip_begin, a.total());
        auto zip_cyc_end = cyclic_iterator<thrust::zip_iterator<thrust::tuple<uint64_t*, uint64_t*>>>(zip_end, a.total());
        thrust::transform(thrust::device, data.begin<uint64_t>(), data.end<uint64_t>(), zip_cyc_begin, data.begin<uint64_t>(), func);
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
    cudaDeviceSynchronize();
}

inline void affine(bnn::Mat &data, bnn::Mat &a, bnn::Mat &b,
                   bnn::Mat &output) {
    BNN_ASSERT(a.total() == b.total(), "");
    if (a.data_type != b.data_type || a.data_type != data.data_type) {
        throw std::invalid_argument("Mismatch datatype");
    } else if (a.data_type == DataType::Float) {
        affine_functor<float> func;
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(a.begin<float>(), b.begin<float>()));
        auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(a.end<float>(), b.end<float>()));
        auto zip_cyc_begin = cyclic_iterator<thrust::zip_iterator<thrust::tuple<float*, float*>>>(zip_begin, a.total());
        auto zip_cyc_end = cyclic_iterator<thrust::zip_iterator<thrust::tuple<float*, float*>>>(zip_end, a.total());
        thrust::transform(thrust::device, data.begin<float>(), data.end<float>(), zip_cyc_begin, output.begin<float>(), func);
    } else if (b.data_type == DataType::Bit) {
        affine_functor<uint64_t> func;
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(a.begin<uint64_t>(), b.begin<uint64_t>()));
        auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(a.end<uint64_t>(), b.end<uint64_t>()));
        auto zip_cyc_begin = cyclic_iterator<thrust::zip_iterator<thrust::tuple<uint64_t*, uint64_t*>>>(zip_begin, a.total());
        auto zip_cyc_end = cyclic_iterator<thrust::zip_iterator<thrust::tuple<uint64_t*, uint64_t*>>>(zip_end, a.total());
        thrust::transform(thrust::device, data.begin<uint64_t>(), data.end<uint64_t>(), zip_cyc_begin, output.begin<uint64_t>(), func);
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
    cudaDeviceSynchronize();
}

void Affine::forward_impl() const {
#ifdef BNN_CHECK_CONSISTENCY
    affine(*data_mat, *a_mat, *b_mat, *output_mat);
#else
    affine_inplace(*data_mat, *a_mat, *b_mat);
#endif
}

}  // namespace bnn
