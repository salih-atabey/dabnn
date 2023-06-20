// Copyright 2019 JD.com Inc. JD AI

#include "AvePool.h"

#include <dabnn/net.h>
#include <dabnn/pad.h>

namespace bnn {

template<typename T>
struct avepool_functor {
    const T* input;
    const size_t pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, output_w, input_hstep;
    const int input_h, input_w, input_c;

    avepool_functor(const T* _input, const size_t _pad_h, const size_t _pad_w,
           const size_t _stride_h, const size_t _stride_w, const size_t _kernel_h,
           const size_t _kernel_w, const int _output_w, const int _input_h,
           const int _input_w, const int _input_c, const size_t _input_hstep)
        : input(_input), pad_h(_pad_h), pad_w(_pad_w), stride_h(_stride_h),
          stride_w(_stride_w), kernel_h(_kernel_h), kernel_w(_kernel_w),
          output_w(_output_w), input_h(_input_h), input_w(_input_w), input_c(_input_c), input_hstep(_input_hstep) {}

    __host__ __device__
    T operator()(const int output_idx) const {
        const int output_y = output_idx / output_w;
        const int output_x = output_idx % output_w;

        int input_y = output_y * stride_h - pad_h;
        int input_x = output_x * stride_w - pad_w;

        size_t n = 0;
        T sum = 0;

        for (int kh = 0; kh < kernel_h; kh++) {
            int y = input_y + kh;
            if (y >= 0 && y < input_h) {
                // const T* input_ptr = input.point<T>(y, 0);
                const T* input_ptr = input + y * input_hstep;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int x = input_x + kw;
                    if (x >= 0 && x < input_w) {
                        const auto val = input_ptr[x * input_c];
                        sum += val;
                        n++;
                    }
                }
            }
        }

        return sum / n;
    }
};


#ifdef __ARM_NEON
void ave_pool_2x2_s2(const bnn::Mat &input, bnn::Mat &output) {
    FORZ(h, output.h) {
        FORZ(w, output.w) {
            const float *ptr0 = input.point<float>(h * 2 + 0, w * 2 + 0);
            const float *ptr1 = input.point<float>(h * 2 + 0, w * 2 + 1);
            const float *ptr2 = input.point<float>(h * 2 + 1, w * 2 + 0);
            const float *ptr3 = input.point<float>(h * 2 + 1, w * 2 + 1);
            float *output_ptr = output.point<float>(h, w);
            size_t nn = input.c >> 2;
#ifdef __aarch64__
            asm volatile(
                "fmov   s30, #4.0               \n"
                "dup    v30.4s, v30.s[0]        \n"
                "0:     \n"
                "ld1    {v0.4s}, [%0], #16      \n"
                "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v1.4s}, [%1], #16      \n"
                "prfm   pldl1keep, [%1, #128]   \n"
                "ld1    {v2.4s}, [%2], #16      \n"
                "prfm   pldl1keep, [%2, #128]   \n"
                "ld1    {v3.4s}, [%3], #16      \n"
                "prfm   pldl1keep, [%3, #128]   \n"
                "fadd   v0.4s, v0.4s, v1.4s     \n"
                "fadd   v2.4s, v2.4s, v3.4s     \n"
                "fadd   v0.4s, v0.4s, v2.4s     \n"
                "fdiv   v0.4s, v0.4s, v30.4s  \n"
                "subs   %5, %5, #1            \n"
                "st1    {v0.4s}, [%4], #16      \n"
                "bne    0b                      \n"

                : "+r"(ptr0),        // %0
                  "+r"(ptr1),        // %1
                  "+r"(ptr2),        // %2
                  "+r"(ptr3),        // %3
                  "+r"(output_ptr),  // %4
                  "+r"(nn)           // %5
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v30");
#else   // __aarch64__
            asm volatile(
                "vmov.f32   q13, #0.25               \n"
                "0:     \n"
                "vld1.32    q0, [%0]!           \n"
                "pld   [%0, #128]              \n"
                "vld1.32    q1, [%1]!           \n"
                "pld   [%1, #128]              \n"
                "vld1.32    q2, [%2]!           \n"
                "pld   [%2, #128]              \n"
                "vld1.32    q3, [%3]!           \n"
                "pld   [%3, #128]              \n"
                "vadd.f32   q0, q0, q1          \n"
                "vadd.f32   q2, q2, q3          \n"
                "vadd.f32   q0, q0, q2          \n"
                "vmul.f32   q0, q0, q13          \n"
                "subs   %5, %5, #1            \n"
                "vst1.32    q0, [%4]!          \n"
                "bne    0b                      \n"

                : "+r"(ptr0),        // %0
                  "+r"(ptr1),        // %1
                  "+r"(ptr2),        // %2
                  "+r"(ptr3),        // %3
                  "+r"(output_ptr),  // %4
                  "+r"(nn)           // %5
                :
                : "cc", "memory", "q0", "q1", "q2", "q3", "q13");
#endif  // __aarch64__
        }
    }
}
#endif  // __ARM_NEON

void ave_pool_fallback(const bnn::Mat &input, const size_t pad_h,
                       const size_t pad_w, const size_t stride_h,
                       const size_t stride_w, const size_t kernel_h,
                       const size_t kernel_w, bnn::Mat &output) {
    const int output_h =
        (input.h + 2 * pad_h - ((kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w =
        (input.w + 2 * pad_w - ((kernel_w - 1) + 1)) / stride_w + 1;

    BNN_ASSERT(input.w * input.c * input.elemsize % 16 == 0, "Not align");
    BNN_ASSERT(output.w * output.c * output.elemsize % 16 == 0, "Not align");
    BNN_ASSERT(input.data_type == input.data_type, "Mismatch datatype");

    const int input_h = input.h;
    const int input_w = input.w;
    const int input_c = input.c;
    const size_t input_hstep = input.hstep;

    thrust::counting_iterator<int> idx_begin(0);
    thrust::counting_iterator<int> idx_end = idx_begin + output_h * output_w;

    if (input.data_type == DataType::Float) {
        avepool_functor<float> func(input, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, output_w, input_h, input_w, input_c, input_hstep);
        thrust::device_vector<float> output_values(output_h * output_w * input_c);
        thrust::transform(thrust::device, idx_begin, idx_end, output.begin<float>(), func);
    } else if (input.data_type == DataType::Bit) {
        avepool_functor<uint64_t> func(input, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, output_w, input_h, input_w, input_c, input_hstep);
        thrust::device_vector<uint64_t> output_values(output_h * output_w * input_c);
        thrust::transform(thrust::device, idx_begin, idx_end, output.begin<uint64_t>(), func);
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
    cudaDeviceSynchronize();
}

AvePool::AvePool(NetCP net, const std::string &name, css input, css output,
                 int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                 int stride_w)
    : Layer(net, name, "AvePool"),
      input_mat(mat(input)),
      output_mat(mat(output)),
      kernel_h(kernel_h),
      kernel_w(kernel_w),
      pad_h(pad_h),
      pad_w(pad_w),
      stride_h(stride_h),
      stride_w(stride_w) {
    auto &mat_map = net.lock()->mat_map_;
    const auto &pad_name = "pad_for_" + output + "_cal";
    if (mat_map.find(pad_name) == mat_map.end()) {
        auto &input_mat = *mat_map[input];
        mat_map[pad_name] = std::make_shared<Mat>(
            input_mat.h + pad_h * 2, input_mat.w + pad_w * 2, input_mat.c,
            input_mat.data_type, pad_name);
    }
    padded_mat = mat_map[pad_name];
}

void AvePool::forward_impl() const {
#ifdef __ARM_NEON
    if (stride_h == 2 && stride_w == 2 && kernel_h == 2 && kernel_w == 2 &&
        input_mat->c % 4 == 0) {
        pad(*input_mat, pad_h, pad_w, *padded_mat);
        ave_pool_2x2_s2(*padded_mat, *output_mat);
    } else {
        ave_pool_fallback(*input_mat, pad_h, pad_w, stride_h, stride_w,
                          kernel_h, kernel_w, *output_mat);
    }
#else
    ave_pool_fallback(*input_mat, pad_h, pad_w, stride_h, stride_w, kernel_h,
                      kernel_w, *output_mat);
#endif  // __ARM_NEON
}

}  // namespace bnn
