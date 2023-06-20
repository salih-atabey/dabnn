// Copyright 2019 JD.com Inc. JD AI

#include "BinConv.h"

#include <common/baseline.h>
#include <dabnn/bconv.h>
#include <dabnn/bgemm.h>
#include <dabnn/bitpack.h>
#include <dabnn/fused_binarize_im2col.h>
#include <dabnn/net.h>
#include <dabnn/pad.h>

namespace bnn {

int align_to(int a, int b) { return ((a + (b - 1)) / b) * b; }

BinConv::BinConv(NetCP net, const std::string &name, css input, css weight,
                 css output, int pad_h, int pad_w, int stride_h, int stride_w)
    : Layer(net, name, "Bin Conv"),
      input_mat(mat(input)),
      weight_mat(mat(weight)),
      output_mat(mat(output)),
      pad_h(pad_h),
      pad_w(pad_w),
      stride_h(stride_h),
      stride_w(stride_w) {
    auto &mat_map = net.lock()->mat_map_;
    if (method() == Method::DIRECT_CONV || method() == Method::BCONV_NAIVE) {
        const auto binaized_name = "binaized_for_" + output + "_cal";
        if (mat_map.find(binaized_name) == mat_map.end()) {
            auto &input_mat = *mat_map[input];
            mat_map[binaized_name] = std::make_shared<Mat>(
                input_mat.h, input_mat.w, input_mat.elem_c, DataType::Bit,
                binaized_name);
        }
        binarized_mat = mat(binaized_name);
    }

    const auto pad_name = "pad_for_" + output + "_cal";
    if (mat_map.find(pad_name) == mat_map.end()) {
        auto &input_mat = *mat_map[input];
        mat_map[pad_name] = std::make_shared<Mat>(
            input_mat.h + pad_h * 2, input_mat.w + pad_w * 2, input_mat.elem_c,
            DataType::Bit, pad_name);
    }
    padded_mat = mat(pad_name);

    if (method() == Method::BGEMM || method() == Method::BGEMM_NAIVE) {
        const auto col_mat_name = "col_for_" + output + "_cal";
        if (mat_map.find(col_mat_name) == mat_map.end()) {
            const auto len =
                output_mat->h * output_mat->w *
                align_to(weight_mat->h * weight_mat->w * input_mat->elem_c,
                         128);
            mat_map[col_mat_name] =
                std::make_shared<Mat>(1, 1, len, bnn::DataType::Bit);
        }
        col_mat = mat(col_mat_name);
        const auto trans_weight_mat_name = "trans_" + weight;
        // transpose the weight for bgemm
        const int m = weight_mat->n;
        BNN_ASSERT(weight_mat->total() % m == 0, "");
        const int k = weight_mat->total() / m;
        transposed_weight_mat = std::make_shared<Mat>(m, k * 64, DataType::Bit);
        auto *trans_data_ptr =
            static_cast<uint64_t *>(transposed_weight_mat->data);
        auto *data_ptr = static_cast<uint64_t *>(weight_mat->data);
        FORZ(i, k) {
            FORZ(j, m) {
                BNN_ASSERT(static_cast<size_t>(i * m + j) <
                               transposed_weight_mat->total(),
                           i * m + j, " ", transposed_weight_mat->total());
                trans_data_ptr[i * m + j] = data_ptr[j * k + i];
            }
        }
        net_.lock()->add_mat(trans_weight_mat_name, transposed_weight_mat);
    }
}

BinConv::Method BinConv::method() const {
    if (net_.lock()->optimize) {
        if (direct_conv_compatible()) {
            return Method::DIRECT_CONV;
        } else if (gemm_compatible()) {
            return Method::BGEMM;
        } else if (input_mat->elem_c == 64) {
            return Method::BCONV_NAIVE;
        } else {
            return Method::BGEMM_NAIVE;
        }
    } else {
        if (input_mat->elem_c == 64) {
            return Method::BCONV_NAIVE;
        } else {
            return Method::BGEMM_NAIVE;
        }
    }
}

bool BinConv::direct_conv_compatible() const {
#ifdef __aarch64__
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->elem_c == 64 &&
        stride_h == stride_w) {
        return true;
    }
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->elem_c == 128 &&
        stride_h == stride_w) {
        return true;
    }
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->elem_c == 256 &&
        stride_h == stride_w) {
        return true;
    }
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->elem_c == 512 &&
        stride_h == stride_w) {
        return true;
    }
    if (weight_mat->h == 3 && weight_mat->w == 3 && input_mat->elem_c == 1024 &&
        stride_h == stride_w) {
        return true;
    }
    return false;
#else
    return false;
#endif
}

bool BinConv::gemm_compatible() const {
#ifdef __ARM_NEON
#ifdef __aarch64__
    return true;
#else
    // If input_mat->elem_c == 1 (weight_mat has 64 channels), we use bconv_64
    // in aarch64 for the fastest speed, however, bconv_64 is not implemented
    // in armv7
    // TODO: Implement bconv_64 for armv7
    return input_mat->elem_c != 64;
#endif
#else
    return false;
#endif
}

inline void binconv(const Mat &input, const Mat &weight,
                        const int kernel_h, const int kernel_w,
                        const int pad_h, const int pad_w, const int stride_h,
                        const int stride_w, const int dilation_h,
                        const int dilation_w, const int output_channels,
                        Mat &output) {
    BNN_ASSERT(weight.total() % weight.n == 0, "");
    const auto HWC = weight.total() / weight.n;
    int input_y = 0;
    FORZ(th, output.h) {
        int input_x = 0;
        FORZ(tw, output.w) {
            FORZ(tc, output_channels) {
                uint32_t acc = 0;
                FORZ(wh, kernel_h) {
                    int y = input_y - pad_h + wh * dilation_h;
                    FORZ(ww, kernel_w) {
                        int x = input_x - pad_w + ww * dilation_w;
                        FORZ(wc, input.c) {
                            int idx = tc * HWC +
                                      wh * kernel_w * input.c + ww * input.c +
                                      wc;
                            const auto w_value =
                                *(static_cast<uint64_t *>(weight.data) + idx);
                            bool out =
                                y < 0 || y >= input.h || x < 0 || x >= input.w;
                            const auto bottom_value =
                                out ? 0 : *(input.point<uint64_t>(y, x) + wc);
                            uint8_t tmp = ::bitcount(w_value ^ bottom_value);
                            acc += tmp;
                        }
                    }
                }
                *(output.point<float>(th, tw) + tc) = static_cast<float>(acc);
            }
            input_x += stride_w;
        }
        input_y += stride_h;
    }
}

void BinConv::forward_impl() const {
    switch (method()) {
        case Method::DIRECT_CONV: {
            pack_mat(*input_mat, *binarized_mat);
            pad(*binarized_mat, pad_h, pad_w, *padded_mat);
            bconv_3x3(*padded_mat, *weight_mat, *output_mat, stride_h);
            break;
        }
        case Method::BGEMM: {
            output_mat->fill<float>(0.f);

            bnn::fused_binarize_im2col(*input_mat, weight_mat->h, weight_mat->w,
                                       pad_h, pad_w, stride_h, stride_w, 1, 1,
                                       *col_mat);

            const int m = weight_mat->n;
            const int n = output_mat->h * output_mat->w;
            const int k = weight_mat->total() / weight_mat->n;
            bgemm(m, n, k, static_cast<uint64_t *>(transposed_weight_mat->data),
                  m, static_cast<uint64_t *>(col_mat->data), k,
                  static_cast<float *>(output_mat->data), m);
            break;
        }
        case Method::BGEMM_NAIVE: {
            output_mat->fill<float>(0.f);

            bnn::fused_binarize_im2col(*input_mat, weight_mat->h, weight_mat->w,
                                       pad_h, pad_w, stride_h, stride_w, 1, 1,
                                       *col_mat);

            const int m = weight_mat->n;
            const int n = output_mat->h * output_mat->w;
            const int k = weight_mat->total() / weight_mat->n;
            bgemm_naive(m, n, k,
                        static_cast<uint64_t *>(transposed_weight_mat->data), m,
                        static_cast<uint64_t *>(col_mat->data), k,
                        static_cast<float *>(output_mat->data), m);
            break;
        }
        case Method::BCONV_NAIVE: {
            pack_mat(*input_mat, *binarized_mat);
            baseline_bconv(*binarized_mat, *weight_mat, weight_mat->h,
                           weight_mat->w, pad_h, pad_w, stride_h, stride_w, 1,
                           1, output_mat->c, *output_mat);
            break;
        }
    }
}

std::string BinConv::to_str() const {
    std::stringstream ss;
    ss << type_ << ", ";
    PNT_TO(ss, input_mat->h, input_mat->w, input_mat->elem_c, weight_mat->h,
           weight_mat->w, weight_mat->n, pad_h, pad_w);

    return ss.str();
}

}  // namespace bnn
