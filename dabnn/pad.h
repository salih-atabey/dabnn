// Copyright 2019 JD.com Inc. JD AI

#ifndef PAD_H
#define PAD_H

#include <common/helper.h>
#include "mat.h"

namespace bnn {
inline void pad(const bnn::Mat &input, const int pad_h, const int pad_w,
                bnn::Mat &output, float val = 0.f) {
    BNN_ASSERT(input.data_type == output.data_type,
               "Input and output data_type is not the same");
    if (output.data_type == bnn::DataType::Bit) {
        BNN_ASSERT(val == 0.f, "val cannot be set for DataType::Bit");
        // 0 is 000000000....000 in binary, and take the role of -1 in xnor
        output.fill<uint64_t>(0);
        FORZ(h, input.h) {
            auto *out_p = output.point<uint64_t>(h + pad_h, pad_w);
            const auto *input_p = input.point<uint64_t>(h, 0);
            memcpy(out_p, input_p, input.w * input.c * input.elemsize);
        }
    } else if (output.data_type == bnn::DataType::Float) {
        output.fill<float>(val);
        FORZ(h, input.h) {
            auto *out_p = output.point<float>(h + pad_h, pad_w);
            const auto *input_p = input.point<float>(h, 0);
            memcpy(out_p, input_p, input.w * input.c * input.elemsize);
        }
    } else {
        BNN_ASSERT(false, "Unknown data_type");
    }
}
}  // namespace bnn

#endif /* PAD_H */
