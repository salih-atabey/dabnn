// Copyright 2019 JD.com Inc. JD AI

#include "Add.h"

namespace bnn {

inline void add_inplace(bnn::Mat &a, const bnn::Mat &b) {
    if (a.data_type != b.data_type) {
        throw std::invalid_argument("Mismatch datatype");
    } else if (a.data_type == DataType::Float) {
        FORZ(n, a.n) {
            FORZ(h, a.h) {
                auto *a_ptr = a.point<float>(n, h, 0);
                const auto *b_ptr = b.point<float>(n, h, 0);
                FORZ(w, a.w) {
                    FORZ(c, a.c) { *a_ptr++ += *b_ptr++; }
                }
            }
        }
    } else if (b.data_type == DataType::Bit) {
        FORZ(n, a.n) {
            FORZ(h, a.h) {
                auto *a_ptr = a.point<uint64_t>(n, h, 0);
                const auto *b_ptr = b.point<uint64_t>(n, h, 0);
                FORZ(w, a.w) {
                    FORZ(c, a.c) { *a_ptr++ += *b_ptr++; }
                }
            }
        }
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
}

inline void add(const bnn::Mat &a, const bnn::Mat &b, bnn::Mat &c) {
    if (a.data_type != b.data_type) {
        throw std::invalid_argument("Mismatch datatype");
        FORZ(n, a.n) {
            FORZ(h, a.h) {
                const auto *a_ptr = a.point<float>(n, h, 0);
                const auto *b_ptr = b.point<float>(n, h, 0);
                auto *c_ptr = c.point<float>(n, h, 0);
                FORZ(w, a.w) {
                    FORZ(c, a.c) { *c_ptr++ = *a_ptr++ + *b_ptr++; }
                }
            }
        }
    } else if (b.data_type == DataType::Bit) {
        FORZ(n, a.n) {
            FORZ(h, a.h) {
                const auto *a_ptr = a.point<uint64_t>(n, h, 0);
                const auto *b_ptr = b.point<uint64_t>(n, h, 0);
                auto *c_ptr = c.point<float>(n, h, 0);
                FORZ(w, a.w) {
                    FORZ(c, a.c) { *c_ptr++ = *a_ptr++ + *b_ptr++; }
                }
            }
        }
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
