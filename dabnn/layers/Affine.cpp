// Copyright 2019 JD.com Inc. JD AI

#include "Affine.h"

namespace bnn {

/**
 * per channel affine, x = a * x + b
 */
inline void affine_inplace(bnn::Mat &data, const bnn::Mat &a,
                           const bnn::Mat &b) {
    if (a.data_type != b.data_type && a.data_type != data.data_type) {
        throw std::invalid_argument("Mismatch datatype");
    } else if (a.data_type == DataType::Float) {
        FORZ(n, data.n) {
            FORZ(h, data.h) {
                auto ptr = data.point<float>(n, h, 0);
                auto a_ptr = a.point<float>(n, h, 0);
                auto b_ptr = b.point<float>(n, h, 0);
                FORZ(w, data.w) {
                    FORZ(c, data.c) {
                        *ptr = (*a_ptr++) * *ptr + (*b_ptr++);
                        ptr++;
                    }
                }
            }
        }
    } else if (b.data_type == DataType::Bit) {
        FORZ(n, data.n) {
            FORZ(h, data.h) {
                auto ptr = data.point<uint64_t>(n, h, 0);
                auto a_ptr = a.point<uint64_t>(n, h, 0);
                auto b_ptr = b.point<uint64_t>(n, h, 0);
                FORZ(w, data.w) {
                    FORZ(c, data.c) {
                        *ptr = (*a_ptr++) * *ptr + (*b_ptr++);
                        ptr++;
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
}

inline void affine(const bnn::Mat &data, const bnn::Mat &a, const bnn::Mat &b,
                   bnn::Mat &output) {
    if (a.data_type != b.data_type && a.data_type != data.data_type) {
        throw std::invalid_argument("Mismatch datatype");
    } else if (a.data_type == DataType::Float) {
        FORZ(n, data.n) {
            FORZ(h, data.h) {
                auto ptr = data.point<float>(n, h, 0);
                auto a_ptr = a.point<float>(n, h, 0);
                auto b_ptr = b.point<float>(n, h, 0);
                auto output_ptr = output.point<float>(n, h, 0);
                FORZ(w, data.w) {
                    FORZ(c, data.c) {
                        *output_ptr++ = (*a_ptr++) * (*ptr++) + (*b_ptr++);
                    }
                }
            }
        }
    } else if (b.data_type == DataType::Bit) {
        FORZ(n, data.n) {
            FORZ(h, data.h) {
                auto ptr = data.point<uint64_t>(n, h, 0);
                auto a_ptr = a.point<uint64_t>(n, h, 0);
                auto b_ptr = b.point<uint64_t>(n, h, 0);
                auto output_ptr = output.point<uint64_t>(n, h, 0);
                FORZ(w, data.w) {
                    FORZ(c, data.c) {
                        *output_ptr++ = (*a_ptr++) * (*ptr++) + (*b_ptr++);
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument("Unknown datatype");
    }
}

void Affine::forward_impl() const {
#ifdef BNN_CHECK_CONSISTENCY
    affine(*data_mat, *a_mat, *b_mat, *output_mat);
#else
    affine_inplace(*data_mat, *a_mat, *b_mat);
#endif
}

}  // namespace bnn
