// Copyright 2019 JD.com Inc. JD AI

#include "Binarize.h"

#include <dabnn/bitpack.h>
#include <dabnn/net.h>

namespace bnn {

struct pack_64_bitfield_functor
{
    float *fptr_base;
    uint64_t *bptr_base;

    pack_64_bitfield_functor(float* _fptr, uint64_t* _bptr) : fptr_base(_fptr), bptr_base(_bptr) {}

    __host__ __device__
    void operator()(int i)
    {
        float* fptr = fptr_base + i * 64;
        uint64_t* buf = bptr_base + i;
        struct bf {
            unsigned int b0 : 1;
            unsigned int b1 : 1;
            unsigned int b2 : 1;
            unsigned int b3 : 1;
            unsigned int b4 : 1;
            unsigned int b5 : 1;
            unsigned int b6 : 1;
            unsigned int b7 : 1;
            unsigned int b8 : 1;
            unsigned int b9 : 1;
            unsigned int b10 : 1;
            unsigned int b11 : 1;
            unsigned int b12 : 1;
            unsigned int b13 : 1;
            unsigned int b14 : 1;
            unsigned int b15 : 1;
            unsigned int b16 : 1;
            unsigned int b17 : 1;
            unsigned int b18 : 1;
            unsigned int b19 : 1;
            unsigned int b20 : 1;
            unsigned int b21 : 1;
            unsigned int b22 : 1;
            unsigned int b23 : 1;
            unsigned int b24 : 1;
            unsigned int b25 : 1;
            unsigned int b26 : 1;
            unsigned int b27 : 1;
            unsigned int b28 : 1;
            unsigned int b29 : 1;
            unsigned int b30 : 1;
            unsigned int b31 : 1;
            unsigned int b32 : 1;
            unsigned int b33 : 1;
            unsigned int b34 : 1;
            unsigned int b35 : 1;
            unsigned int b36 : 1;
            unsigned int b37 : 1;
            unsigned int b38 : 1;
            unsigned int b39 : 1;
            unsigned int b40 : 1;
            unsigned int b41 : 1;
            unsigned int b42 : 1;
            unsigned int b43 : 1;
            unsigned int b44 : 1;
            unsigned int b45 : 1;
            unsigned int b46 : 1;
            unsigned int b47 : 1;
            unsigned int b48 : 1;
            unsigned int b49 : 1;
            unsigned int b50 : 1;
            unsigned int b51 : 1;
            unsigned int b52 : 1;
            unsigned int b53 : 1;
            unsigned int b54 : 1;
            unsigned int b55 : 1;
            unsigned int b56 : 1;
            unsigned int b57 : 1;
            unsigned int b58 : 1;
            unsigned int b59 : 1;
            unsigned int b60 : 1;
            unsigned int b61 : 1;
            unsigned int b62 : 1;
            unsigned int b63 : 1;
        };

        union bf_u64 {
            bf t;
            uint64_t u64;
        };

        bf_u64 u;
        u.t.b0 = fptr[0] > 0;
        u.t.b1 = fptr[1] > 0;
        u.t.b2 = fptr[2] > 0;
        u.t.b3 = fptr[3] > 0;
        u.t.b4 = fptr[4] > 0;
        u.t.b5 = fptr[5] > 0;
        u.t.b6 = fptr[6] > 0;
        u.t.b7 = fptr[7] > 0;
        u.t.b8 = fptr[8] > 0;
        u.t.b9 = fptr[9] > 0;
        u.t.b10 = fptr[10] > 0;
        u.t.b11 = fptr[11] > 0;
        u.t.b12 = fptr[12] > 0;
        u.t.b13 = fptr[13] > 0;
        u.t.b14 = fptr[14] > 0;
        u.t.b15 = fptr[15] > 0;
        u.t.b16 = fptr[16] > 0;
        u.t.b17 = fptr[17] > 0;
        u.t.b18 = fptr[18] > 0;
        u.t.b19 = fptr[19] > 0;
        u.t.b20 = fptr[20] > 0;
        u.t.b21 = fptr[21] > 0;
        u.t.b22 = fptr[22] > 0;
        u.t.b23 = fptr[23] > 0;
        u.t.b24 = fptr[24] > 0;
        u.t.b25 = fptr[25] > 0;
        u.t.b26 = fptr[26] > 0;
        u.t.b27 = fptr[27] > 0;
        u.t.b28 = fptr[28] > 0;
        u.t.b29 = fptr[29] > 0;
        u.t.b30 = fptr[30] > 0;
        u.t.b31 = fptr[31] > 0;
        u.t.b32 = fptr[32] > 0;
        u.t.b33 = fptr[33] > 0;
        u.t.b34 = fptr[34] > 0;
        u.t.b35 = fptr[35] > 0;
        u.t.b36 = fptr[36] > 0;
        u.t.b37 = fptr[37] > 0;
        u.t.b38 = fptr[38] > 0;
        u.t.b39 = fptr[39] > 0;
        u.t.b40 = fptr[40] > 0;
        u.t.b41 = fptr[41] > 0;
        u.t.b42 = fptr[42] > 0;
        u.t.b43 = fptr[43] > 0;
        u.t.b44 = fptr[44] > 0;
        u.t.b45 = fptr[45] > 0;
        u.t.b46 = fptr[46] > 0;
        u.t.b47 = fptr[47] > 0;
        u.t.b48 = fptr[48] > 0;
        u.t.b49 = fptr[49] > 0;
        u.t.b50 = fptr[50] > 0;
        u.t.b51 = fptr[51] > 0;
        u.t.b52 = fptr[52] > 0;
        u.t.b53 = fptr[53] > 0;
        u.t.b54 = fptr[54] > 0;
        u.t.b55 = fptr[55] > 0;
        u.t.b56 = fptr[56] > 0;
        u.t.b57 = fptr[57] > 0;
        u.t.b58 = fptr[58] > 0;
        u.t.b59 = fptr[59] > 0;
        u.t.b60 = fptr[60] > 0;
        u.t.b61 = fptr[61] > 0;
        u.t.b62 = fptr[62] > 0;
        u.t.b63 = fptr[63] > 0;

        *buf = u.u64;
    }
};

inline void binarize(const bnn::Mat &float_mat, bnn::Mat &binary_mat) {
    /**
     * This is the bit-packing for tensor of less than 128 channels.
     */
    BNN_ASSERT(
        float_mat.w * float_mat.c > 0 && float_mat.w * float_mat.c % 64 == 0,
        float_mat.w * float_mat.c);
    BNN_ASSERT(float_mat.c / 64 == binary_mat.c && float_mat.c % 64 == 0,
               "float_mat.c ", float_mat.c, ", binary_mat.c ", binary_mat.c);
       
    int size = float_mat.n * float_mat.h * (float_mat.w * float_mat.c / 64);
    pack_64_bitfield_functor func((float*)float_mat.data, (uint64_t*)binary_mat.data);
    thrust::counting_iterator<int> iter(0);
    thrust::for_each(iter, iter + size, func);
    cudaDeviceSynchronize();
}

void Binarize::forward_impl() const { binarize(*input_mat, *output_mat); }

}  // namespace bnn
