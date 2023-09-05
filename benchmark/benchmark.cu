// Copyright 2019 JD.com Inc. JD AI

#include <chrono>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <benchmark/benchmark.h>
#include <common/baseline.h>
#include <common/helper.h>
#include <dabnn/bconv.h>
#include <dabnn/bgemm.h>
#include <dabnn/bitpack.h>
#include <dabnn/mat.h>
#include <dabnn/net.h>
#include <dabnn/layers/Add.cu>
#include <dabnn/layers/Affine.cu>
#include <dabnn/layers/AvePool.cu>
#include <dabnn/layers/Binarize.cu>
#include <dabnn/layers/BinConv.cu>
#include <dabnn/layers/MaxPool.cu>
#include <dabnn/layers/Pad.cpp>
#include <dabnn/layers/MaxPool.h>
#include <common/cyclic_iterator.h>

static void BM_pack_mat_64_small(benchmark::State &state) {
    const bnn::Mat a(1, 32, 32, 128, bnn::DataType::Float, false);
    bnn::Mat b(1, 32, 32, 128, bnn::DataType::Bit, false);
    for (auto _ : state) {
        pack_mat_64(a, b);
    }
}

#ifdef __aarch64__
static void BM_pack_mat_128_small(benchmark::State &state) {
    const bnn::Mat a(1, 32, 32, 128, bnn::DataType::Float, false);
    bnn::Mat b(1, 32, 32, 128, bnn::DataType::Bit, false);
    for (auto _ : state) {
        pack_mat_128(a, b);
    }
}
#endif  // __aarch64__

static void BM_pack_mat_64(benchmark::State &state) {
    const bnn::Mat a(1, 64, 64, 128, bnn::DataType::Float);
    bnn::Mat b(1, 64, 64, 128, bnn::DataType::Bit);
    for (auto _ : state) {
        pack_mat_64(a, b);
    }
}

#ifdef __aarch64__
static void BM_pack_mat_128(benchmark::State &state) {
    const bnn::Mat a(1, 64, 64, 128, bnn::DataType::Float);
    bnn::Mat b(1, 64, 64, 128, bnn::DataType::Bit);
    for (auto _ : state) {
        pack_mat_128(a, b);
    }
}
#endif  // __aarch64__

#define SETUP_BCONV_FLOAT(size_a, size_b, num_output)                         \
    const size_t AHEIGHT = size_a;                                            \
    const size_t AWIDTH = size_a;                                             \
    const size_t CHANNEL = num_output;                                        \
                                                                              \
    const size_t BHEIGHT = size_b;                                            \
    const size_t BWIDTH = size_b;                                             \
    const size_t NUM_OUTPUT = num_output;                                     \
                                                                              \
    const size_t CHEIGHT = AHEIGHT - BHEIGHT + 1;                             \
    const size_t CWIDTH = AWIDTH - BWIDTH + 1;                                \
                                                                              \
    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;                           \
    const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL / 64;         \
                                                                              \
    float a_data[ALEN];                                                       \
    uint64_t b_data[BLEN];                                                    \
    FORZ(i, ALEN) { a_data[i] = 3 * i; }                                      \
    FORZ(i, BLEN) { b_data[i] = 2 * i; }                                      \
                                                                              \
    const bnn::Mat a(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Float); \
    const bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL, b_data,            \
                     bnn::DataType::Bit);                                     \
                                                                              \
    bnn::Mat a_binary(AHEIGHT, AWIDTH, CHANNEL, bnn::DataType::Bit);          \
                                                                              \
    bnn::Mat c(CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);

#ifdef __aarch64__
static void BM_bconv_float_3x3_128(benchmark::State &state) {
    SETUP_BCONV_FLOAT(30, 3, 128);
    for (auto _ : state) {
        pack_mat_128(a, a_binary);
        bnn::bconv_3x3(a_binary, b, c);
    }
}

static void BM_bconv_float_1x1_128(benchmark::State &state) {
    SETUP_BCONV_FLOAT(28, 1, 128);
    for (auto _ : state) {
        pack_mat_128(a, a_binary);
        bnn::bconv_1x1_128(a_binary, b, c);
    }
}
#endif  // __aarch64__

#undef SETUP_BCONV_FLOAT

#define SETUP_BCONV(size_a, size_b, num_output, stride)                     \
    const size_t AHEIGHT = size_a;                                          \
    const size_t AWIDTH = size_a;                                           \
    const size_t CHANNEL = num_output / 64;                                 \
                                                                            \
    const size_t BHEIGHT = size_b;                                          \
    const size_t BWIDTH = size_b;                                           \
    const size_t NUM_OUTPUT = num_output;                                   \
                                                                            \
    const size_t CHEIGHT = (AHEIGHT - BHEIGHT + 1) / stride;                \
    const size_t CWIDTH = (AWIDTH - BWIDTH + 1) / stride;                   \
                                                                            \
    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;                         \
    const size_t BLEN = NUM_OUTPUT * BHEIGHT * BWIDTH * CHANNEL;            \
                                                                            \
    uint64_t *a_data;                                                       \
    uint64_t *b_data;                                                       \
    cudaMallocManaged((void **)&a_data, ALEN * sizeof(uint64_t) / 64);      \
    cudaMallocManaged((void **)&b_data, BLEN * sizeof(uint64_t) / 64);      \
    FORZ(i, ALEN) { a_data[i] = 3 * i; }                                    \
    FORZ(i, BLEN) { b_data[i] = 2 * i; }                                    \
                                                                            \
    bnn::Mat a(1, AHEIGHT, AWIDTH, CHANNEL * sizeof(uint64_t) * 8, a_data,  \
               bnn::DataType::Bit);                                         \
    bnn::Mat b(NUM_OUTPUT, BHEIGHT, BWIDTH, CHANNEL * sizeof(uint64_t) * 8, \
               b_data, bnn::DataType::Bit, false);                          \
                                                                            \
    bnn::Mat c(1, CHEIGHT, CWIDTH, NUM_OUTPUT, bnn::DataType::Float);

static void BM_bnn_bconv_debug(benchmark::State &state) {
    SETUP_BCONV(8, 3, 128, 1);
    for (auto _ : state) {
        std::cout << "--- Debug BinConv ---" << std::endl;
        std::cout << "Vector A:" << std::endl;
        a.display();
        std::cout << "Vector B:" << std::endl;
        b.display();
        std::cout << "Conv vectors A and B..." << std::endl;
        bnn::baseline_bconv(a, b, BHEIGHT, BWIDTH, 0, 0, 1, 1, 1, 1, NUM_OUTPUT,
                            c);
        std::cout << "Vector C:" << std::endl;
        c.display();
    }
    cudaFree(a_data);
    cudaFree(b_data);
}

static void BM_bnn_bconv_3x3_naive_128(benchmark::State &state) {
    SETUP_BCONV(30, 3, 128, 1);
    for (auto _ : state) {
        bnn::binconv(a, b, BHEIGHT, BWIDTH, 0, 0, 1, 1, 1, 1, NUM_OUTPUT,
                            c);
    }
}

static void BM_bnn_bconv_1x1_naive_128(benchmark::State &state) {
    SETUP_BCONV(28, 1, 128, 1);
    for (auto _ : state) {
        bnn::binconv(a, b, BHEIGHT, BWIDTH, 0, 0, 1, 1, 1, 1, NUM_OUTPUT,
                            c);
    }
}

#ifdef __aarch64__
static void BM_bnn_bconv_1x1_64(benchmark::State &state) {
    SETUP_BCONV(56, 1, 64, 1);
    for (auto _ : state) {
        bnn::bconv_1x1_64(a, b, c);
    }
}

static void BM_bnn_bconv_1x1_128(benchmark::State &state) {
    SETUP_BCONV(28, 1, 128, 1);
    for (auto _ : state) {
        bnn::bconv_1x1_128(a, b, c);
    }
}

static void BM_bnn_bconv_1x1_256(benchmark::State &state) {
    SETUP_BCONV(14, 1, 256, 1);
    for (auto _ : state) {
        bnn::bconv_1x1_256(a, b, c);
    }
}

static void BM_bnn_bconv_1x1_512(benchmark::State &state) {
    SETUP_BCONV(7, 1, 512, 1);
    for (auto _ : state) {
        bnn::bconv_1x1_512(a, b, c);
    }
}
#endif  // __aarch64__

static void BM_bnn_bconv_3x3_64(benchmark::State &state) {
    SETUP_BCONV(58, 3, 64, 1);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c);
    }
}

static void BM_bnn_bconv_3x3_128(benchmark::State &state) {
    SETUP_BCONV(30, 3, 128, 1);
    for (auto _ : state) {
        // bnn::bconv_3x3(a, b, c);
        // baseline_bconv(a, b, 3, 3, 0, 0, stride_h, stride_w, 1, 1, NUM_OUTPUT, c);
        binconv(a, b, 3, 3, 0, 0, 1, 1, 1, 1, NUM_OUTPUT, c);
    }
    cudaFree(a_data);
    cudaFree(b_data);
}

static void BM_bnn_bconv_3x3_256(benchmark::State &state) {
    SETUP_BCONV(16, 3, 256, 1);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c);
    }
}

static void BM_bnn_bconv_3x3_256_s2(benchmark::State &state) {
    SETUP_BCONV(16, 3, 256, 2);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c, 2);
    }
}

static void BM_bnn_bconv_3x3_512(benchmark::State &state) {
    SETUP_BCONV(9, 3, 512, 1);
    for (auto _ : state) {
        bnn::bconv_3x3(a, b, c);
    }
}

static void BM_bnn_bconv_3x3_1024(benchmark::State &state) {
    SETUP_BCONV(9, 3, 1024, 1);
    for (auto _ : state) {
        // bnn::bconv_3x3(a, b, c);
        // baseline_bconv(a, b, 3, 3, 0, 0, stride_h, stride_w, 1, 1, NUM_OUTPUT, c);
        binconv(a, b, 3, 3, 0, 0, 1, 1, 1, 1, NUM_OUTPUT, c);
    }
    cudaFree(a_data);
    cudaFree(b_data);
}

static void BM_bnn_bconv(benchmark::State &state) {
    SETUP_BCONV(30, 3, 128, 1);
    for (auto _ : state) {
        bnn::binconv(a, b, BHEIGHT, BWIDTH, 0, 0, 1, 1, 1, 1, NUM_OUTPUT,
                            c);
    }
}

#undef SETUP_BCONV

#if 0
static void BM_maxpool3x3(benchmark::State &state) {
    const size_t AHEIGHT = 32;
    const size_t AWIDTH = 32;
    const size_t CHANNEL = 128;

    const size_t CHEIGHT = 30;
    const size_t CWIDTH = 30;

    const size_t ALEN = AHEIGHT * AWIDTH * CHANNEL;

    uint64_t a_data[ALEN];
    FORZ(i, ALEN) { a_data[i] = 3 * i; }

    const auto a = std::make_shared<bnn::Mat>(AHEIGHT, AWIDTH, CHANNEL, a_data, bnn::DataType::Float);

    auto c = std::make_shared<bnn::Mat>(CHEIGHT, CWIDTH, CHANNEL, bnn::DataType::Float);

    const auto m = bnn::MaxPool(a, a, c, 3, 3, 0, 0, 1, 1);
    for (auto _ : state) {
        m.forward();
    }
}
#endif

#define SETUP_BGEMM     \
    uint64_t a[102400]; \
    uint64_t b[102400]; \
    float c[602400];

static void BM_bgemm_64(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(64, 56 * 56, 9, a, 64, b, 9, c, 64);
    }
}

static void BM_bgemm_128(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(128, 28 * 28, 18, a, 128, b, 18, c, 128);
    }
}

static void BM_bgemm_naive_128(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm_naive(128, 28 * 28, 18, a, 128, b, 18, c, 128);
    }
}

static void BM_bgemm_256(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(256, 14 * 14, 36, a, 256, b, 36, c, 256);
    }
}

static void BM_bgemm_256_s2(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(256, 7 * 7, 36, a, 256, b, 36, c, 256);
    }
}

static void BM_bgemm_512(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(512, 7 * 7, 72, a, 512, b, 72, c, 512);
    }
}

static void BM_bgemm_5x5_256(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm(256, 14 * 14, 100, a, 256, b, 100, c, 256);
    }
}

static void BM_bgemm_naive_256(benchmark::State &state) {
    SETUP_BGEMM;
    for (auto _ : state) {
        bgemm_naive(256, 14 * 14, 36, a, 256, b, 36, c, 256);
    }
}

static void BM_bireal18_cifar(benchmark::State &state) {
    float input[3 * 32 * 32];

    auto net = bnn::Net::create();
    net->read("/data/local/tmp/model_cifar.dab");
    for (auto _ : state) {
        net->run(input);
    }
}

static void BM_bireal18_imagenet(benchmark::State &state) {
    // float input[3 * 224 * 224];
    float *input;
    cudaMallocManaged((void **)&input, 3 * 224 * 224 * sizeof(float));

    auto net = bnn::Net::create();
    net->read("/workspace/dabnn/models/birealnet18.dab");
    for (auto _ : state) {
        net->run(input);
    }
    cudaDeviceSynchronize();
    cudaFree(input);
    // net->~Net();
}

static void BM_bireal18_imagenet_stem(benchmark::State &state) {
    // float input[3 * 224 * 224];
    float *input;
    cudaMallocManaged((void **)&input, 3 * 224 * 224 * sizeof(float));

    auto net = bnn::Net::create();
    net->read("/workspace/dabnn/models/birealnet18stem.dab");
    for (auto _ : state) {
        net->run(input);
    }
    cudaDeviceSynchronize();
    cudaFree(input);
    // net->~Net();
}

static void BM_bireal18_cifar_wo_fconv(benchmark::State &state) {
    float input[3 * 32 * 32];

    auto net = bnn::Net::create();
    net->run_fconv = false;
    net->strict = false;
    net->read("/data/local/tmp/model_cifar.dab");
    for (auto _ : state) {
        net->run(input);
    }
}

static void BM_bireal18_imagenet_wo_fconv(benchmark::State &state) {
    float input[3 * 224 * 224];

    auto net = bnn::Net::create();
    net->run_fconv = false;
    net->strict = false;
    net->read("/data/local/tmp/model_imagenet.dab");
    for (auto _ : state) {
        net->run(input);
    }
}

#define SETUP_BADD(n, w, h, c)                                           \
    const size_t LENGTH = n * w * h * c;                                 \
                                                                         \
    uint64_t *a_data;                                                    \
    uint64_t *b_data;                                                    \
    cudaMallocManaged((void **)&a_data, LENGTH * sizeof(uint64_t) / 64); \
    cudaMallocManaged((void **)&b_data, LENGTH * sizeof(uint64_t) / 64); \
    FORZ(i, LENGTH) { a_data[i] = 3 * i; }                               \
    FORZ(i, LENGTH) { b_data[i] = 2 * i; }                               \
                                                                         \
    bnn::Mat a(n, w, h, c, a_data, bnn::DataType::Bit);                  \
    bnn::Mat b(n, w, h, c, b_data, bnn::DataType::Bit);

static void BM_badd_debug(benchmark::State &state) {
    SETUP_BADD(2, 4, 4, 64);
    for (auto _ : state) {
        std::cout << "--- Debug Add ---" << std::endl;
        std::cout << "a.n: " << a.n << std::endl;
        std::cout << "a.h: " << a.h << std::endl;
        std::cout << "a.w: " << a.w << std::endl;
        std::cout << "a.c: " << a.c << std::endl;
        std::cout << "a.ptr: " << a.data << std::endl;
        std::cout << "b.ptr: " << b.data << std::endl;
        std::cout << "a.begin(): " << a.begin<uint64_t>() << std::endl;
        std::cout << "b.begin(): " << b.begin<uint64_t>() << std::endl;
        std::cout << "a.end(): " << a.end<uint64_t>() << std::endl;
        std::cout << "b.end(): " << b.end<uint64_t>() << std::endl;
        std::cout << "a.total(): " << a.total() << std::endl;
        std::cout << "b.total(): " << b.total() << std::endl;
        std::cout << "a.diff: " << a.end<uint64_t>() - a.begin<uint64_t>()
                  << std::endl;
        std::cout << "b.diff: " << b.end<uint64_t>() - b.begin<uint64_t>()
                  << std::endl;
        std::cout << "Vector A:" << std::endl;
        a.display();
        std::cout << "Vector B:" << std::endl;
        b.display();
        std::cout << "Add vectors A and B..." << std::endl;
        add_inplace(a, b);
        std::cout << "Vector A:" << std::endl;
        a.display();
    }
    cudaFree(a_data);
    cudaFree(b_data);
}

static void BM_badd_256(benchmark::State &state) {
    SETUP_BADD(1, 1, 256, 64);
    for (auto _ : state) {
        add_inplace(a, b);
    }
    cudaFree(a_data);
    cudaFree(b_data);
}

static void BM_badd_1024(benchmark::State &state) {
    SETUP_BADD(1, 1, 1024, 64);
    for (auto _ : state) {
        add_inplace(a, b);
    }
    cudaFree(a_data);
    cudaFree(b_data);
}
#undef SETUP_BADD

#define SETUP_BAFFINE(n, w, h, c)                                    \
    const size_t LENGTH = n * w * h * c;                             \
                                                                     \
    uint64_t *a_data;                                                   \
    uint64_t *x_data;                                                   \
    uint64_t *b_data;                                                   \
    cudaMallocManaged((void **)&a_data, LENGTH * sizeof(uint64_t) / 64); \
    cudaMallocManaged((void **)&x_data, LENGTH * sizeof(uint64_t) / 64); \
    cudaMallocManaged((void **)&b_data, LENGTH * sizeof(uint64_t) / 64); \
    FORZ(i, LENGTH) { a_data[i] = 13; }                              \
    FORZ(i, LENGTH) { x_data[i] = 7; }                               \
    FORZ(i, LENGTH) { b_data[i] = 5; }                               \
                                                                     \
    bnn::Mat a(n, w, h, c, a_data, bnn::DataType::Bit);            \
    bnn::Mat x(n, w, h, c, x_data, bnn::DataType::Bit);            \
    bnn::Mat b(n, w, h, c, b_data, bnn::DataType::Bit);

static void BM_baffine_debug(benchmark::State &state) {
    SETUP_BAFFINE(1, 4, 4, 1);
    for (auto _ : state) {
        std::cout << "--- Debug Affine ---" << std::endl;
        std::cout << "a.ptr: " << a.data << std::endl;
        std::cout << "b.ptr: " << b.data << std::endl;
        std::cout << "a.begin(): " << a.begin<uint64_t>() << std::endl;
        std::cout << "b.begin(): " << b.begin<uint64_t>() << std::endl;
        std::cout << "a.end(): " << a.end<uint64_t>() << std::endl;
        std::cout << "b.end(): " << b.end<uint64_t>() << std::endl;
        std::cout << "a.total(): " << a.total() << std::endl;
        std::cout << "b.total(): " << b.total() << std::endl;
        std::cout << "a.diff: " << a.end<uint64_t>() - a.begin<uint64_t>()
                  << std::endl;
        std::cout << "b.diff: " << b.end<uint64_t>() - b.begin<uint64_t>()
                  << std::endl;
        std::cout << "Vector A:" << std::endl;
        a.display();
        std::cout << "Vector X:" << std::endl;
        x.display();
        std::cout << "Vector B:" << std::endl;
        b.display();
        std::cout << "Affine vectors A and B..." << std::endl;
        bnn::affine_inplace(a, x, b);
        std::cout << "Vector A:" << std::endl;
        a.display();
    }
    cudaFree(a_data);
    cudaFree(x_data);
    cudaFree(b_data);
}

static void BM_baffine_256(benchmark::State &state) {
    SETUP_BAFFINE(1, 1, 256, 64);
    for (auto _ : state) {
        bnn::affine_inplace(a, x, b);
    }
    cudaFree(a_data);
    cudaFree(x_data);
    cudaFree(b_data);
}

static void BM_baffine_1024(benchmark::State &state) {
    SETUP_BAFFINE(1, 1, 1024, 64);
    for (auto _ : state) {
        bnn::affine_inplace(a, x, b);
    }
    cudaFree(a_data);
    cudaFree(x_data);
    cudaFree(b_data);
}
#undef SETUP_BAFFINE

#define SETUP_BAVEPOOL(n, w, h, c, p)                            \
    const size_t PWIDTH = p;                                     \
    const size_t PHEIGHT = p;                                    \
    const size_t LENGTH = n * w * h * c;                         \
                                                                 \
    float *a_data;                                               \
    cudaMallocManaged((void **)&a_data, LENGTH * sizeof(float)); \
    FORZ(i, LENGTH) { a_data[i] = i * i; }                       \
                                                                 \
    bnn::Mat a(n, w, h, c, a_data, bnn::DataType::Float);        \
    bnn::Mat b(n, w, h, c, bnn::DataType::Float);

static void BM_bavepool_debug(benchmark::State &state) {
    SETUP_BAVEPOOL(1, 8, 8, 1, 3);
    for (auto _ : state) {
        std::cout << "--- Debug AvePool ---" << std::endl;
        std::cout << "Vector A:" << std::endl;
        a.display();
        std::cout << "AvePool vectors A and B..." << std::endl;
        bnn::ave_pool_fallback(a, 1, 1, 1, 1, PWIDTH, PHEIGHT, b);
        std::cout << "Vector B:" << std::endl;
        b.display();
    }
    cudaFree(a_data);
}

static void BM_bavepool_256(benchmark::State &state) {
    SETUP_BAVEPOOL(1, 256, 256, 1, 3);
    for (auto _ : state) {
        bnn::ave_pool_fallback(a, 1, 1, 1, 1, PWIDTH, PHEIGHT, b);
    }
    cudaFree(a_data);
}

static void BM_bavepool_512(benchmark::State &state) {
    SETUP_BAVEPOOL(1, 512, 512, 1, 3);
    for (auto _ : state) {
        bnn::ave_pool_fallback(a, 1, 1, 1, 1, PWIDTH, PHEIGHT, b);
    }
    cudaFree(a_data);
}
#undef SETUP_BAVEPOOL

#define SETUP_BINARIZE(n, w, h, c)                               \
    const size_t LENGTH = n * w * h * c;                         \
                                                                 \
    float *a_data;                                               \
    cudaMallocManaged((void **)&a_data, LENGTH * sizeof(float)); \
    FORZ(i, LENGTH) { a_data[i] = i; }                           \
                                                                 \
    bnn::Mat a(n, w, h, c, a_data, bnn::DataType::Float);        \
    bnn::Mat b(n, w, h, c, bnn::DataType::Bit);

static void BM_binarize_debug(benchmark::State &state) {
    SETUP_BINARIZE(1, 4, 4, 64);
    for (auto _ : state) {
        std::cout << "--- Debug Binarize ---" << std::endl;
        std::cout << "Vector A:" << std::endl;
        a.display();
        std::cout << "Binarize vector A..." << std::endl;
        bnn::binarize(a, b);
        std::cout << "Vector B:" << std::endl;
        b.display();
    }
    cudaFree(a_data);
}

static void BM_binarize_256(benchmark::State &state) {
    SETUP_BINARIZE(1, 1, 256, 64);
    for (auto _ : state) {
        bnn::binarize(a, b);
    }
    cudaFree(a_data);
}

static void BM_binarize_1024(benchmark::State &state) {
    SETUP_BINARIZE(1, 1, 1024, 64);
    for (auto _ : state) {
        bnn::binarize(a, b);
    }
    cudaFree(a_data);
}
#undef SETUP_BINARIZE

#define SETUP_BMAXPOOL(n, w, h, c, p)                            \
    const size_t PWIDTH = p;                                     \
    const size_t PHEIGHT = p;                                    \
    const size_t LENGTH = n * w * h * c;                         \
                                                                 \
    float *a_data;                                               \
    cudaMallocManaged((void **)&a_data, LENGTH * sizeof(float)); \
    FORZ(i, LENGTH) { a_data[i] = i * i; }                       \
                                                                 \
    bnn::Mat a(n, w, h, c, a_data, bnn::DataType::Float);        \
    bnn::Mat b(n, w, h, c, bnn::DataType::Float);

static void BM_bmaxpool_debug(benchmark::State &state) {
    SETUP_BMAXPOOL(1, 8, 8, 1, 3);
    for (auto _ : state) {
        std::cout << "--- Debug MaxPool ---" << std::endl;
        std::cout << "Vector A:" << std::endl;
        a.display();
        std::cout << "MaxPool vectors A and B..." << std::endl;
        bnn::max_pool_fallback(a, 1, 1, 1, 1, PWIDTH, PHEIGHT, b);
        std::cout << "Vector B:" << std::endl;
        b.display();
    }
    cudaFree(a_data);
}

static void BM_bmaxpool_256(benchmark::State &state) {
    SETUP_BMAXPOOL(1, 256, 256, 1, 3);
    for (auto _ : state) {
        bnn::max_pool_fallback(a, 1, 1, 1, 1, PWIDTH, PHEIGHT, b);
    }
    cudaFree(a_data);
}

static void BM_bmaxpool_512(benchmark::State &state) {
    SETUP_BMAXPOOL(1, 512, 512, 1, 3);
    for (auto _ : state) {
        bnn::max_pool_fallback(a, 1, 1, 1, 1, PWIDTH, PHEIGHT, b);
    }
    cudaFree(a_data);
}
#undef SETUP_BMAXPOOL

#define SETUP_BPAD(n, w, h, c, p)                                        \
    const size_t PWIDTH = p;                                             \
    const size_t PHEIGHT = p;                                            \
    const size_t LENGTH = n * w * h * c;                                 \
                                                                         \
    uint64_t *a_data;                                                    \
    cudaMallocManaged((void **)&a_data, LENGTH * sizeof(uint64_t) / 64); \
    FORZ(i, LENGTH) { a_data[i] = 1; }                                   \
                                                                         \
    bnn::Mat a(n, w, h, c, a_data, bnn::DataType::Bit);                  \
    bnn::Mat b(n, w + p, h + p, c, bnn::DataType::Bit);

static void BM_bpad_debug(benchmark::State &state) {
    SETUP_BPAD(1, 16, 16, 64, 16);
    for (auto _ : state) {
        std::cout << "--- Debug Pad ---" << std::endl;
        std::cout << "Vector A:" << std::endl;
        a.display();
        std::cout << "Pad vector A..." << std::endl;
        bnn::pad(a, PWIDTH, PHEIGHT, b, 0.f);
        std::cout << "Vector B:" << std::endl;
        b.display();
    }
    cudaFree(a_data);
}

static void BM_bpad_16(benchmark::State &state) {
    SETUP_BPAD(1, 16, 16, 64, 16);
    for (auto _ : state) {
        bnn::pad(a, PWIDTH, PHEIGHT, b, 0.f);
    }
    cudaFree(a_data);
}

static void BM_bpad_32(benchmark::State &state) {
    SETUP_BPAD(1, 32, 32, 64, 32);
    for (auto _ : state) {
        bnn::pad(a, PWIDTH, PHEIGHT, b, 0.f);
    }
    cudaFree(a_data);
}
#undef SETUP_BPAD

BENCHMARK_MAIN();

/* ORIGIN */
// BENCHMARK(BM_pack_mat_64);
// BENCHMARK(BM_pack_mat_128);
// BENCHMARK(BM_bnn_bconv_1x1_64);
// BENCHMARK(BM_bnn_bconv_1x1_128);
// BENCHMARK(BM_bnn_bconv_1x1_256);
// BENCHMARK(BM_bnn_bconv_1x1_512);
// BENCHMARK(BM_bgemm_128);
// BENCHMARK(BM_bgemm_256);
// BENCHMARK(BM_bgemm_256_s2);
// BENCHMARK(BM_bgemm_5x5_256);
// BENCHMARK(BM_bgemm_512);
// BENCHMARK(BM_bnn_bconv_3x3_64);
// BENCHMARK(BM_bnn_bconv_3x3_128);
// BENCHMARK(BM_bnn_bconv_3x3_256);
// BENCHMARK(BM_bnn_bconv_3x3_256_s2);
// BENCHMARK(BM_bnn_bconv_3x3_512);
// BENCHMARK(BM_bnn_bconv_3x3_1024);
// BENCHMARK(BM_bireal18_cifar_wo_fconv);
// BENCHMARK(BM_bireal18_imagenet_wo_fconv);
// BENCHMARK(BM_bireal18_cifar);
// BENCHMARK(BM_bireal18_imagenet);
// BENCHMARK(BM_bireal18_imagenet_stem);
// BENCHMARK(BM_bnn_bconv_3x3_naive_128);
// BENCHMARK(BM_bconv_float_1x1_128);
// BENCHMARK(BM_bconv_float_3x3_128);/*

// BIREAL
BENCHMARK(BM_bireal18_imagenet);
BENCHMARK(BM_bireal18_imagenet_stem);

// ADD
BENCHMARK(BM_badd_256);
BENCHMARK(BM_badd_1024);
// BENCHMARK(BM_badd_debug)->Iterations(1);

// AFFINE
BENCHMARK(BM_baffine_256);
BENCHMARK(BM_baffine_1024);
// BENCHMARK(BM_baffine_debug)->Iterations(1);

// AVEPOLL
BENCHMARK(BM_bavepool_256);
BENCHMARK(BM_bavepool_512);
// BENCHMARK(BM_bavepool_debug)->Iterations(1);

// BINARIZE
BENCHMARK(BM_binarize_256);
BENCHMARK(BM_binarize_1024);
// BENCHMARK(BM_binarize_debug)->Iterations(1);

// BINCONV
BENCHMARK(BM_bnn_bconv_3x3_128);
BENCHMARK(BM_bnn_bconv_3x3_1024);
// BENCHMARK(BM_bnn_bconv_debug)->Iterations(1);

// MAXPOLL
BENCHMARK(BM_bmaxpool_256);
BENCHMARK(BM_bmaxpool_512);
// BENCHMARK(BM_bmaxpool_debug)->Iterations(1);

// PAD
BENCHMARK(BM_bpad_16);
BENCHMARK(BM_bpad_32);
// BENCHMARK(BM_bpad_debug)->Iterations(1);
