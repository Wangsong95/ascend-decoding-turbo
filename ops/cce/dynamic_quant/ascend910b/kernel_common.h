#ifndef _KERNEL_COMMON_H_
#define _KERNEL_COMMON_H_

#include "kernel_operator.h"
using namespace AscendC;

#define ROUND_DOWN(x, y) (((x) / (y)) * (y))
#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define DIV_ROUND_UP(x, y) (((x) + ((y) - 1)) / (y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define BLOCK_SIZE 32
#define VECTOR_MAX_REPEAT 255
#define VECTOR_MAX_BYTESIZE 256     // The maximum byte size of one repeat in vector
#define VECTOR_MAX_NUM_OF_FP32 64   // The maximum num of float32 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_FP16 128  // The maximum num of float16 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_BF16 128  // The maximum num of bfloat16 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_INT8 256  // The maximum num of int8 dtype in one vector repeat
#define AIC_CACHE_LINE_SIZE 512
#define MATMUL_M0_N0_K0_DEFAULT_VALUE ((uint64_t)(-1))
#define UB_SIZE 196608
#define UB_BUF_ALIGN_SIZE 32  // The align size of UB buffer address
#define PINGPONG_BUF_NUM 2
#define C2_DATABLOCK 64        // The data block size of C2
#define FIXPIPE_DATABLOCK 128  // The data block size of fixpipe
#define FLOAT_MIN -3.4028235e+38

#ifdef __DAV_C220_VEC__

inline __aicore__ void SetMask(int32_t len)
{
    uint64_t temp = len % 64;
    uint64_t mask = ((uint64_t)1 << temp) - 1;

    if (len == 128) {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    } else if (len >= 64) {
        set_vector_mask(mask, (uint64_t)-1);
    } else {
        set_vector_mask(0x0, mask);
    }
}

// make sure that: len != 0
inline __aicore__ void SetMaskFromHighBit(int32_t high, int32_t len)
{
    uint64_t temp = len % 64;
    uint64_t mask = ~(((uint64_t)1 << (64 - temp)) - 1);

    if (high == 128) {
        // If dtype size is 2 bytes, all 128 bits of mask take effect.
        if (len > 64) {
            set_vector_mask((uint64_t)-1, mask);
        } else {
            set_vector_mask(mask, 0x0);
        }
    } else if (high == 64) {
        // If dtype size is 4 bytes, only lower 64 bits take effect.
        set_vector_mask(0, mask);
    }
}

template <typename Dtype>
__inline__ __aicore__ void ReduceMax(__ubuf__ Dtype *dst, __ubuf__ Dtype *src, uint32_t dim)
{
    Dtype min;
    uint32_t remain = dim;
    __ubuf__ Dtype *calc = src;

    int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    int instPad = VECTOR_MAX_REPEAT * pad;
    if constexpr (std::is_same<Dtype, half>::value) {
        min = half(-65504);
    } else if constexpr (std::is_same<Dtype, float>::value) {
        min = -3.4028235e+38;
    }

    uint32_t repeat = DIV_ROUND_UP(remain, pad);
    set_mask_norm();
    if (remain == 1) {
        SetMask(remain);
        vcmax(dst, calc, repeat, 1, 1, 8, Order_t::ONLY_VALUE);
        pipe_barrier(PIPE_V);
    }

    while (remain != 1) {
        if (repeat == 1) {
            SetMask(remain);
        } else {
            if (remain % pad != 0) {
                SetMaskFromHighBit(pad, pad - remain % pad);
                vector_dup(calc + ROUND_DOWN(remain, pad), min, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
            }
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }

        if (repeat > VECTOR_MAX_REPEAT) {
            int instNum = DIV_ROUND_UP(repeat, VECTOR_MAX_REPEAT);
            for (int i = 0; i < instNum; i++) {
                int currRepeat = VECTOR_MAX_REPEAT;
                if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                    currRepeat = repeat - i * VECTOR_MAX_REPEAT;
                }
                vcmax(dst + i * VECTOR_MAX_REPEAT, calc + i * instPad, currRepeat, 1, 1, 8,
                      Order_t::ONLY_VALUE);
            }
        } else {
            vcmax(dst, calc, repeat, 1, 1, 8, Order_t::ONLY_VALUE);
        }
        calc = dst;
        pipe_barrier(PIPE_V);
        remain = repeat;
        repeat = DIV_ROUND_UP(remain, pad);
    }
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

#endif
#endif