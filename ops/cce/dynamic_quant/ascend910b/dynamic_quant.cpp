/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add.cpp
 * \brief
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>
#include "kernel_common.h"

namespace adt_ops {
namespace DynamicQuant {
    // Register the operator's schema
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("dynamic_quant(Tensor x) -> (Tensor, Tensor)");
}

// Meta function implementation of Add
std::tuple<torch::Tensor, torch::Tensor> dynamic_quant_meta(const torch::Tensor &x)
{
    TORCH_CHECK(x.defined(), "Input tensor is a null pointer (undefined tensor)");
    auto y = torch::empty(x.sizes(), x.options().dtype(torch::kInt8));
    int64_t token_nums = x.size(0);
    auto z = torch::empty({token_nums}, x.options().dtype(torch::kFloat32));
    return {y, z};
}

// Register the Meta implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("dynamic_quant", dynamic_quant_meta);
}

template <typename T>
extern __global__ __aicore__ void dynamic_quant_kernel(GM_ADDR x, GM_ADDR quant_out, GM_ADDR scales, int64_t token_nums, int64_t hidden_size)
{
    auto *x1 = reinterpret_cast<__ubuf__ bfloat16_t *>((uintptr_t)0);
    auto *x2 = reinterpret_cast<__ubuf__ bfloat16_t *>(x1 + hidden_size);
    auto *xf32 = reinterpret_cast<__ubuf__ float *>(x2 + hidden_size);
    auto *xAbs = reinterpret_cast<__ubuf__ float *>(xf32 + hidden_size);
    auto *zf16 = reinterpret_cast<__ubuf__ half *>(xAbs + hidden_size);
    auto *z1 = reinterpret_cast<__ubuf__ int8_t *>(zf16 + hidden_size);
    auto *z2 = reinterpret_cast<__ubuf__ int8_t *>(z1 + hidden_size);
    auto *scaleUb = reinterpret_cast<__ubuf__ float *>(z2 + hidden_size);
    auto *sum_addr = reinterpret_cast<__ubuf__ float *>(scaleUb + 1);
    assert((uint64_t)sum_addr <= 196608);

    __ubuf__ bfloat16_t *xBufs[2] = {x1, x2};
    __ubuf__ int8_t *zBufs[2] = {z1, z2};


#ifdef __DAV_C220_VEC__
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    
    int eventId = 0;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

    for (int row = block_idx; row < token_nums; row += block_num) {
        int rowOffset = row * hidden_size;

        wait_flag(PIPE_V, PIPE_MTE2, eventId);
        copy_gm_to_ubuf_align_b16(xBufs[eventId],
                                  reinterpret_cast<__gm__ bfloat16_t *>(x) + rowOffset, 0, 1,
                                  hidden_size * sizeof(bfloat16_t), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, eventId);

        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        vconv_bf162f32(xf32, xBufs[eventId], hidden_size / 64, 1, 1, 8, 4);
        set_flag(PIPE_V, PIPE_MTE2, eventId);

        pipe_barrier(PIPE_V);
        vabs(xAbs, xf32, hidden_size / 64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        ReduceMax(xAbs, xAbs, hidden_size);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);

        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float scale = float(*xAbs) / float(127);
        float scaleRec = float(127) / float(*xAbs);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        *scaleUb = float(scale);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b32(reinterpret_cast<__gm__ float *>(scales) + row, scaleUb, 0, 1,
                                  sizeof(float), 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

        vmuls(xf32, xf32, float(scaleRec), hidden_size / 64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16a(zf16, xf32, hidden_size / 64, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, eventId);
        vconv_f162s8a(zBufs[eventId], zf16, hidden_size / 128, 1, 1, 4, 8);
        set_flag(PIPE_V, PIPE_MTE3, eventId);

        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        copy_ubuf_to_gm_align_b8(quant_out + rowOffset, zBufs[eventId], 0, 1, hidden_size, 0, 0, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, eventId);

        eventId = 1 - eventId;
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
#endif
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_quant_npu(const torch::Tensor &x)
{
    // OptionalDeviceGuard 确保后续操作在正确的设备上下文执行
 	// 它会记录当前设备状态，执行完作用域代码后自动恢复
    std::cout << "Execute dynamic_quant_npu \n"; 
 	const c10::OptionalDeviceGuard guard(x.device());
    auto [quant_tensor, scale] = dynamic_quant_meta(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t token_num = x.size(0);
    int64_t hidden_size = x.size(1);

    uint64_t aiv_num = 40;
    // // aclrtGetDeviceInfo(0, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv_num);
    int64_t numBlocks = token_num < aiv_num ? token_num : aiv_num;

    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto quant_tensor_ptr = (GM_ADDR)quant_tensor.data_ptr();
    auto scale_ptr = (GM_ADDR)scale.data_ptr();
    auto acl_call = [=]() -> int {
        AT_DISPATCH_SWITCH(
            x.scalar_type(), "dynamic_quant_npu",
            AT_DISPATCH_CASE(torch::kBFloat16, [&] {
                using scalar_t = bfloat16_t;
                dynamic_quant_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, quant_tensor_ptr, scale_ptr, token_num, hidden_size);
            })
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("DynamicQuant", acl_call);
    return {quant_tensor, scale};
}

// Register the NPU implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("dynamic_quant", dynamic_quant_npu);
}

} // namespace DynamicQuant
} // namespace flash_infer_ops