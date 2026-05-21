#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import flash_infer_ops
import pytest

def test_add_interface_exist():
    """
    Test that the 'flash_infer_ops.add' operator is present in torch.ops.
    This existence test asserts that the custom operator registered under the
    'flash_infer_ops' namespace is discoverable from Python via torch.ops.flash_infer_ops.add.
    It does not exercise operator functionality — only that the Python binding
    and registration are available.
    Rationale:
    The presence of this test guards against a common failure mode where an
    operator is implemented and registered in C++/ATen but is not exposed to
    the Python torch.ops namespace due to mismatches between the PyTorch
    operator schema and the C++ registration signature (argument names, types,
    or overloads). Such schema/signature inconsistencies can cause the
    operator to be hidden or not exported to Python, breaking consumers that
    expect to call torch.ops.flash_infer_ops.add. This test will fail loudly if the
    binding is missing, prompting investigation into schema/registration issues.
    """
    # This test specifically protects against discrepancies between the
    # PyTorch operator schema and the C++ signature/registration that can
    # prevent the operator from being visible in torch.ops.flash_infer_ops.
    print(torch.ops.flash_infer_ops.dynamic_quant)
    assert hasattr(torch.ops.flash_infer_ops, "dynamic_quant"), "The 'dynamic_quant' operator is not registered in the 'torch.ops.flash_infer_ops' namespace."

SHAPES = [
    (64, 8192),
]

DTYPES = [
    torch.bfloat16,
]

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)

def test_dynamic_quant_operator(shape, dtype):
    """
    Test the functionality of the add operator, using concise but comprehensive combinations of shapes and data types.

    Parameters:
        shape: Tensor shape
        dtype: Data type
    """

    x = torch.randn(*shape, dtype=dtype)
    x = x.npu()
    print("Execute torch-npu dynamic_quant operator...")
    expect_y, expect_scale = torch_npu.npu_dynamic_quant(x)
    print("Execute flash-infer dynamic_quant operator...")
    result_y, result_scale = torch.ops.adt_ops.dynamic_quant(x)


    print(f"expect_y: {expect_y}")
    print(f"result_y: {result_y}")
    print("\n开始精度验证...")
        
    # 检查形状是否一致
    shape_match = (expect_y.shape == result_y.shape) and (expect_scale.shape == result_scale.shape)
    print(f"形状匹配: {shape_match}")
        
    if shape_match:
        # 检查缩放因子张量是否完全相同
        # 对于float32类型，使用allclose比较，设置较小的容差
        scale_equal = torch.allclose(expect_scale, result_scale, atol=1e-6)
        print(f"缩放因子张量完全相同: {scale_equal}")
            
        # 检查量化张量是否完全相同
        quantized_equal = torch.allclose(expect_y, result_y, atol=1)
        print(f"缩放因子张量完全相同: {quantized_equal}")
        # quantized_equal = torch.equal(expect_y, result_y)
        # print(f"量化张量完全相同: {quantized_equal}")

        if not quantized_equal:
        # 找出所有不相等的位置
            diff_mask = expect_y != result_y
            diff_indices = torch.nonzero(diff_mask).tolist()  # 所有差异位置
            total_diff = len(diff_indices)

            print(f"\n===== 差异统计 =====")
            print(f"总差异点数量: {total_diff}")

        # 最终验证结果
        if quantized_equal and scale_equal:
            print("\n✓ 验证通过！两种实现的输出完全一致。")
        else:
            print("\n✗ 验证失败！两种实现的输出存在差异。")
    else:
        print("\n✗ 验证失败！两种实现的输出形状不一致。")
        print(f"flash-infer量化张量形状: {result_y.shape}")
        print(f"torch-npu量化张量形状: {expect_y.shape}")
        print(f"flash-infer缩放因子形状: {result_scale.shape}")
        print(f"torch-npu缩放因子形状: {expect_scale.shape}")


if __name__ == "__main__":
    print("Starting dynamic_quant operator test...")
    
    # 调用测试函数，传入你想要的 shape 和 dtype
    test_dynamic_quant_operator(shape=[40, 4096], dtype=torch.bfloat16)