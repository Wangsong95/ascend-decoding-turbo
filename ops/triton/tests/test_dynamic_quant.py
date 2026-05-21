import torch
import torch_npu

import triton
import triton.language as tl
from triton_core.dynamic_quant import dynamic_quant

# 测试代码
if __name__ == "__main__":
    
    # 设置随机种子
    torch.manual_seed(0)
    
    # 创建测试数据
    batch_size = 64
    seq_len = 1
    hidden_size = 4096
    repeat_time = 30
    
    
    # 创建bf16类型的输入tensor
    try:
        input_tensor = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
        print(f"CPU上创建bf16 tensor成功")
        
        # 移动到NPU
        if torch.npu.is_available():
            input_tensor = input_tensor.npu()
            print(f"NPU上创建bf16 tensor成功")
        else:
            print(f"NPU不可用，使用CPU")
        
        # 执行量化
        print("NPU API triton.dynamic_quant开始执行...")
        quantized_tensor0, scale_tensor0 = dynamic_quant(input_tensor)

        print("NPU API torch_npu.npu_dynamic_quant开始执行...")
        quantized_tensor1, scale_tensor1 = torch_npu.npu_dynamic_quant(input_tensor)
        
        
        # 精度验证
        print("\n开始精度验证...")
        
        # 检查形状是否一致
        shape_match = (quantized_tensor0.shape == quantized_tensor1.shape) and (scale_tensor0.shape == scale_tensor1.shape)
        print(f"形状匹配: {shape_match}")
        
        if shape_match:
            # 检查缩放因子张量是否完全相同
            scale_equal = torch.allclose(scale_tensor0, scale_tensor1, atol=1e-6)
            print(f"缩放因子张量完全相同: {scale_equal}")
            
            # 检查量化张量是否完全相同
            quantized_equal = torch.allclose(quantized_tensor0, quantized_tensor1, atol=1)
            print(f"量化张量完全相同: {quantized_equal}")

            # 最终验证结果
            if quantized_equal and scale_equal:
                print("\n✓ 验证通过！两种实现的输出完全一致。")
            else:
                print("\n✗ 验证失败！两种实现的输出存在差异。")
        else:
            print("\n✗ 验证失败！两种实现的输出形状不一致。")
            print(f"自定义实现量化张量形状: {quantized_tensor0.shape}")
            print(f"NPU API量化张量形状: {quantized_tensor1.shape}")
            print(f"自定义实现缩放因子形状: {scale_tensor0.shape}")
            print(f"NPU API缩放因子形状: {scale_tensor1.shape}")

        
    except Exception as e:
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()

