import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def dynamic_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tokens = batch_size * seq_len
    token_per_core = total_tokens // BLOCK_DIM
    tokens_proc = token_per_core
    more_token_core_num = total_tokens - token_per_core * BLOCK_DIM
    io_offset = 0

    if pid < more_token_core_num:
        tokens_proc += 1
        io_offset = pid * tokens_proc
    else:
        io_offset = more_token_core_num * (token_per_core + 1) + (pid - more_token_core_num) * token_per_core

    for i in range(tokens_proc):
        token_idx = io_offset + i
        data_offset = token_idx * hidden_size
        scale_offset = token_idx

        # 计算当前块的偏移 - 由于hidden_size < BLOCK_SIZE，只需要处理一次
        offset = data_offset + tl.arange(0, BLOCK_SIZE)
        # 创建掩码，防止越界访问
        mask = tl.arange(0, BLOCK_SIZE) < hidden_size
        
        # 加载bf16数据并转换为float32进行计算
        x = tl.load(input_ptr + offset, mask=mask).to(tl.float32)
        
        # 计算当前token的最大值
        max_val = tl.max(tl.abs(x))
        
        # 计算scale：max_val / 127.0
        scale = max_val / 127.0
        scale_r = 127.0 / max_val
        # 量化输入数据：直接实现NPU API的量化公式
        quantized = x * scale_r
        # 转换为int8并使用饱和模式防止溢出
        quantized = tl.cast(quantized, dtype=tl.int8)
        
        # 存储量化结果
        tl.store(output_ptr + offset, quantized, mask=mask)
        
        # 存储scale（确保只存储标量值）
        tl.store(scale_ptr + scale_offset, scale)


def dynamic_quant(input_tensor: torch.Tensor):
    """
    对bf16类型的输入tensor进行per-token动态量化，输出int8 tensor和量化系数
    
    Args:
        input_tensor: bf16类型的tensor，形状为[batch_size, seq_len, hidden_size]
        
    Returns:
        quantized_tensor: int8类型的tensor，形状与输入相同
        scale_tensor: float32类型的tensor，形状为[batch_size, seq_len]
    """

    batch_size, seq_len, hidden_size = input_tensor.shape
    
    # 预分配输出tensor
    quantized_tensor = torch.empty_like(input_tensor, dtype=torch.int8)
    scale_tensor = torch.empty((batch_size, seq_len), dtype=torch.float32, device=input_tensor.device)
    
    # 设置block大小
    BLOCK_SIZE = 4096
    BLOCK_DIM = 40

    # 计算grid大小
    total_tokens = batch_size * seq_len
    grid = (BLOCK_DIM,)
    # 启动kernel
    dynamic_quant_kernel[grid](
        input_tensor,
        quantized_tensor,
        scale_tensor,
        batch_size,
        seq_len,
        hidden_size,
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return quantized_tensor, scale_tensor
