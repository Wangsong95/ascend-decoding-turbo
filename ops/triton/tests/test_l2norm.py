import torch
import torch.nn as nn
from l2norm import L2Norm

# 设置设备（NPU）
device = 'npu'
print(f"Using device: {device}")

# ========================
# 1. 准备输入数据
# ========================
batch_size = 4
seq_len = 32
d_model = 512 

x = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device=device)
print(f"Input shape: {x.shape}")
print(f"Input dtype: {x.dtype}")

# ========================
# 2. 实例化 L2Norm 模块
# ========================
l2norm_layer = L2Norm(eps=1e-6, output_dtype=torch.float32).to(device)

# ========================
# 3. 前向传播
# ========================
y = l2norm_layer(x) 
print(f"Output shape: {y.shape}")

# 检查 L2 范数是否接近 1
norms = torch.norm(y, p=2, dim=-1)
print(f"Mean L2 norm: {norms.mean().item():.6f}")
print(f"Max deviation from 1: {torch.abs(norms - 1).max().item():.6f}")