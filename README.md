# ascend-decoding-turbo

## 🚀概述

本项目基于CCE/Triton语言，聚焦Transformer decoding场景，实现极致性能算子优化。突破CANN算子泛化限制，针对Decode场景进行专项加速；规避Ascend C头开销在Decode阶段的放大问题，充分释放算力潜能；有效补齐昇腾平台开箱性能短板，实现对标业界最优推理库的推理性能竞争力。

## 🔥Latest News

- [2026/05] ascend-decoding-turbo 项目首次上线。

## ADT目录结构
关键目录如下：
```
├── cmake                              # 项目工程编译目录
├── adt_ops                            # PyTorch Extension相关依赖
├── ops                                # 自定义算子实现目录
│   ├── cce                            # 自定义CCE 算子目录
│       ├── add                        # add 实现目录
│           ├── ascend910b             # npu对应版本
│               ├── add.cpp            # add 算子实现文件
│               └── CMakeLists.txt     # add 算子编译文件
│           └── tests                  # add 算子测试目录
│   └── trition                        # Triton 算子目录
│       ├── __init__.py                # python目录依赖
│       ├── l2norm.py                  # l2nrom 算子实现文件
│       ├── utils.py                   # Triton 算子实现依赖
│       └── tests                      # Triton 算子测试目录
├── README.md                          # ADT仓介绍文档
├── setup.py                           # whl包生成脚本
├── build.sh                           # 项目工程编译whl脚本，依赖Camke与setup.py
├── CMakeLists.txt                     # 项目工程编译脚本
├── SECURITY.md                        # 安全声明
├── LICENSE                            # 许可证
└── requirements.txt                   # 本项目需要的第三方依赖包
```

# ADT CCE

## 简介

本文档演示如何使用 CCE 和[PyTorch Extension](https://docs.pytorch.org/tutorials/extension.html)能力开发自定义NPU算子。

**核心优势:**

- **单交付件：** 一个文件完成算子开发和PyTorch框架适配。
- **高效调用：** 使用`<<<>>>`语法启动核函数，流程简单高效。

## 环境部署 | Prerequisites
- triton 3.5.0
- 对应版本的[triton-ascend](https://gitcode.com/Ascend/triton-ascend/releases/v3.2.1)
- 请先参考[环境部署](../../docs/zh/install/quick_install.md)完成基础环境搭建
- gcc 9.4.0+
- python 3.8+
- torch>=2.6.0
- 对应版本的[torch_npu](https://gitcode.com/Ascend/pytorch/releases)

### ​CANN 开发环境部署

首先需安装 CANN 开发包，提供 NPU 算子运行所需的底层驱动与工具链。
推荐使用是社区版8.5.2，总共要下2个run包，这里以A3机器为例（即需要下载A3-ops、toolkit）
下载地址为
[https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2)
需要找到与你当前机器对应的包

```
#设置需要安装的路径
export INSTALL_PATH=/usr/local/Ascend

./Ascend-cann-toolkit*run --install-path=$INSTALL_PATH --full  --quiet
./Ascend-cann-*run --install-path=$INSTALL_PATH --install --quiet
source $INSTALL_PATH/ascend-toolkit/set_env.sh
```

## 安装步骤 | Installation Steps

1. 进入`ascend_decoding_turbo`目录。

2. 安装依赖 | Install Dependencies:

    ```sh
    python3 -m pip install -r requirements.txt
    ```

3. 构建Wheel包 | Build the Wheel:

    ```sh
    # -n: non-isolated build (uses existing environment)
    (编译全部算子)bash build.sh
    (指定算子编译) bash build.sh --ops=add,sub
    ```

    构建完成后，产物在当前目录的`dist`文件夹下，产物名`adt_ops-1.0.0-${python_version}-abi3-${arch}.whl`，
    `${python_version}`表示当前环境中的python版本(python3.8.3为cp38)，`${arch}`表示CPU架构。

4. 安装Wheel包 | Install Package:

    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

5. （可选）再次构建前建议先执行以下命令清理编译缓存

   ```sh
    python setup.py clean
    ```

6. （可选）执行测试用例
    进入`ops/cce/${op_name}/tests`目录
   ```sh
    pytest test_${op_name}.py
    ```

## 快速开始 | Quick Start

安装完成后，您可以像使用普通PyTorch操作一样使用NPU算子，以add算子调用为例。

```python
import torch
import torch_npu
import adt_ops  # 构建出的python包

# Initialize data on NPU
x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()

# Call the custom NPU operator
npu_result = torch.ops.adt_ops.add(x, y)  # PyTorch Custom Operator Dispatch机制: torch.ops.<library_name>.<operator_name>

# Verify against CPU ATen implementation
cpu_x = x.cpu()
cpu_y = y.cpu()
cpu_result = cpu_x + cpu_y

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("Verification successful!")
```

## 开发指南：新增一个算子 | Developer Guide: Adding a New Operator

为了实现一个新算子(如`add`)，您只需要提供一个C++实现即可。

1. 首先您需要在ops/cce目录下使用算子名`add`建立一个文件夹，在此文件夹内使用你当前想要开发的soc名建立一个子文件夹`ascend910b`。

2. 在soc目录下新建一个`CMakeLists.txt`

    ```
    add_sources("--npu-arch=dav-2201")
    ```

    这里`dav-2201`为ascend910b芯片对应的编译参数，获取方法参考[NpuArch说明和使用指导](https://gitcode.com/cann/ops-math/wiki/NpuArch%E8%AF%B4%E6%98%8E%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC.md)。

3. 在soc目录下新建一个`add.cpp`(建议使用算子名为文件名)。这个文件包含了开发一个AI Core算子所需要的全部模块。
    - 算子Schema注册
    - 算子Meta Function实现 & 注册
    - 算子Kernel实现 (Ascend C)
    - 算子NPU调用实现 & 注册

    ```cpp
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/framework/OpCommand.h"
    #include "kernel_operator.h"
    #include "platform/platform_ascendc.h"
    #include <type_traits>

    namespace adt_ops {  // 当前项目为一个命名空间
    namespace Add {         // 建议每个算子自己有一个独立的namespace，防止全局变量污染

    /**
     * 将算子schema注册给PyTorch框架
     * 框架知道有这样一个算子
     */
    // Register the operator's schema
    TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
    {
        m.def("add(Tensor x, Tensor y) -> Tensor");
    }

    /**
     * 实现算子的Meta函数，即InferShape+InferDtype
     * 根据输入推导出这个算子的输出是什么样子，需要多少空间，不需要实际计算这个算子
     */
    // Meta function implementation of Add
    torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
    {
        TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
        auto z = torch::empty_like(x);
        return z;
    }

    /**
     * 将算子的Meta函数注册给框架
     * 框架可以调用这个Meta函数，在真正执行这个算子计算前知道需要多大空间
     * 后续可以支持torch.compile/AutoGrad/AclGraph等图加速
     */
    // Register the Meta implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
    {
        m.impl("add", add_meta);
    }

    /**
     * NPU算子Kernel实现，使用AscendC API，面向当前的soc编写
     */
    template <typename T>
    __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        // kernel implementation
    }

    /**
     * 实现算子调用接口
     * 在这个接口中, 需要完成NPU Kernel的调用
     * 1. 计算出输出的Tensor的个数/Shape/Dtype(可以调用Meta函数实现，也可以直接实现)
     * 2. 计算Tiling：根据Shape得到如何分块计算
     * 3. 调用NPU Kernel
     *
     */
    torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
    {
        // OptionalDeviceGuard 确保后续操作在正确的设备上下文执行
        // 它会记录当前设备状态，执行完作用域代码后自动恢复
        const c10::OptionalDeviceGuard guard(x.device());
        auto z = add_meta(x, y);
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        int64_t totalLength, numBlocks, blockLength, tileSize;
        totalLength = x.numel();
        std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength);
        auto x_ptr = (GM_ADDR)x.data_ptr();
        auto y_ptr = (GM_ADDR)y.data_ptr();
        auto z_ptr = (GM_ADDR)z.data_ptr();
        auto acl_call = [=]() -> int {
            AT_DISPATCH_SWITCH(
                x.scalar_type(), "add_npu",
                // 根据不同的数据类型，调用不同的NPU Kernel
                AT_DISPATCH_CASE(torch::kFloat32, [&] {
                    using scalar_t = float;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kFloat16, [&] {
                    using scalar_t = half;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kInt32, [&] {
                    using scalar_t = int32_t;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
            );
            return 0;
        };
        // 需要使用RunOpApi/RunOpApiV2接口调用，保证时序与TorchNPU调用aclnn接口一致。
        at_npu::native::OpCommand::RunOpApi("Add", acl_call);
        return z;
    }

    /**
     * 将算子的调用函数注册给框架，Device为PrivateUse1
     * 框架知道当输入均在NPU Device上时，Dispatch到这个算子实现
     */
    // Register the NPU implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
    {
        m.impl("add", add_npu);
    }

    }  // namespace Add
    }  // namespace adt_ops

    ```

4. 参考[安装步骤](#安装步骤--installation-steps)章节重新构建Wheel包并安装。
5. 基于pytest测试算子API，请参考[test_add.py](tests/add/test_add.py)的实现。

# ADT Triton

## 简介

本文档演示如何在NPU上开发triton算子。

**核心优势:**

- **单交付件：** 一个文件完成算子开发和PyTorch框架适配。
- **高效调用：** 使用`import`语法启动核函数，流程简单高效。

## 测试步骤 | Installation Steps

1. 进入`ops/triton`目录。

2. 执行测试。

    ```sh
    python tests/test_l2norm.py
    ```

## 开发指南：新增一个算子 | Developer Guide: Adding a New Operator

为了实现一个新算子(如`add`)，您只需要提供一个python实现即可。

1. 首先您需要在ops/triton目录下使用算子名`add`建立add.py，在其中实现调用即可。

# 📝相关信息

- [安全声明](SECURITY.md)
- [许可证](LICENSE)

# 🙏致谢

本项目的部分实现参考了 [ops-transformer](https://gitcode.com/cann/ops-transformer) 仓库，感谢华为 CANN 社区及相关开发团队的开源贡献。
