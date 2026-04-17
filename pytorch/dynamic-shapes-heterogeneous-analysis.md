# PyTorch 编译器动态形状支持在异构硬件架构下的深度技术分析报告

## 1. 概述

### 1.1 动态形状问题背景

在深度学习的实际应用中，模型经常需要处理变长输入。自然语言处理任务中句子长度各异，推荐系统中用户行为序列长度不一，目标检测模型的输出数量依赖于输入图像内容。在默认的静态编译模式下，PyTorch 假设张量形状固定，任何维度变化都会触发昂贵的重新编译过程。动态形状（Dynamic Shapes）机制通过符号化推理，允许单个编译图适配多种输入尺寸，从而将编译开销从 O(n) 降至 O(1)，其中 n 为不同形状的数量 [1]。

动态形状的核心应用场景包括三大类：变量维度（自适应批处理或变长序列）、数据依赖型输出（如目标检测中数量不定的边界框）、以及稀疏表示（稀疏张量、图神经网络等）。PyTorch 通过符号化追踪张量维度（使用 SymInt 符号整数）而非具体数值来实现这一能力，但这也带来了额外的复杂性——编译器需要在不知道具体数值的情况下进行推理和优化。

### 1.2 异构硬件挑战概览

不同硬件架构对动态形状的支持存在显著差异。GPU 依赖 CUDA Graphs 和 TensorCore 的静态内存布局要求，而 NPU 通常采用图编译模式，对形状变化更为敏感。这种差异导致了各平台在编译流程、优化策略和性能表现上的显著分化，理解这些差异对于实现高效的跨平台部署至关重要。

|硬件类型|代表平台|核心挑战|主要应对策略|
|:---|:---|:---|:---|
|GPU|NVIDIA CUDA|TensorCore对齐、CUDA Graphs静态拓扑|Triton JIT、分桶策略|
|NPU|华为昇腾|图编译开销大、Tiling参数依赖形状|DVM虚拟机、动态算子更新|
|TPU|Google TPU|XLA强依赖静态形状|Padding到128倍数、Fused Eager模式|
|AI加速卡|Intel Gaudi/AMD MI|编译预热时间长|指数分桶、TunableOp离线调优|

## 2. MLIR 编译框架集成

### 2.1 torch-mlir 架构与分层方言设计

在 `torch-mlir` 项目中，动态形状的支持是连接 PyTorch 生态与 MLIR 生态的核心。其架构设计旨在将命令式的 PyTorch 程序转换为声明式的 MLIR 表达，并保留形状的动态性。torch-mlir 采用分层方言（Dialect）设计，最上层是 **Torch Dialect**，它与 PyTorch 的 JIT IR 或 ATen 算子几乎一一对应。随后，通过一系列转换 Pass，IR 被降低到 **Linalg Dialect**（线性代数方言），这是实现硬件无关优化的关键层 [5][13]。

Linalg 方言的设计哲学是将高级张量操作分解为通用的线性代数原语（`linalg.generic`），利用 Affine 表达式处理动态索引。这种设计使得同一套优化 Pass 可以复用于不同的硬件后端，同时保留了对动态维度的完整支持。

### 2.2 ShapedType 与动态维度表示

在 MLIR 中，张量类型由 `RankedTensorType` 表示。动态维度使用特定的占位符 `?` 来标识，在 C++ API 中对应常数 `ShapedType::kDynamic` 或数值 `-1` [18][20]。这种表示方式允许编译器在不知道具体数值的情况下进行类型推断和优化。

|形状类型|MLIR表示|说明|
|:---|:---|:---|
|完全静态|`tensor<4x8xf32>`|所有维度编译时已知|
|部分动态|`tensor<?x8xf32>`|batch维度动态，特征维度静态|
|完全动态|`tensor<?x?xf32>`|所有维度运行时确定|

### 2.3 FxImporter 转换流程

随着 PyTorch 2.0 的普及，`FxImporter` 成为主要的模型捕获路径。其流程从图捕获开始，利用 `torch.export` 或 Dynamo 捕获模型生成 `torch.fx.GraphModule`，然后进行算子标准化，将 FX 图中的算子标准化为 ATen 算子。接下来 `FxImporter` 遍历 FX 节点，将其映射为 Torch Dialect 中的操作。最后通过 `--convert-torch-to-linalg` 等 Pipeline，将 Torch 算子分解并转换为 Linalg 泛型操作 [5][11]。

### 2.4 MLIR 集成架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PyTorch 用户代码                                  │
│                    @torch.compile / torch.export                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TorchDynamo (图捕获层)                               │
│    ┌────────────────┐    ┌────────────────┐    ┌─────────────────────────┐  │
│    │ Frame Eval API │ →  │  FX Tracer     │ →  │ torch.fx.GraphModule    │  │
│    │ (字节码拦截)    │    │  (符号追踪)    │    │ (带SymInt的FX图)         │  │
│    └────────────────┘    └────────────────┘    └────────────┬────────────┘  │
└─────────────────────────────────────────────────────────────┼───────────────┘
                                                              │
                    ┌─────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         torch-mlir (FxImporter)                             │
│    ┌────────────────┐    ┌────────────────┐    ┌─────────────────────────┐  │
│    │ FX Node遍历    │ →  │ ATen算子映射   │ →  │ Torch Dialect IR        │  │
│    │               │    │               │    │ (torch.aten.*)          │  │
│    └────────────────┘    └────────────────┘    └────────────┬────────────┘  │
└─────────────────────────────────────────────────────────────┼───────────────┘
                                                              │
                    ┌─────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MLIR Lowering Pipeline                                 │
│    ┌────────────────┐    ┌────────────────┐    ┌─────────────────────────┐  │
│    │ Torch Dialect  │ →  │ Linalg Dialect │ →  │ 硬件特定方言             │  │
│    │ tensor<?x?xf32>│    │ linalg.generic │    │ (GPU/TPU/NPU IR)        │  │
│    └────────────────┘    └────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
    ┌───────────┐          ┌───────────┐          ┌───────────┐
    │ Triton IR │          │ LLVM IR   │          │ StableHLO │
    │ (NVIDIA)  │          │ (CPU)     │          │ (TPU/XLA) │
    └───────────┘          └───────────┘          └───────────┘
```

## 3. 运行时与编译时协同机制

### 3.1 SymInt 符号整数与 ShapeEnv 工作原理

为了在不确定具体数值的情况下进行推理，PyTorch 引入了 **SymInt**（符号整数）数据结构。它用符号变量（如 `s0`、`s1`）表示可变的张量维度，而非具体的整数值。当编译器遇到形状相关的操作时，SymInt 允许进行符号化的数学推理，而无需知道具体数值 [2]。

**ShapeEnv**（Shape Environment）是管理符号变量及其约束关系的核心组件。它通过 SymPy 表达式系统跟踪维度间的数学关系，并在 FX IR 中传播尺寸信息。当 Dynamo 追踪代码时，它不再将张量维度视为固定整数，而是分配一个 SymInt。例如，形状为 `[batch, 64]` 的张量会被记录为 `[s0, 64]` [11]。

|组件|功能描述|关键作用|
|:---|:---|:---|
|SymInt|符号化整数表示|替代具体数值，支持数学推理|
|ShapeEnv|形状环境管理器|跟踪约束关系，传播尺寸信息|
|Guards|编译图有效性检查|违反条件时触发重编译|
|Runtime Asserts|运行时约束检查|通过torch._check提供提示|
|Hint Values|运行时已知数值|用于局部优化决策|

### 3.2 Guard 机制与重编译触发

**Guard** 是确保编译产物正确性的运行时检查。Guards 机制负责验证编译图的有效性——当输入张量的实际形状违反预设的 Guard 条件时，系统将触发重新编译 [1][3]。

触发条件包括：如果输入张量的形状不满足编译时建立的假设（例如 `s0 > 2` 变为 `False`），则触发 Guard Failure。此外，如果代码中存在基于形状的条件分支（如 `if x.size()[0] == 10`），系统会为该条件量身定制图，若新输入不满足条件，也会触发重新编译。

### 3.3 多级缓存策略

PyTorch 维护多个层级的缓存以减少编译开销，形成了完整的缓存层次结构 [14][15]：

|缓存层级|名称|功能|持久化|
|:---|:---|:---|:---|
|L1|FXGraphCache|缓存生成的FX图|支持|
|L2|TritonCache|缓存编译后的Triton内核(.cubin)|支持|
|L3|PGO-cache|存储动态形状决策信息|支持|
|L4|AOTAutogradCache|缓存自动微分图|支持|

### 3.4 运行时协同机制示意图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            运行时协同机制                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │         输入张量 x              │
                    │    shape = [batch, seq, dim]   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Guard 检查层                                        │
│    ┌──────────────────────────────────────────────────────────────────────┐ │
│    │  检查条件: s0 > 0 && s1 > 0 && s2 == 64 && isinstance(x, Tensor)    │ │
│    └──────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
         [Guard Pass]                          [Guard Fail]
              │                                     │
              ▼                                     ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│     缓存查找              │          │     重新编译              │
│  ┌────────────────────┐  │          │  ┌────────────────────┐  │
│  │ FXGraphCache       │  │          │  │ Dynamo 重新追踪    │  │
│  │      ↓             │  │          │  │      ↓             │  │
│  │ TritonCache        │  │          │  │ ShapeEnv 更新      │  │
│  │      ↓             │  │          │  │      ↓             │  │
│  │ PGO-cache          │  │          │  │ 生成新 Guards      │  │
│  └────────────────────┘  │          │  └────────────────────┘  │
└────────────┬─────────────┘          └────────────┬─────────────┘
             │                                     │
             └──────────────────┬──────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         内核执行层                                           │
│    ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐  │
│    │ 形状参数传入   │ →  │ Triton Kernel │ →  │ 输出张量                   │  │
│    │ (s0=32,s1=128)│    │ Launch        │    │ shape=[32,128,64]         │  │
│    └───────────────┘    └───────────────┘    └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4. GPU 平台实现深度分析

### 4.1 TorchInductor 编译流程

TorchInductor 作为 PyTorch 2.0 的核心后端，通过符号化整数（SymInts）和 ShapeEnv 环境来处理动态形状。在 CUDA 平台上，Inductor 将 FX 图转换为 Triton IR，并生成参数化的 Triton 内核 [5]。编译流程涵盖了从高级 Python 代码到底层 GPU 指令的完整转换链路。

首先，FX Graph 携带 SymInt 信息进入 Lowering 阶段，在此阶段高级算子被分解为基础操作。然后这些操作被转换为 Triton IR，其中形状信息以符号参数形式保留。最后 Triton JIT 编译器将 IR 转换为 PTX/SASS 指令，实现最终的 GPU 执行代码。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GPU 编译流程详解                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FX Graph   │ ──→ │   Lowering   │ ──→ │  Triton IR   │
│ (带SymInt)    │     │  (算子分解)   │     │  (符号参数)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌───────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Triton JIT 编译器                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │ tt.program  │ → │ tt.divisib- │ → │ AxisInfo    │ → │ PTX/SASS    │      │
│  │ (入口函数)   │   │ ility 分析   │   │ Pass优化    │   │ 代码生成     │      │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                                 │
                     ┌───────────────────────────┘
                     ▼
┌──────────────────────────────────────────────┐
│             运行时执行                         │
│  ┌──────────┐    ┌──────────┐    ┌─────────┐ │
│  │ Guard检查 │ →  │ 形状参数   │ →  │ Kernel  │ │
│  │ (s0>0?)  │    │ 传入内核   │    │ Launch  │ │
│  └──────────┘    └──────────┘    └─────────┘ │
└──────────────────────────────────────────────┘
```

### 4.2 Triton JIT 代码生成与优化

Triton 编译器通过 JIT 机制处理动态维度，将张量形状作为运行时参数传入。为了优化性能，Triton 会根据 `tt.divisibility` 和 `tt.contiguity` 等提示进行对齐优化，确保内存访问的合并性 [10]。这种设计使得同一 Triton 内核可以处理不同尺寸的输入，而无需为每种尺寸重新编译底层 PTX 代码。

关键优化技术包括可除性分析（`tt.divisibility`），用于判断维度是否为特定值（如16）的倍数以启用向量化加载；连续性分析（`tt.contiguity`），用于识别内存布局的连续性以优化合并访问；以及 AxisInfo Pass，负责在 IR 级别传播和推理这些属性。

### 4.3 TensorCore 对齐约束

TensorCore 对动态形状的支持受到硬件对齐的严格限制。NVIDIA TensorCore 通常要求矩阵运算的 M/N/K 维度为 8 或 16 的倍数，以实现最优的计算吞吐量 [21]。编译器会根据实际维度值采取不同策略：

|情况|处理策略|性能影响|
|:---|:---|:---|
|维度为8/16倍数|直接使用TensorCore|最优性能|
|维度接近对齐值|自动Padding填充|轻微开销|
|维度严重不对齐|回退SIMD路径|显著性能下降|

### 4.4 CUDA Graphs 动态处理

CUDA Graphs 原生要求静态拓扑和固定内存地址，这与动态形状存在本质冲突。在 `torch.compile` 中，通过 `mode="reduce-overhead"` 开启 CUDA Graphs 时，编译器会将模型划分为静态和动态部分 [19]。对于动态 batch size，主要采用分桶策略——为每个预定义的 batch 尺寸（如 1, 2, 4, 8, 16, 32）预先捕获独立的 Graph，运行时根据实际输入选择最接近的桶。

## 5. NPU 平台实现对比分析

### 5.1 四大平台对比

不同 NPU 平台在动态形状支持上采用了差异化的技术路线：

|特性|华为昇腾|Google TPU|Intel Gaudi|AMD MI系列|
|:---|:---|:---|:---|:---|
|**核心框架**|torch_npu + CANN|TorchTPU + XLA|vLLM + Gaudi Plugin|ROCm + HIP|
|**动态处理**|DVM虚拟机|重编译/Padding|指数分桶|TunableOp调优|
|**IR格式**|GE Graph/Ascend IR|HLO|Gaudi IR|Triton/HIP|
|**编译策略**|图下沉+动态Tiling|强特化|持久化缓存|离线Shape调优|
|**效率提升**|11x (DVM)|50-100% (Fused)|80%编译时间↓|因模型而异|

### 5.2 华为昇腾实现细节

华为通过 `torch_npu` 插件适配 PyTorch，其动态形状处理依赖于 CANN 软件栈。核心技术包括 **aclGraph 图下沉机制**和 **DVM 虚拟机技术**。

aclGraph 使昇腾能够实现全图下沉，将计算图完整提交给 NPU 执行。对于动态形状，支持"动态算子更新"机制，在执行过程中根据输入形状实时计算 Tiling 参数 [20]。

针对动态形状编译慢的问题，华为推出了 DVM（Bytecode Virtual Machine）实时编译器。通过字节码虚拟机在 NPU 上直接执行动态逻辑，相比传统编译方式可提升 11 倍以上的效率 [18]。这一技术有效解决了图编译模式下形状变化导致的频繁重编译问题。此外，TorchAIR 工具将 PyTorch 2.0 的 `torch.compile` 接入昇腾架构，建议开启算子二进制包以避免在每个 Epoch 重新编译动态算子 [23][24]。

### 5.3 Google TPU 执行模式

TPU 强依赖静态形状，动态形状会导致 XLA 编译器频繁触发昂贵的重新编译。TorchTPU 引入了三种执行模式以平衡灵活性与性能 [6]：

|执行模式|特点|性能表现|适用场景|
|:---|:---|:---|:---|
|Debug Eager|单算子同步执行|最慢，便于调试|开发调试阶段|
|Strict Eager|异步单算子执行|中等|原型验证|
|Fused Eager|自动算子融合|较Strict提升50-100%|生产推理|

Google 官方建议将输入 Padding 到 128 的倍数以匹配 MXU Tile 大小，这是 TPU 硬件架构的核心优化策略 [12]。此外，XLA 采用"有界动态形状"（Bounded Dynamic Shape）方案，允许维度在 `[<=Max_Bound]` 范围内变化，通过实验发现该机制能将重编译次数从 102 次降至 49 次，训练时间缩短约 32% [19][21]。

### 5.4 Intel Gaudi 分桶策略

Intel Gaudi 采用分桶（Bucketing）策略来应对动态形状挑战。通过 `VLLM_EXPONENTIAL_BUCKETING` 环境变量，可以将动态长度映射到有限的指数级桶中（如 32, 64, 128, 256...），实测可减少 80% 的重新编译时间 [15]。这种策略在 vLLM 大语言模型推理场景中尤为有效。此外，`PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES` 标志支持自动分桶，进一步简化了配置流程 [17][18]。

### 5.5 AMD MI 系列适配

AMD MI 系列通过 ROCm 软件栈适配 PyTorch。针对动态形状，AMD 提供了 `TunableOp` 工具进行离线形状调优——预先为预期的形状范围生成最优内核配置，并将调优结果持久化到磁盘。运行时根据实际形状查表选择最优配置 [16]。此外，`hipCUB` 库提供了针对不规则形状的高效扫描和排序原语。

## 6. GPU/NPU 编译流程对比

### 6.1 IR 层级差异对比图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      多平台编译流程对比                                       │
└─────────────────────────────────────────────────────────────────────────────┘

【NVIDIA GPU (TorchInductor)】
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Python    │ →  │  FX Graph  │ →  │ Triton IR  │ →  │  PTX/SASS  │
│  Model     │    │ (SymInt)   │    │ (符号参数)  │    │  Kernel    │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                        │
                  SymInt符号参数贯穿全流程，运行时传入具体值

【华为昇腾 NPU】
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Python    │ →  │  FX Graph  │ →  │  GE Graph  │ →  │ Ascend IR  │
│  Model     │    │            │    │ (CANN)     │    │  Kernel    │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                                          │
                              DVM虚拟机处理动态分支，支持运行时Tiling

【Google TPU (XLA)】
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Python    │ →  │  FX Graph  │ →  │ StableHLO  │ →  │    HLO     │
│  Model     │    │            │    │            │    │  Kernel    │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                                          │
                          静态编译为主，动态形状触发重编译或Padding

【Intel Gaudi】
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Python    │ →  │  FX Graph  │ →  │  Gaudi IR  │ →  │   Gaudi    │
│  Model     │    │            │    │            │    │  Kernel    │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                                          │
                              分桶后按静态形状处理，减少80%编译时间
```

### 6.2 动态处理策略对比

|维度|TorchInductor (GPU)|PyTorch/XLA (TPU)|torch_npu (昇腾)|
|:---|:---|:---|:---|
|**核心IR**|FX Graph / Triton IR|HLO (High-Level Optimizer)|GE Graph / Ascend IR|
|**符号化支持**|完整SymInt系统|无原生支持|部分支持|
|**动态处理**|符号化推理+Guards|重新编译/Padding|动态Tiling/DVM虚拟机|
|**特化策略**|默认特化，失败后转动态|强特化，依赖Padding|混合模式，支持图下沉|
|**重编译代价**|中等（Guard失败）|高（完整XLA编译）|低（DVM热更新）|

## 7. 性能量化数据与对比分析

### 7.1 编译时间与延迟数据

根据多项基准测试，动态形状的支持在带来灵活性的同时也伴随着一定的开销。首次编译（Cold Start）通常需要 **7-17 秒**，其中包含 Triton 内核自动调优的时间 [25][27]。编译后的模型在推理阶段可实现 **15-35%** 的单 Token 延迟降低 [25]。

然而，在某些极端场景下（如 `reduce-overhead` 模式处理变长序列），重编译可能导致单次迭代耗时从 1 秒飙升至 **115-565 秒** [27]。这凸显了精细化动态形状控制的重要性。

|指标|数值|说明|
|:---|:---|:---|
|首次编译时间|7-17秒|包含Triton自动调优|
|单Token延迟改进|15-35%|编译模式vs Eager模式|
|dynamic=True开销|~10x|vs原生FlashInfer|
|重编译最坏情况|115-565秒|reduce-overhead变长序列|

### 7.2 编译器加速对比

不同编译器框架在动态形状负载下的加速效果存在显著差异：

|编译器|测试模型|加速比|内存优化|
|:---|:---|:---|:---|
|BladeDISC|BERT|1.1x - 1.2x|-|
|BladeDISC|Stable Diffusion|2.42x - 3.05x|80%内存降低|
|DISC (MLIR)|混合负载|3.3x - 6.95x|-|
|TorchInductor|通用模型|1.5x - 2.5x|中等|

### 7.3 各平台性能优化效果

|优化策略|平台|效果量化|适用场景|
|:---|:---|:---|:---|
|指数分桶|Intel Gaudi|减少80%编译时间|LLM推理|
|DVM虚拟机|华为昇腾|11x效率提升|动态控制流模型|
|Fused Eager|Google TPU|50-100%性能提升|生产推理|
|有界动态形状|XLA/TPU|重编译从102次降至49次|训练场景|
|mark_dynamic精确标记|通用|避免不必要重编译|已知动态维度|

## 8. 完整实现示例代码

### 8.1 mark_dynamic 显式标记示例

```python
import torch
import torch._dynamo

def model_fn(x):
    return x * x.size()[0]

# 创建输入张量
x = torch.randn(8, 16)

# 在编译前标记第0维（batch维度）为动态
torch._dynamo.mark_dynamic(x, 0)

# 编译模型
compiled_model = torch.compile(model_fn)

# 首次调用触发编译
compiled_model(x)

# 不同batch size无需重编译
compiled_model(torch.randn(16, 16))  # 复用已编译图
compiled_model(torch.randn(32, 16))  # 复用已编译图
```

### 8.2 torch.export 动态导出示例

```python
from torch.export import export, Dim
import torch

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()
example_input = torch.randn(8, 16)

# 定义 batch 维度范围为 1 到 1024
batch = Dim("batch", min=1, max=1024)
dynamic_shapes = {"x": {0: batch}}

# 导出带动态形状约束的模型
exported = export(model, (example_input,), dynamic_shapes=dynamic_shapes)

# 编译为共享库（AOTInductor）
torch._inductor.aoti_compile_and_package(
    exported, 
    package_path="/usr/local/app/workspace/model.pt2"
)
```

### 8.3 华为昇腾 TorchAIR 配置示例

```python
import torch
import torch_npu
import torchair

# 创建编译器配置
config = torchair.CompilerConfig()

# 获取昇腾 NPU 后端
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 定义模型
class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x.sum(dim=-1))

model = MyModel().npu()

# 使用 torch.compile 接入昇腾图编译器
compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)

# 执行推理
x = torch.randn(32, 128).npu()
output = compiled_model(x)
```

### 8.4 各平台环境变量配置示例

```python
import os

# ==================== 通用配置 ====================
# 启用动态形状调试日志
os.environ["TORCH_LOGS"] = "dynamic"

# 启用 Profile-Guided Optimization
os.environ["TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO"] = "1"

# ==================== NVIDIA GPU ====================
# CUDA Graphs 跳过动态图
os.environ["TORCH_INDUCTOR_TRITON_CUDAGRAPH_SKIP_DYNAMIC_GRAPHS"] = "True"

# 指定编译缓存目录
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/usr/local/app/workspace/.cache/torch_compile"

# ==================== Intel Gaudi ====================
# 启用指数分桶策略
os.environ["VLLM_EXPONENTIAL_BUCKETING"] = "True"

# 启用动态形状细化
os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"

# ==================== 华为昇腾 ====================
# 启用算子二进制包（避免重复编译）
os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"

# ==================== Google TPU ====================
# 设置 XLA 设备
os.environ["PJRT_DEVICE"] = "TPU"
```

## 9. 最佳实践与问题排查

### 9.1 动态形状启用决策矩阵

|场景特征|推荐策略|理由|
|:---|:---|:---|
|输入形状完全固定|dynamic=False（默认）|最大化单次推理性能|
|batch size可变，其他固定|mark_dynamic标记batch维|精确控制，避免过度泛化|
|序列长度频繁变化|分桶策略+编译缓存|平衡灵活性与编译开销|
|形状完全动态且不可预测|dynamic=True+PGO|让编译器自动学习模式|
|生产部署（已知形状范围）|torch.export+AOTInductor|脱离Python，最优部署性能|

### 9.2 问题排查流程

当遇到动态形状相关的性能问题时，建议按以下流程排查：

**Step 1: 启用诊断日志**
设置 `TORCH_LOGS=dynamic` 查看动态决策详情，识别哪些维度被标记为动态，哪些触发了重编译。

**Step 2: 识别重编译原因**
- Guard 违反 → 检查基于形状的控制流，考虑重构或使用 `mark_dynamic`
- 0/1 特化 → 确认是否为边界情况，考虑 Padding 或 `maybe_mark_dynamic`
- 数据依赖 → 检查 `GuardOnDataDependentSymNode` 错误，使用 `mark_unbacked`

**Step 3: 验证优化效果**
使用 `torch._dynamo.utils.CompileCounter` 监控重编译频率，确认优化措施是否生效。

### 9.3 各平台适配建议

|目标平台|核心建议|关键配置|
|:---|:---|:---|
|NVIDIA GPU|优先使用Triton，注意TensorCore对齐|确保M/N/K为8/16倍数|
|华为昇腾|启用DVM模式处理动态控制流|检查CANN版本兼容性|
|Google TPU|Padding到128倍数，使用Fused Eager|避免频繁形状变化|
|Intel Gaudi|配置指数分桶，预热编译缓存|VLLM_EXPONENTIAL_BUCKETING=True|
|AMD MI|离线TunableOp调优预期形状范围|持久化调优结果|

## 10. 参考文献

[1] PyTorch Documentation, 2026-04-09. Dynamic Shapes. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html

[2] PyTorch Documentation, 2025-12-03. Dynamic Shapes Core Concepts. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html

[3] PyTorch Documentation, 2025-12-03. Troubleshooting Dynamic Shapes. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting.html

[4] PyTorch Documentation. Advanced Options to Control Dynamic Behavior. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html

[5] ezyang's blog, 2025-08-13. State of torch.compile for training (August 2025). https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025

[6] Google Blog, 2026-04-07. TorchTPU: Running PyTorch Natively on TPUs at Google Scale. https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale

[10] Triton-lang GitHub, 2026-01-23. Deep dive into JIT Cache Eviction and AxisInfo Pass. https://github.com/triton-lang/triton/issues/9298

[11] PyTorch Documentation, 2024-04-02. Dynamo Deep-Dive. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html

[12] LinkedIn, 2026-02-16. TPU Training: Top Takeaways from PyTorch to XLA Migration. https://linkedin.com/posts/ivan-nardini_machinelearning-artificialintelligence-activity-7429185342207062016-lLwb

[13] GitHub, 2024-07-08. [TorchToLinalg] Add lowering of torch.aten.pixel_unshuffle. https://github.com/llvm/torch-mlir/issues/4260

[14] PyTorch Documentation, 2024-06-20. Compile Time Caching in torch.compile. https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html

[15] vLLM Documentation, 2025-12-06. Managing and Reducing Warm-up Time on Gaudi. https://docs.vllm.ai/projects/gaudi/en/0.11.2/configuration/warm-up/managing_warm-up.html

[16] AMD ROCm Blogs, 2026-02-24. PyTorch Offline Tuning with TunableOp. https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop-offline/README.html

[17] Habana Docs, 2025. Handling Dynamic Shapes — Gaudi Documentation. https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Dynamic_Shapes.html

[18] arXiv, 2026-04-03. DVM: A Bytecode Virtual Machine Approach for Dynamic Tensor Computation. https://arxiv.org/html/2603.24239v2

[19] GPU Notes, 2025-12-23. How CUDA Graph Works in torch.compile. https://fkong.tech/posts/2025-12-23-cuda-graph-in-torch-compile

[20] GitHub Issues, 2025-10-26. aclGraph Instructions And Precaution. https://github.com/sgl-project/sglang/issues/12194

[21] Emergent Mind, 2026-02-05. NVIDIA Tensor Core Programmability. https://emergentmind.com/topics/nvidia-tensor-core-programmability

[22] Alibaba BladeDISC Releases, 2023-03-26. Release Notes 0.4.0. https://github.com/alibaba/BladeDISC/releases

[23] 昇腾社区, 2026-02-25. 使用TorchAIR进行模型图编译推理优化. https://ascendai.csdn.net/699e9cb554b52172bc5d8ac4.html

[24] 华为云, 2023-10-12. NPU上PyTorch模型调优问题案例. https://bbs.huaweicloud.com/blogs/412890

[25] GitHub, 2025. torch-compile-benchmarks. https://github.com/TheRootOf3/torch-compile-benchmarks

[27] GitHub Issue #128424, 2024-06-11. very long compile time + GPU memory grow. https://github.com/pytorch/pytorch/issues/128424

[28] arXiv, 2021-11-23. DISC: A Dynamic Shape Compiler for Machine Learning Workloads. https://arxiv.org/abs/2103.05288