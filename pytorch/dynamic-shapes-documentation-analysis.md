# PyTorch Dynamic Shapes 文档深度分析报告

## 1. 文档概览

### 1.1 文档定位与背景

PyTorch Dynamic Shapes 文档是 `torch.compiler` 用户指南的核心组成部分，位于 PyTorch 2.x 编译器技术栈的关键位置。该文档系统性地阐述了如何让编译后的模型高效处理不同形状的输入张量，是理解和使用 PyTorch 编译器进行生产级优化的必读材料 [1]。

### 1.2 解决的核心问题

在深度学习的实际应用中，模型经常需要处理变长输入。例如，自然语言处理任务中句子长度各异，推荐系统中用户行为序列长度不一。在默认的静态编译模式下，PyTorch 假设张量形状固定，任何维度变化都会触发昂贵的重新编译过程。动态形状机制通过符号化推理，允许单个编译图适配多种输入尺寸，从而显著降低编译开销，提升推理吞吐量 [1]。

## 2. 整体框架图谱

### 2.1 文档体系架构

Dynamic Shapes 主文档与其相关链接构成了一个完整的技术知识体系，可划分为两大核心分支：

```
                    ┌─────────────────────────────────────────┐
                    │   torch.compiler_dynamic_shapes.html   │
                    │          (Dynamic Shapes 主文档)         │
                    └─────────────────┬───────────────────────┘
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           │                                                      │
           ▼                                                      ▼
┌─────────────────────────┐                          ┌─────────────────────────┐
│  Dynamic Shapes 专题系列  │                          │  torch.compiler 核心架构 │
│      (深度指南 × 5)       │                          │      (基础组件 × 5)       │
└───────────┬─────────────┘                          └───────────┬─────────────┘
            │                                                    │
    ┌───────┼───────┬───────┬───────┐                ┌───────┬───┴───┬───────┬───────┐
    ▼       ▼       ▼       ▼       ▼                ▼       ▼       ▼       ▼       ▼
 Core    Trouble  Advanced Beyond  Debug         torch.  torch.  Fine-  Fake   AOT
Concepts shooting Options  Basics  tlparse      compiler export  grain  Tensor Inductor
```

### 2.2 文档层级关系表

|分支|文档名称|核心职责|
|:---|:---|:---|
|专题系列|Core Concepts|符号整数、Guards、ShapeEnv等底层原理|
|专题系列|Troubleshooting|调试方法与常见错误排查|
|专题系列|Advanced Options|PGO、分布式Collective等高级控制|
|专题系列|Beyond the Basics|0/1特化、Unbacked Symints进阶话题|
|专题系列|Debugging tlparse|日志解析与PGO标记识别|
|核心架构|torch.compiler API|TorchDynamo、TorchInductor入口|
|核心架构|torch.export|AOT全图捕获与部署导出|
|核心架构|Fine-grain APIs|图断裂控制与编译状态查询|
|核心架构|Fake Tensor|符号张量与无数据推理机制|
|核心架构|AOTInductor|服务端C++部署方案|

## 3. 主页面核心内容详解

### 3.1 动态形状定义与核心优势

动态形状（Dynamic Shapes）允许编译器生成的计算图处理不同维度的输入，而无需为每种新形状重新编译。这一机制的核心价值在于：编译一次，适配多种输入尺寸，从而将编译开销从 O(n) 降至 O(1)，其中 n 为不同形状的数量 [1]。

### 3.2 三大核心适用场景

文档明确指出动态形状适用于以下三类典型场景：

|场景类型|具体描述|典型应用|
|:---|:---|:---|
|变量维度|张量某一维度在推理时可变|NLP中变长句子、不同Batch Size|
|数据依赖型输出|输出形状取决于输入数据的具体数值|条件筛选、非极大值抑制|
|稀疏表示|处理非规则、稀疏结构的数据|图神经网络、稀疏注意力|

### 3.3 静态与动态模式对比

在 `torch.compile` 中，`dynamic` 参数是控制编译行为的关键开关。以下代码展示了两种模式的本质差异：

```python
import torch

# 静态模式（默认）：dynamic=False
# 每次形状变化都会触发重新编译
@torch.compile(dynamic=False)
def static_fn(x):
    return x * x.size()[0]

static_fn(torch.rand(10))  # 第一次编译
static_fn(torch.rand(20))  # 形状变化，触发重新编译

# 动态模式：dynamic=True
# 单次编译适配多种形状
@torch.compile(dynamic=True)
def dynamic_fn(x):
    return x * x.size()[0]

dynamic_fn(torch.rand(10))  # 第一次编译
dynamic_fn(torch.rand(20))  # 复用已编译图，无需重新编译
```

### 3.4 特化（Specialization）机制

特化是编译器针对特定形状条件生成优化代码的过程。当代码中存在基于形状的控制流时，编译器会记录当前分支条件作为 Guard。后续输入若违反该条件，则触发重新编译以生成新的特化版本：

```python
@torch.compile(dynamic=True)
def conditional_fn(x):
    if x.size()[0] == 10:    # Guard: size[0] == 10
        return x * 10
    if x.size()[0] <= 30:    # Guard: size[0] <= 30
        return x * 200
    return x * x.size()[0]   # 通用分支
```

### 3.5 关键API详解

PyTorch 提供了三个核心 API 用于精细控制动态行为：

|API|功能|行为特点|
|:---|:---|:---|
|`mark_dynamic(tensor, dim)`|显式标记维度为动态|若编译器尝试特化该维度则报错|
|`maybe_mark_dynamic(tensor, dim)`|建议性标记为动态|允许编译器在必要时进行特化|
|`mark_unbacked(tensor)`|标记无后备存储的动态维度|用于形状完全由运行时决定的场景|

`mark_dynamic` 的典型使用模式如下：

```python
import torch

@torch.compile
def f(x):
    return x * x.size()[0]

x = torch.randn(10)
# 在调用编译函数前标记第0维为动态
torch._dynamo.mark_dynamic(x, 0)
f(x)  # 编译器知悉dim=0是动态的
```

### 3.6 环境变量TORCH_COMPILE_DYNAMIC_SOURCES

通过设置 `TORCH_COMPILE_DYNAMIC_SOURCES` 环境变量，用户可以全局控制动态行为的优先级。该配置支持整数常量和张量大小的动态化处理，适用于需要批量调整动态策略的场景 [1]。

## 4. 相关链接内容深度分析

### 4.1 Dynamic Shapes 专题系列

#### 4.1.1 Core Concepts：符号化推理基础

动态形状的实现依赖于一套完整的符号化推理体系 [2]：

|核心组件|功能描述|
|:---|:---|
|Symbolic Integers (Symints)|用符号变量（如`s0`）表示可变维度，支持数学推理|
|Guards机制|编译图有效性的检查点，违反条件时触发重编译|
|Runtime Asserts|通过`torch._check`提供运行时约束提示|
|Hint Values|利用运行时已知数值进行局部优化|
|ShapeEnv|形状环境管理器，通过FX IR和Sympy表达式传播尺寸信息|

#### 4.1.2 Troubleshooting：调试与错误排查

针对动态形状的常见问题，文档推荐使用 `TORCH_LOGS=dynamic` 环境变量启用详细日志，重点关注 `GuardOnDataDependentSymNode` 错误，该错误通常表明代码中存在数据依赖的形状操作，需要通过 `mark_unbacked` 或代码重构来解决 [3]。

#### 4.1.3 Advanced Options：高级控制策略

|高级特性|描述|启用方式|
|:---|:---|:---|
|Profile-Guided Optimization|序列化动态决策并跨作业重用|`TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO=1`|
|Compiler Collective|分布式环境下跨Rank共享形状信息|SPMD模式自动启用|
|减少编译策略|通过约束提示减少特化版本数量|`torch._check`约束|

#### 4.1.4 Beyond the Basics：进阶话题

进阶内容涵盖两个关键话题 [5]：

- **0/1特化问题**：编译器默认对维度值为0或1的情况进行特化，因为这些值通常具有特殊语义（如空张量、标量）
- **Backed vs Unbacked Symints**：Backed Symints 有对应的具体张量作为后备，而 Unbacked Symints 完全依赖符号推理，后者常见于动态输出形状的场景

#### 4.1.5 Debugging with tlparse

tlparse 是 PyTorch 提供的日志解析工具，用于分析编译器日志中的动态决策记录，帮助识别哪些维度被 PGO 自动标记为动态 [6]。

### 4.2 torch.compiler 核心架构

#### 4.2.1 torch.compiler API：编译器入口

`torch.compiler` 是 PyTorch 2.x 编译器技术栈的统一命名空间，包含以下核心组件 [7]：

|组件|功能|技术特点|
|:---|:---|:---|
|TorchDynamo|图捕获引擎|利用CPython Frame Evaluation API安全捕获|
|TorchInductor|默认代码生成后端|基于OpenAI Triton支持多GPU架构|
|AOT Autograd|自动微分捕获|提前捕获前向与反向传播逻辑|

支持的 Backends 包括：`inductor`、`cudagraphs`、`tensorrt`、`openvino` 等。

#### 4.2.2 torch.export：AOT全图捕获

`torch.export` 与 `torch.compile` 的核心区别在于 [8]：

|特性|torch.compile (JIT)|torch.export (AOT)|
|:---|:---|:---|
|捕获时机|运行时即时编译|运行前全图捕获|
|图完整性|允许图断裂|要求完整无断裂|
|Python依赖|保留Python运行时|脱离Python独立执行|
|适用场景|开发调试、动态部署|生产部署、跨平台移植|

#### 4.2.3 Fine-grain APIs：细粒度控制

|API|功能|使用场景|
|:---|:---|:---|
|`torch.compiler.disable`|禁用特定函数编译|跳过不兼容代码段|
|`disallow_in_graph`|强制算子Eager执行|处理不可追踪操作|
|`allow_in_graph`|允许自定义算子入图|扩展编译器支持范围|
|`graph_break`|手动插入图断点|调试与隔离问题代码|
|`is_compiling`|检测编译状态|条件执行编译/Eager逻辑|

#### 4.2.4 Fake Tensor：符号张量机制

Fake Tensor 是一种不含实际数据但保留完整元信息的张量类型 [10]：

- 模拟设备属性（CPU/CUDA）而不消耗显存
- 支持符号化尺寸推导与形状传播
- 与 ShapeEnv 集成实现动态形状推理
- 与 Meta Tensor 概念相关但专注于编译器内部使用

#### 4.2.5 AOTInductor：服务端部署方案

AOTInductor 专为服务端推理部署设计 [11]：

```
torch.export → aoti_compile_and_package → .so共享库 → AOTIModelPackageLoader
```

- 支持 Python 和 C++ 两种加载方式
- 完整支持动态形状
- 生成独立于 Python 运行时的推理引擎

## 5. 技术关联图谱分析

### 5.1 组件依赖关系

```
┌────────────────────────────────────────────────────────────────────┐
│                        用户代码 (PyTorch Model)                     │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    TorchDynamo (图捕获层)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │
│  │ Frame Eval  │→ │  FX Tracer  │→ │ ShapeEnv + Symints + Guards │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘ │
└───────────────────────────────┬────────────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌─────────────────────┐                    ┌─────────────────────────┐
│   torch.compile     │                    │     torch.export        │
│   (JIT 即时编译)     │                    │    (AOT 全图导出)        │
└─────────┬───────────┘                    └───────────┬─────────────┘
          │                                            │
          ▼                                            ▼
┌─────────────────────┐                    ┌─────────────────────────┐
│   TorchInductor     │                    │     AOTInductor         │
│  (GPU代码生成)       │                    │   (部署共享库生成)        │
└─────────────────────┘                    └─────────────────────────┘
```

### 5.2 从开发到部署的完整工作流

|阶段|工具/API|动态形状相关操作|
|:---|:---|:---|
|开发调试|`torch.compile(dynamic=True)`|快速验证动态行为|
|性能调优|`mark_dynamic` + `TORCH_LOGS`|精细控制+问题诊断|
|高级优化|PGO + Compiler Collective|跨作业/跨Rank优化|
|模型导出|`torch.export`|带动态形状约束的全图捕获|
|生产部署|AOTInductor|生成支持动态形状的.so文件|

### 5.3 动态形状在编译器栈中的位置

动态形状机制贯穿整个编译器栈：

- **图捕获层**：通过 ShapeEnv 和 Symints 跟踪符号化形状
- **优化层**：Guards 机制确保优化假设有效
- **代码生成层**：TorchInductor 生成支持符号尺寸的内核
- **部署层**：AOTInductor 保留动态形状信息至推理阶段

## 6. 核心API与工具速查表

### 6.1 动态形状控制API

|API|所属模块|功能|参数|
|:---|:---|:---|:---|
|`mark_dynamic`|`torch._dynamo`|强制标记维度为动态|tensor, dim|
|`maybe_mark_dynamic`|`torch._dynamo`|建议性标记为动态|tensor, dim|
|`mark_unbacked`|`torch._dynamo`|标记无后备存储维度|tensor|
|`torch.compile`|`torch`|编译入口|dynamic=True/False/"auto"|

### 6.2 调试与诊断工具

|工具/环境变量|功能|使用方法|
|:---|:---|:---|
|`TORCH_LOGS=dynamic`|输出动态形状决策日志|设置环境变量|
|`TORCH_LOGS=+guards`|输出Guard添加详情|设置环境变量|
|`tlparse`|解析编译器日志|命令行工具|
|`torch._dynamo.explain`|解释编译行为|函数调用|

### 6.3 高级配置选项

|配置项|功能|默认值|
|:---|:---|:---|
|`TORCH_COMPILE_DYNAMIC_SOURCES`|全局动态优先级|未设置|
|`TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO`|启用本地PGO|0|
|`torch._dynamo.config.automatic_dynamic_shapes`|自动动态检测|True|

## 7. 总结与最佳实践建议

### 7.1 何时使用动态形状

动态形状适用于以下场景：

- 输入序列长度频繁变化（如NLP任务）
- Batch Size 需要动态调整
- 模型输出形状依赖输入数据
- 需要减少编译开销以提升整体吞吐量

**不建议使用**的场景：

- 输入形状固定且追求极致单次推理延迟
- 形状空间极大导致符号推理开销过高

### 7.2 推荐的启用方式

|场景|推荐方式|理由|
|:---|:---|:---|
|快速原型验证|`dynamic=True`|简单直接，覆盖全面|
|生产环境（已知动态维度）|`mark_dynamic`指定具体维度|精确控制，避免不必要的通用化|
|生产环境（不确定哪些维度动态）|Automatic Dynamic（默认）|编译器自动检测，平衡性能与灵活性|
|分布式训练|Compiler Collective|跨Rank统一动态决策|

### 7.3 常见问题排查流程

```
遇到频繁重编译
      │
      ▼
启用 TORCH_LOGS=dynamic 查看日志
      │
      ├─── Guard违反 ───► 检查是否有基于形状的控制流
      │                    考虑使用 mark_dynamic 或重构代码
      │
      ├─── 0/1特化 ────► 确认是否为边界情况
      │                  考虑填充或使用 maybe_mark_dynamic
      │
      └─── 数据依赖 ───► 检查 GuardOnDataDependentSymNode 错误
                        使用 mark_unbacked 或 torch._check
```

### 7.4 性能优化建议

1. **优先使用用户注解**：相比 `dynamic=True`，精确的 `mark_dynamic` 能让编译器生成更优化的代码
2. **启用PGO**：对于长期运行的服务，PGO 可以利用历史信息优化动态决策
3. **监控编译次数**：使用 `torch._dynamo.utils.CompileCounter` 追踪重编译频率
4. **合理设置约束**：通过 `torch._check` 提供形状范围提示，减少过度通用化

## 参考文献

[1] PyTorch Documentation, 2026-04-09. Dynamic Shapes. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html

[2] PyTorch Documentation, 2025-12-03. Dynamic Shapes Core Concepts. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html

[3] PyTorch Documentation, 2025-12-03. Troubleshooting Dynamic Shapes. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting.html

[4] PyTorch Documentation. Advanced Options to Control Dynamic Behavior. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_advanced_control_options.html

[5] PyTorch Documentation, 2025-12-03. Beyond the Basics. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_beyond_the_basics.html

[6] PyTorch Documentation. Debugging with tlparse. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_debugging_tlparse_torch_logs.html

[7] PyTorch Documentation. torch.compiler API. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html

[8] PyTorch Documentation. torch.export Guide. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html

[9] PyTorch Documentation. TorchDynamo APIs. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_fine_grain_apis.html

[10] PyTorch Documentation. Fake Tensor Mechanism. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_fake_tensor.html

[11] PyTorch Documentation. AOTInductor Backend. https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html