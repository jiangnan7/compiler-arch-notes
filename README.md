# compiler-arch-notes

本仓库分析和分享编译器设计、系统架构、机器学习框架等领域的核心技术知识点与深度文章。内容涵盖从底层编译原理到前沿模型优化的多个技术方向，是之前学习的一些记录，大部分由AI生成，部分手动编辑。

## 目录结构

## 推荐资源

### 1. 计算机体系结构与底层硬件 (Computer Architecture & Hardware) 
- **[alexandre-lecoq/awesome-computer-architecture-learning](https://github.com/alexandre-lecoq/awesome-computer-architecture-learning)** - 一份极佳的体系结构学习清单，涵盖了 CPU 设计原理、硬件描述语言（Verilog/VHDL）、数字逻辑基础，以及各种开源硬件仿真器的工具链资源。
- **[fengyln/Awesome-AI-Accelerator](https://github.com/fengyln/Awesome-AI-Accelerator)** - 专注于 AI 硬件加速器设计与计算机体系结构四大顶会（ISCA, MICRO, HPCA, ASPLOS）的权威论文汇总，深入硬件底层，细分归类了如 Google TPU、各类 NPU、脉动阵列、存算一体 PIM、SRAM/DRAM 数据流调度等领域特定架构（DSA）的学术界与工业界前沿研究。

### 2. AI 编译器与张量计算 (AI Compilers & Tensor Computation) 
- **[merrymercy/awesome-tensor-compilers](https://github.com/merrymercy/awesome-tensor-compilers)** - 张量计算和深度学习编译器的权威文献与项目汇总，详细对标了各个图级优化、自动调优（Auto-tuning）、计算代数以及算子生成技术的学术论文及源码。
- **[ilya-palachev/awesome-ai-compilers](https://github.com/ilya-palachev/awesome-ai-compilers)** - 极具价值的 AI 编译器综述列表，重点细分了针对 NPU/TPU 等专用芯片的编译器设计、软硬件协同设计，以及相关微架构的深入研究。
- **[zwang4/awesome-machine-learning-in-compilers](https://github.com/zwang4/awesome-machine-learning-in-compilers)** - 专注于"将机器学习技术应用于编译器优化（ML for Compilers）"的资源汇总，探讨如何利用深度学习/强化学习解决传统编译器中的启发式难题（如寄存器分配、指令调度、相位排序 Phase Ordering 等），是连接 ML 与编译技术的交叉领域宝库。

### 3. 深度学习系统全栈 (SysML & Deep Learning Systems) 
- **[zengzhongjie/Awesome-System-for-Machine-Learning](https://github.com/zengzhongjie/Awesome-System-for-Machine-Learning)** - 专注于 SysML（机器学习系统）的顶级学术论文与开源代码汇总，深入剖析了分布式训练的参数服务器架构、GPU 显存优化池化、张量并行/流水线并行底层的切分逻辑，以及硬件拓扑感知计算。
- **[microsoft/AI-System](https://github.com/microsoft/AI-System)** - 微软亚洲研究院（MSRA）开源的系统级人工智能学习路线图和文献库，精选了从深度学习框架（PyTorch 底层执行流）、AI 编译器（TVM/MindSpore 架构）、分布式训练底座，一直到底层硬件体系结构（GPU/NPU 拓扑）的顶级论文、讲义和工业界实践，是构建 SysML 大局观的天花板级别资料库。
- **[chenzomi12/DeepLearningSystem](https://github.com/chenzomi12/DeepLearningSystem)** - 极度硬核的中文"深度学习系统全栈"开源知识库，系统梳理了理论文档与必读论文，涵盖 AI 架构设计、底层算子（FlashAttention/CUDA 优化）、AI 编译器原理（图优化、MLIR 剖析）、推理引擎架构（TensorRT 等）及芯片底层架构。

### 4. 大语言模型系统与推理基础设施 (LLM Systems & Inference) 
- **[DefTruth/Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)** - 专注于大语言模型推理与系统优化的顶尖文献、技术博客和资源汇总，深挖底层系统设计，详细汇总了 K-V Cache 显存管理、PagedAttention 机制、张量/流水线并行（TP/PP）的切分策略、各类量化算法（PTQ/QAT 数学推导及系统实现）以及长文本上下文处理的系统级论文，是 LLM 架构师和系统工程师的必读清单。

## Compiler

- **[Tiling 大小设计：GPU vs NPU 深度解析](./compiler/tiling-analysis.md)** - 分析GPU和NPU的Tiling策略差异

## GPU 

- **[GPU相关技术专栏](https://zhuanlan.zhihu.com/c_1437330196193640448)** - 知乎专栏：GPU相关技术系列文章

## MLIR

- **[Pattern Rewriting 与 Dialect Conversion 的机制分析与应用边界](./mlir/pattern-rewriting-dialect-conversion.md)** - MLIR编译基础设施的深度解析
- **[SpeculatableOpInterface 与推测执行机制](./mlir/speculatable-op-interface.md)** - MLIR推测执行机制的详细解析
- **[MLIR WalkResult & WalkOrder 详解](./mlir/mlir-walk-guide.md)** - MLIR遍历机制的详细指南
- **[MLIR Traits 和 Interface 详解](./mlir/mlir-traits-interface.md)** - MLIR中Traits和Interface的详细介绍

## PyTorch

- **[PyTorch torch.compile 核心机制详解](./pytorch/torch-compile-core.md)** - PyTorch 2.0核心编译技术解析
- **[PyTorch 编译器动态形状支持在异构硬件架构下的深度技术分析](./pytorch/dynamic-shapes-heterogeneous-analysis.md)** - 动态形状在异构硬件上的技术分析
- **[PyTorch Dynamic Shapes 文档深度分析](./pytorch/dynamic-shapes-documentation-analysis.md)** - 动态形状文档的详细分析
- **[PyTorch 编译器动态形状支持的深度理论分析](./pytorch/dynamic-shapes-theoretical-analysis.md)** - 动态形状的理论基础和MLIR实现

## LLM优化

- **[kv cache 原理及优化概述](https://www.armcvai.cn/2024-11-01/kv-cache-optimize.html)** - 详细介绍KV Cache的原理和优化技术
- **[flashattention1 论文解读](https://www.armcvai.cn/2024-10-02/flashattention1-paper.html)** - FlashAttention算法的深入解析
- **[张量并行技术详解](https://www.armcvai.cn/2025-04-10/tensor-parallelism.html)** - 张量并行技术的原理和实现
- **[vllm 优化之 PagedAttention 源码解读](https://www.armcvai.cn/2024-12-06/vllm-pagedattention.html)** - VLLM中PagedAttention的源码分析
- **[Batch Size 与 Sequence Length 解析](./llm/batch-size-sequence-length-analysis.md)** - LLM中批大小和序列长度的深度分析
- **[从 LayerNorm 到 RMSNorm 的理论与底层优化机制分析](./llm/layernorm-rmsnorm-analysis.md)** - LLM归一化策略的优化分析


## 知识点

### 编译技术

- **[LLVM 的字符串处理类与 std::string 的区别](./compiler/llvm-string-classes.md)** - LLVM字符串处理类的优势和使用场景


### C++ 八股文

- **[智能指针详解](./C++/smart-pointers.md)** - C++三种智能指针的详细介绍和使用场景
- **[多态详解](./C++/polymorphism.md)** - C++多态的实现方式和原理
- **[dyn_cast 模板详解](./C++/dyn-cast-template.md)** - LLVM/MLIR中dyn_cast模板的原理和使用
- **[四种 Cast 转换的区别与应用场景](./C++/cast-conversions.md)** - C++类型转换的详细解析
- **[C++11 核心机制详解](./C++/c++11-core-mechanisms.md)** - C++11的核心特性和机制
- **[模板特化机制与典型工程陷阱深度解析](./C++/template-specialization.md)** - C++模板特化的原理和工程实践
- **[C++ 核心机制详解](./C++/cpp-core-mechanisms.md)** - C++右值引用、const机制、并发编程等核心机制解析
- **[std::vector vs llvm::SmallVector 对比分析](./C++/vector-vs-smallvector.md)** - C++标准容器与LLVM优化容器的对比


### 操作系统