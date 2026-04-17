# compiler-arch-notes

本仓库专注于分析和分享编译器设计、系统架构、机器学习框架等领域的核心技术知识点与深度文章。内容涵盖从底层编译原理到前沿模型优化的多个技术方向，旨在为开发者提供系统化的技术参考。

## 目录结构

- **[c++/](./c++)** - C++语言核心机制与工程实践分析
- **[compiler/](./compiler/)** - 编译器设计与优化技术分析
- **[mlir/](./mlir/)** - MLIR中间表示与编译基础设施深度解析
- **[pytorch/](./pytorch/)** - PyTorch编译优化与动态形状支持分析
- **[llm/](./llm/)** - 大型语言模型(LLM)架构与优化技术分析
- **[system-architecture/](./system-architecture/)** - 系统架构设计与性能优化
- **[code-optimization/](./code-optimization/)** - 代码优化技术与最佳实践
- **[performance-tuning/](./performance-tuning/)** - 性能调优策略与工具分析

## Compiler

- **[Tiling 大小设计：GPU vs NPU 深度解析](./compiler/tiling-analysis.md)** - 分析GPU和NPU的Tiling策略差异

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

- **[智能指针详解](./c++/smart-pointers.md)** - C++三种智能指针的详细介绍和使用场景
- **[多态详解](./c++/polymorphism.md)** - C++多态的实现方式和原理
- **[dyn_cast 模板详解](./c++/dyn-cast-template.md)** - LLVM/MLIR中dyn_cast模板的原理和使用
- **[四种 Cast 转换的区别与应用场景](./c++/cast-conversions.md)** - C++类型转换的详细解析
- **[C++11 核心机制详解](./c++/c++11-core-mechanisms.md)** - C++11的核心特性和机制
- **[模板特化机制与典型工程陷阱深度解析](./c++/template-specialization.md)** - C++模板特化的原理和工程实践
- **[C++ 核心机制详解](./c++/cpp-core-mechanisms.md)** - C++右值引用、const机制、并发编程等核心机制解析
- **[std::vector vs llvm::SmallVector 对比分析](./c++/vector-vs-smallvector.md)** - C++标准容器与LLVM优化容器的对比


### 操作系统