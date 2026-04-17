# compiler-arch-notes

本仓库分析和分享编译器设计、系统架构、机器学习框架等领域的核心技术知识点与深度文章。内容涵盖从底层编译原理到前沿模型优化的多个技术方向，是之前学习的一些记录，大部分由AI生成，部分手动编辑。

## 目录结构


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