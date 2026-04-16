# compiler-arch-notes

本仓库用于分享编译器设计与系统架构相关的知识点和文章。

## 目录结构


## Compiler

- **[Tiling 大小设计：GPU vs NPU 深度解析](./compiler/tiling-analysis.md)** - 分析GPU和NPU的Tiling策略差异
- **[MLIR WalkResult & WalkOrder 详解](./compiler/mlir-walk-guide.md)** - MLIR遍历机制的详细指南
- **[MLIR Traits 和 Interface 详解](./compiler/mlir-traits-interface.md)** - MLIR中Traits和Interface的详细介绍

## PyTorch

- **[PyTorch torch.compile 核心机制详解](./pytorch/torch-compile-core.md)** - PyTorch 2.0核心编译技术解析

## LLM优化

- **[kv cache 原理及优化概述](https://www.armcvai.cn/2024-11-01/kv-cache-optimize.html)** - 详细介绍KV Cache的原理和优化技术
- **[flashattention1 论文解读](https://www.armcvai.cn/2024-10-02/flashattention1-paper.html)** - FlashAttention算法的深入解析
- **[张量并行技术详解](https://www.armcvai.cn/2025-04-10/tensor-parallelism.html)** - 张量并行技术的原理和实现
- **[vllm 优化之 PagedAttention 源码解读](https://www.armcvai.cn/2024-12-06/vllm-pagedattention.html)** - VLLM中PagedAttention的源码分析


## 知识点

### 编译技术

- **[LLVM 的字符串处理类与 std::string 的区别](./compiler/llvm-string-classes.md)** - LLVM字符串处理类的优势和使用场景


### C++ 八股文

- **[智能指针详解](./c++/smart-pointers.md)** - C++三种智能指针的详细介绍和使用场景
- **[多态详解](./c++/polymorphism.md)** - C++多态的实现方式和原理
- **[dyn_cast 模板详解](./c++/dyn-cast-template.md)** - LLVM/MLIR中dyn_cast模板的原理和使用
- **[四种 Cast 转换的区别与应用场景](./c++/cast-conversions.md)** - C++类型转换的详细解析
- **[C++11 核心机制详解](./c++/c++11-core-mechanisms.md)** - C++11的核心特性和机制


### 操作系统