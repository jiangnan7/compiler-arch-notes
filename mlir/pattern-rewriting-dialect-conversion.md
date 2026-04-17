# MLIR 编译基础设施：Pattern Rewriting 与 Dialect Conversion 的机制分析与应用边界

在 MLIR（Multi-Level Intermediate Representation）的编译器设计中，中间表示的变换（IR Transformation）是核心环节。为了应对从高阶语义到机器码的多层级变换需求，MLIR 提供了两套密切相关但设计目标迥异的基础设施：**Pattern Rewriting（模式重写框架）** 与 **Dialect Conversion（方言转换框架）**。本文旨在从系统架构、合法化约束（Legalization）、类型系统演进以及 API 设计等维度，深入剖析两者的机制差异及工程适用场景。

## 1. 核心架构对比

两者在编译器流水线中扮演着不同层级的角色。简单而言，Pattern Rewriting 是一种通用的、基于规则的局部等价变换工具；而 Dialect Conversion 则是建立在重写机制之上，带有全局合法性约束和类型系统映射的系统化降级（Lowering）框架。

| 评估维度 | Pattern Rewriting (e.g., Greedy Rewrite) | Dialect Conversion |
|---------|-----------------------------------------|-------------------|
| 核心驱动逻辑 | 工作表驱动（Worklist-driven）：重复应用模式直至图收敛（稳态）。 | 目标驱动（Target-driven）：驱动所有非法的 Operation 转换为合法的 Operation。 |
| 合法化约束（Legality） | 无约束。纯粹的贪心或启发式局部替换。 | 强约束。依赖 ConversionTarget，支持 Legal / Illegal / Dynamically Legal。 |
| 类型演进（Type Conversion） | 通常保持类型不变（Type-preserving）。 | 核心能力之一。通过 TypeConverter 管理签名更新与类型的一致性替换。 |
| 失败语义（Failure Semantics） | 模式匹配失败仅代表该节点无法优化，不阻塞整体流程。 | 任意非法节点无法被成功转换时，导致整个 Conversion Pipeline 宣告失败。 |
| IR 变异与回滚机制 | 破坏性修改（Destructive mutation）。 | 事务化（Transactional）：自带撤销（Undo）机制，整体失败时可回滚 IR 状态。 |
| 核心抽象接口 | RewritePattern, PatternRewriter | ConversionPattern, ConversionPatternRewriter |

## 2. Pattern Rewriting：驱动启发式优化的基础机制

Pattern Rewriting 是 MLIR 中进行 IR 结构变异的基础设施。它主要用于执行无需改变方言（Dialect）整体层级的局部优化。

### 2.1 工作机制

1. 开发者通过定义继承自 `RewritePattern` 或其子类（如 `OpRewritePattern<T>`）的模式。
2. MLIR 的驱动器（通常是 `GreedyPatternRewriteDriver`）会维护一个操作（Operation）的工作表：
   - 从工作表中提取 Op。
   - 尝试应用所有注册的模式。
   - 若匹配成功（`matchAndRewrite` 返回 success），则使用 `PatternRewriter` 生成新节点并替换旧节点。
   - 将受影响的相邻节点重新加入工作表，直至无模式可匹配（收敛）。

### 2.2 典型应用场景

- **窥孔优化（Peephole Optimizations）**：例如代数化简（$x + 0 \rightarrow x$）。
- **规范化（Canonicalization）**：将 IR 转换为标准形式，以便后续 pass 分析。
- **局部算子融合（Operator Fusion）**：例如将 matmul 与紧随其后的 add 融合为 fused_matmul。

**C++**
```cpp
// 典型的 Pattern Rewriter 示例：消除加零操作 
struct SimplifyAddZero : public OpRewritePattern<AddOp> { 
  LogicalResult matchAndRewrite(AddOp op, 
                               PatternRewriter &rewriter) const override { 
    if (isZeroConstant(op.getRhs())) { 
      rewriter.replaceOp(op, op.getLhs()); 
      return success(); 
    } 
    return failure(); 
  } 
};
```

## 3. Dialect Conversion：基于合法性约束的系统化降级

Dialect Conversion 是一个用于跨方言迁移和系统级降级的复杂框架。它解决了编译器在不同抽象层级间转换时面临的全局一致性问题：如何确保所有节点和类型都被正确且完整地映射到新的方言体系中。

### 3.1 框架的核心要素

Dialect Conversion 摒弃了纯粹的贪心策略，引入了以下核心组件：

- **ConversionTarget（转换目标）**：定义图的最终合法状态。开发者需要显式声明哪些方言或具体的操作是“合法的（Legal）”，哪些是“非法的（Illegal）”。

- **TypeConverter（类型转换器）**：在跨层级降级中（例如从 Tensor 到 MemRef，或从 I64 到 I32），数据类型必须发生根本性改变。TypeConverter 负责维护类型之间的映射规则，并处理函数签名（Function Signature）和块参数（Block Arguments）的更新。

- **ConversionPattern 与 ConversionPatternRewriter**：
  - `ConversionPattern` 的重写函数会额外接收一个 `ValueRange operands`。由于类型转换是深度优先或系统性发生的，传入的 operands 代表已经被转换后的新操作数，这极大地简化了模式编写者的逻辑。
  - `ConversionPatternRewriter` 记录了所有 IR 变异的轨迹（类似事务日志）。如果最终 `ConversionTarget` 中的合法性要求未被完全满足，Rewriter 会触发回滚，将 IR 恢复到转换前的状态，防止生成“半成品”的错误 IR。

### 3.2 典型应用场景

- **层级降级（Dialect Lowering）**：例如将 tosa 或 stablehlo 降级到 linalg，或将 memref 降级到 llvm。
- **全局类型重塑**：将整个模块中基于按值传递的 Tensor 语义替换为基于指针的 MemRef（Bufferization 过程）。

**C++**
```cpp
// 典型的 Conversion 设定：定义目标与合法性 
ConversionTarget target(getContext()); 
target.addIllegalOp<tensor::ExtractOp>(); 
target.addLegalDialect<memref::MemRefDialect>(); 

// 并在执行时严格校验 
if (failed(applyPartialConversion(module, target, std::move(patterns)))) 
  signalPassFailure();
```

## 4. 架构关系：包含与扩展

从软件工程抽象的角度看，Conversion 是 Rewriter 的超集与系统化演进。

$$\text{Dialect Conversion} = \text{Pattern Rewriting} + \text{Target Legalization} + \text{Type Conversion} + \text{Transactional Rollback}$$

Conversion 底层依然使用 Pattern 匹配操作，但它剥夺了模式对 IR 最终形态的绝对决定权。模式的成功匹配只是一步提议（Proposal），整个流水线的成功与否由 `ConversionTarget` 在全局合法化校验阶段进行最终裁决。

## 5. 工程实践指南（何时使用？）

在开发 MLIR Pass 时，应严格遵循以下边界选择适当的工具：

### 选择 Pattern Rewriting 驱动 (如 `applyPatternsAndFoldGreedily`)：
- 任务的目的是优化现有代码，而非改变代码的语义抽象层级。
- 转换前后，数据的抽象类型（如 Tensor, MemRef）保持一致。
- 即使部分节点匹配失败，整体 IR 依然是可以被后端或下一阶段接受的合法状态。

### 选择 Dialect Conversion 驱动 (`applyPartialConversion` / `applyFullConversion`)：
- 任务的目的是降级（Lowering）或跨抽象域映射。
- 涉及到类型的全局迁移（例如：指针具象化、位宽转换）。
- 必须提供“全有或全无（All-or-Nothing）”的保证：任何遗漏的旧方言节点都会导致后续编译阶段崩溃，因此需要严格的合法化检查机制兜底。