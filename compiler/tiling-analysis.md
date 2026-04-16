# Tiling 大小设计：GPU vs NPU 深度解析

## 核心目标
Tiling（分块）的核心目的是打破**访存墙（Memory Wall）**，增加数据的局部性（Data Reuse）。由于 GPU 和 NPU 的底层架构截然不同，它们对 Tiling 大小的诉求和约束有本质的区别。

## 1. GPU 的 Tiling 策略：基于层次化缓存与 SIMT 架构

### 核心特征
- 高并发、深层次的 Cache（Global -> L2 -> L1/Shared Memory -> Register）
- 隐式的硬件调度

### 优化目标
- 掩盖延迟（Hide Latency）
- 最大化 Occupancy（占用率）
- 喂饱 Tensor Cores

### Tiling 大小约束
1. **Shared Memory 限制**：Block 级的 Tiling 必须能完全塞进对应 SM（Streaming Multiprocessor）的 Shared Memory 中，并且要留有余地以支持足够多的 Thread Blocks 驻留，保证高 Occupancy。

2. **对齐与指令限制**：内部 Tiling 大小必须是 Warp size (32) 的倍数，以避免线程分歧（Thread Divergence）和非合并访存（Uncoalesced Memory Access）。

3. **MMA 指令强绑定**：如果要使用 Tensor Cores（如 mma.sync 指令），最内层的 Tiling 大小被硬件指令写死（例如对于 FP16 的 Matmul，形如 $16 \times 16 \times 16$ 或 $32 \times 8 \times 16$）。

### 特点
- GPU 的 Tiling 通常是 2 级到 3 级的（Block Tile -> Warp Tile -> Thread Tile）
- 相对灵活，就算大小没卡准，硬件的 L1/L2 Cache 也能兜底一部分性能
- 优化空间通常是一个连续的凸函数

## 2. NPU / DSA 的 Tiling 策略：基于空间架构与软件管理显存

### 核心特征
- 空间计算架构（Spatial Architecture, 如 2D/3D PE 阵列）
- 软件管理的便笺内存（Scratchpad Memory / SRAM）
- 显式的 DMA 数据搬移

### 优化目标
- 最大化 PE 阵列的时空利用率（Spatial Unrolling）
- 建立高效的向量-数据流执行（Vector-Dataflow Execution）
- 通过双缓冲完全隐藏 DMA 延迟

### Tiling 大小约束
1. **SRAM 的硬边界（Hard Capacity Constraints）**：NPU 通常没有硬件自动管理的 Cache。Tiling 大小加上双缓冲（Double Buffering）的需求，绝对不能超过片上 SRAM 的物理上限。一旦超过，程序直接跑飞或编译失败。

2. **PE 阵列维度的刚性匹配**：如果 NPU 的脉动阵列（Systolic Array）或 CGRA 拓扑是 $64 \times 64$ 的，那么最内层的空间 Tiling 维度最好是 64 的整数倍。如果不满，必须进行显式的 Padding，否则会导致边缘 PE 闲置，大幅拉低计算力利用率。

3. **Dataflow 数据流模式的绑定**：Tiling 大小的形状取决于 NPU 采用的数据流策略（如 Weight Stationary 或 Output Stationary）。例如在 Weight Stationary 下，Tiling 会优先保证切出来的权值块能一直驻留在 PE 内部的寄存器中，而让 Activation 不断流过。

### 特点
- NPU 的 Tiling 容错率极低
- 由于缺乏硬件 Cache 兜底，一个不合理的 Tiling shape 会导致频繁的 DRAM thrashing（抖动），性能可能断崖式下跌 10 倍以上
- 必须在编译期精确匹配 PE 拓扑和 SRAM 容量

## 场景分析

### GPU 场景：基于缓存与并发调度的隐式博弈
- **Tiling 过大**：导致 Register Spilling（寄存器溢出）和 Occupancy 急剧下降
- **Tiling 过小**：导致指令开销主导、Tensor Core 饥饿与未对齐、缓存线浪费

### NPU / CGRA 场景：确定性数据流与空间架构的刚性约束
- **Tiling 过大**：导致 Hard Failure（编译失败或运行时崩溃）和流水线气泡（Pipeline Stalls）
- **Tiling 过小**：导致空间展开失败（Spatial Underutilization）和 DMA 启动开销主导（Thrashing）

## 总结
在 MLIR 中处理这两个后端的优化时，心态是完全不同的：
- **对于 GPU**：Tiling 是一个在 L1/Shared Memory/Registers 之间找平衡的试探过程，容错率较高
- **对于 NPU**：Tiling 是一个具有严格边界条件的约束求解过程，任何对物理架构的偏离都会产生巨大的性能抖动