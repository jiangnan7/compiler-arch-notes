# 大型语言模型（LLM）中 Batch Size 与 Sequence Length 的解析

在深度学习与大型语言模型（LLM）的架构中，Batch Size（批大小） 与 Sequence Length（序列长度） 是决定计算效率、内存占用及模型收敛性能的核心超参数。下文将从学术与工程实践的角度，对这两个概念进行深度拆解。

## 1. Batch Size (批大小)

### 1.1 定义

Batch Size 指在神经网络的一次前向传播（Forward Pass）和反向传播（Backward Pass）中，同时处理的独立训练样本（句子或文本段落）的数量。

### 1.2 核心影响维度

- **梯度估计稳定性**：根据大数定律，Batch Size 越大，计算出的梯度越接近全量数据的真实梯度方向，梯度更新更加平滑。
  $$g_t = \frac{1}{B} \sum_{i=1}^{B} \nabla_{\theta} L(x_i, y_i; \theta_t)$$
  其中 $B$ 为 Batch Size。较大的 $B$ 可减少随机性，但过大的 $B$ 可能使模型陷入极小值点，降低泛化能力。

- **计算吞吐量（Throughput）**：现代硬件（如 NVIDIA H100 GPU）擅长并行计算。增加 Batch Size 能充分利用算力核心（Tensor Cores），提高每秒处理的 Token 数量。

- **显存占用**：显存占用与 Batch Size 成线性比例关系。

## 2. Sequence Length (序列长度 / SeqLen)

### 2.1 定义

Sequence Length 指输入模型中的单个样本所包含的 Token（标记） 数量。它定义了模型在处理当前 Token 时，能够回溯和参考的上下文窗口大小。

### 2.2 核心影响维度

- **注意力机制的计算开销**：在标准的 Transformer 架构中，自注意力（Self-Attention）机制的时间复杂度和空间复杂度均为 $O(n^2)$，其中 $n$ 是序列长度。

- **上下文感知能力**：更长的 SeqLen 意味着模型具备"长程记忆"，能够理解跨度巨大的逻辑关系（如法律文档、长篇小说）。

- **KV Cache（推理阶段）**：在推理（生成）过程中，SeqLen 直接决定了 KV Cache 的大小，这是导致大模型推理显存溢出（OOM）的主要原因。

## 3. Batch Size 与 Sequence Length 的交互关系

在底层实现中，输入数据通常被组织为一个形状为 (Batch Size, Sequence Length) 的二维张量（或加入 Embedding 维度后的三维张量）。

### 3.1 总 Token 数计算

单次迭代处理的总 Token 数由下式决定：
$$\text{Total Tokens} = \text{Batch Size} \times \text{Sequence Length}$$

### 3.2 显存权衡策略

由于硬件显存（VRAM）是固定的，工程师必须在两者之间进行平衡：

| 场景 | 调整策略 | 目的 |
|------|---------|------|
| 预训练 (Pre-training) | 增大 Batch Size | 追求极高的计算吞吐量和训练稳定性。 |
| 长文本微调 (SFT) | 增大 SeqLen，减小 Batch Size | 牺牲并行性以换取模型对长文档的理解。 |
| 资源受限环境 | 使用梯度累积 (Gradient Accumulation) | 模拟大 Batch Size 而不增加单次计算的显存负担。 |

## 4. 总结

- **Batch Size** 侧重于**"横向并行"**：决定了模型一次能"看"多少个例子，主要影响训练速度和梯度准确性。

- **Sequence Length** 侧重于**"纵向关联"**：决定了模型一次能"读"多长的内容，主要影响模型对上下文细节的掌握能力。

在当代 LLM 演进中，长序列长度（Long Context） 已成为核心竞争点（如 128k、1M tokens），而 Batch Size 的选择则更多地取决于分布式训练集群的规模与互联带宽。