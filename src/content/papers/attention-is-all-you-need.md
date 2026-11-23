---
title: "Attention Is All You Need：重建 Transformer 的计算逻辑"
description: "从 RNN 的序列依赖约束出发，解析编码器、解码器、多头注意力与位置表示，并对照现代 Decoder-only 模型。"
category: "papers"
section: "论文研读"
tags: ["Transformer", "Attention", "论文研读"]
difficulty: "进阶"
order: 1
updated: "2026-11-23"
created: "2017-06-01"
draft: false
featured: true
readingTime: "约 14 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 研究问题：序列建模为何需要摆脱递归

Transformer 的贡献不能仅概括为“引入 Attention”。在机器翻译的编码器—解码器框架中，循环神经网络按时间步处理序列，使训练的时间维度难以完全并行。长距离依赖还会增加信息传递路径。

Vaswani 等人的问题是：能否主要通过注意力和位置表示处理整个序列，减少对递归结构的依赖，同时保持翻译建模能力？原论文给出的方案是完整的 Encoder–Decoder Transformer。

## 架构：六个关键组成部分

Encoder 每层包含多头自注意力与逐位置前馈网络；Decoder 还包含遮蔽自注意力和跨注意力。残差连接与归一化用于维持深层网络的训练稳定性，位置编码提供 Token 顺序信息。

需要区分三个不同的操作：Encoder Self-Attention 读取源序列内部；Decoder Masked Self-Attention 只能使用当前及之前目标 Token；Cross-Attention 由目标状态读取源序列表示。它们共享注意力计算形式，但 Q/K/V 来自不同的数据流。

## 缩放点积注意力

给定 Q、K、V：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

普通写法：Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_k)) @ V。

若序列长度为 N，QKᵀ 对应 N×N 的分数矩阵。每一行按键位置进行 Softmax，并对 Value 作加权求和。此处的分数是根据输入内容动态计算的，而非固定参数矩阵。

## 多头注意力为什么有意义

单头注意力在一个投影空间中构造权重。多头注意力将表示投影到多个子空间，分别聚合后再连接，提供不同的交互模式。

$$
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(head_1,\ldots,head_h)W^O.
$$

普通写法：MultiHead = concat(attention_head_1, ... attention_head_h) @ W_out。

增加头数也带来投影和实现复杂度；真实模型中头数、头维度和 KV 共享策略需要一并考虑。

## 位置编码的角色

没有位置输入的标准自注意力对于输入 Token 的排列具有特定的等变性，不能凭空知道原始顺序。原论文通过固定正弦余弦位置编码把绝对位置加入表示。

之后的许多架构采用 RoPE、ALiBi 等不同位置机制。原始 Transformer 的位置编码方式并非所有现代大模型的共同配置。

## 为什么能够并行，又为什么有平方成本

自注意力允许一个层内各 Token 的交互批量计算，不必像 RNN 那样严格串行遍历历史状态。对完整序列进行密集 QKᵀ 计算，注意力矩阵规模随 N² 增长。

因此 Transformer 的优势与瓶颈同时来自 Token 间的全局交互。之后 FlashAttention、稀疏注意力、长上下文位置扩展和 KV Cache 分别从访存、连接模式、位置建模和推理复用入手优化。

## 从原始论文到 Decoder-only 模型

现代自回归语言模型通常只保留遮蔽自注意力堆叠，不再要求一套独立的 Encoder 表示。它们沿着下一个 Token 预测目标训练，而不是原论文的机器翻译 Seq2Seq 目标。

理解这一区别有助于阅读原始论文时避免把 Decoder-only LLM 的全部设计回投到 2017 年的架构。

## 如何读原论文的实验

首先看翻译任务的训练成本与质量指标，判断论文是否支撑“移除递归仍能有效建模”的主张。其次看层数、隐藏维度、头数等关键超参数，解释模型容量和计算代价。

复现时应区分架构本身与训练配方：学习率 warmup、label smoothing、数据处理及解码策略都可能显著影响结果。单独实现注意力公式并不能复现完整原始论文结果。

## 把一个 Transformer Block 拆成可验证的子模块

假设输入序列长度为 L，隐藏维度为 D。Attention 子层先投影 Q/K/V，再将多个 Head 的输出连接并映射回 D 维。前馈子层则对每个位置独立应用两次线性变换和非线性激活。

典型的 FFN 可以抽象为：

$$
\operatorname{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2.
$$

普通写法：ffn(x) = linear_2(activation(linear_1(x)))。

注意力在 Token 之间混合信息，FFN 主要在每个位置的特征维度上提供非线性变换。两者承担不同职责，因此仅实现 QKᵀ 和 Softmax 并不等于实现完整 Transformer。

原论文采用 Post-Norm 风格的 Add & Norm，而许多现代 Decoder-only 架构使用 Pre-Norm、RMSNorm 或门控 FFN。阅读技术演进时应明确对应的归一化顺序，而不把所有变种混为同一个 Block。

## 因果掩码为什么影响模型训练目标

Decoder 的 Masked Self-Attention 只允许位置 t 使用不晚于 t 的上下文。如果去掉 Mask 并仍执行下一个 Token 预测，模型可能在训练时直接看到目标位置之后的信息，引入明显的信息泄漏。

从矩阵角度看，长度为四的序列对应一个下三角可见性矩阵：

~~~text
       K0 K1 K2 K3
Q0      1  0  0  0
Q1      1  1  0  0
Q2      1  1  1  0
Q3      1  1  1  1
~~~

这张矩阵不仅影响生成方式，也是训练目标能否成立的前提。与之不同，Encoder 的自注意力在常见设置下允许双向读取。Encoder–Decoder Cross-Attention 再让目标侧表示读取源侧编码结果。


## 继续研究

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017。
- [四个 Token 推导 Attention](/docs/transformer/attention-four-tokens)。
- [FlashAttention 论文研读](/papers/flashattention)。
- [KV Cache](/docs/transformer/kv-cache)。
