---
title: "从四个 Token 推导 Self-Attention"
description: "完整计算 Q/K/V、缩放打分、Softmax 和 Value 聚合，并讨论因果掩码与张量形状。"
category: "transformer"
section: "Transformer 专题"
tags: ["Transformer","Attention","QKV"]
difficulty: "入门"
order: 1
updated: "2026-11-23"
created: "2026-05-29"
draft: false
featured: true
cover: ""
readingTime: "约 13 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 从四个 Token 的数据流开始

Attention 并不是一个“关注重要词语”的固定规则。给定一组输入表示，它先用可学习投影得到 Query、Key 和 Value，再根据输入内容动态计算位置间的联系。每一个新输入序列都可能产生不同的注意力矩阵。

为了理解完整的数据流，可以暂时忽略 Batch 维度，假设序列长度 N=4，模型输入维度 d_model=2，单个头的 Key 维度 d_k=2。这虽然比真实模型简单得多，但不会改变计算逻辑。

## 输入到 Q、K、V：三个投影各司其职

记输入矩阵 X∈ℝ^(4×2)。常见的 Self-Attention 使用独立参数矩阵把 X 投影为：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
$$

普通写法：Q = X @ W_Q；K = X @ W_K；V = X @ W_V。

三者由同一序列生成，但参数不要求相同。Query 负责构造当前位置的查询向量，Key 负责供其他 Query 匹配，Value 则承载真正被聚合的信息。这种解释只是帮助理解计算角色，不能把 Q/K/V 当成固定语义标签。

为了方便手算，假设投影后四行 Query 和 Key 都是 (1,0)、(0,1)、(1,1)、(1,-1)，而 Value 分别取 (1,0)、(0,1)、(1,1)、(2,0)。这些具体数值是教学示例，不是训练好的模型参数。

## 第一轮矩阵乘法：得到谁与谁的匹配分数

每个 Query 与所有 Key 做内积，产生 4×4 分数矩阵：

$$
S=QK^\top=
\begin{bmatrix}
1&0&1&1\\
0&1&1&-1\\
1&1&2&0\\
1&-1&0&2
\end{bmatrix}.
$$

普通写法：S[i,j] = dot(Q[i], K[j])；S shape = [4,4]。

第 i 行对应第 i 个位置对所有 Key 的分数。注意这里还没有进行归一化，分数也不代表概率。它们是通过当前输入与投影矩阵确定的兼容性度量。

## 缩放与 Softmax：将分数转成归一化权重

Scaled Dot-Product Attention 先除以 √d_k，再在 Key 维度应用 Softmax：

$$
A_{ij}=
\frac{\exp(S_{ij}/\sqrt{d_k})}
{\sum_{m=1}^{4}\exp(S_{im}/\sqrt{d_k})}.
$$

普通写法：A[i,j] = exp(S[i,j]/sqrt(dk)) / sum_m exp(S[i,m]/sqrt(dk))。

第三个 Query 的分数行是 (1,1,2,0)。由于 d_k=2，经过缩放后约为 (0.707,0.707,1.414,0)，其 Softmax 权重约为 (0.221,0.221,0.449,0.109)。

这些权重是动态产生的，四项非负且合计为 1。它们表示第三个位置读取四个 Value 的相对比例，但不等同于可解释的语义因果贡献。

## 第二轮矩阵乘法：真正的信息聚合

接下来将权重矩阵乘以 V：

$$
O=AV=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

普通写法：Output = softmax(Q @ K.T / sqrt(dk)) @ V。

对于第三个位置，按照上面的教学数值，输出约为 (0.888,0.670)。第一维由第 1、3、4 行 Value 的加权和得到；第二维由第 2、3 行 Value 的加权和得到。

整个输出 O 的维度为 4×2。它不是在四个输入位置中选一个，而是为每个位置生成新的上下文表示。这种加权聚合是 Self-Attention 的核心计算。

## 因果掩码如何改变这个例子

在 Decoder-only 自回归语言模型中，第 i 个位置不能读取未来 Token。掩码在 Softmax 之前将禁止位置的分数设为负无穷：

$$
A=\operatorname{softmax}\left(S/\sqrt{d_k}+M\right).
$$

普通写法：A = softmax(scaled_scores + causal_mask)，future entries of mask are -infinity。

第三个位置因此无法读取第四个 Value。它只在前三个位置上归一化，对应教学权重约为 (0.248,0.248,0.504,0)，输出也随之变化。

这说明 Mask 不是在最终输出后删除未来位置，而是在构造权重分布时约束信息可访问性。

## 多头注意力与完整 Transformer 之间的关系

真实 Transformer 往往使用 h 个头，每个头拥有自己的投影。各头产生输出之后沿特征维连接，再经过输出投影。并行的头提供不同投影子空间，允许模型学习多种交互模式。

除了 Attention，还需要残差连接、归一化、前馈网络、位置表示，以及训练目标。只有把这些模块按计算图连接起来，才构成完整的 Transformer Block。

## 计算复杂度与长上下文问题

在密集 Self-Attention 中，QKᵀ 包含 N² 对位置交互。当头维度 d_k 固定时，分数矩阵的计算与存储规模通常随 N² 增长。这解释了为什么长上下文推理会关注内存访问、稀疏连接与 KV Cache。

对模型性能优化，应区分三个问题：Attention 内核怎样减少数据搬运、Decode 怎样复用历史 K/V、推理服务怎样调度多个会话。它们对应不同的瓶颈。

## 进一步阅读

- [为什么要除以 √dₖ](/docs/transformer/why-divide-sqrt-dk)
- [Attention Is All You Need 论文研读](/papers/attention-is-all-you-need)
- [FlashAttention 论文研读](/papers/flashattention)
- [KV Cache 的显存与延迟](/docs/transformer/kv-cache)
