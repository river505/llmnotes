---
title: "用 4 个 Token 看懂 Attention"
description: "用 4 个 token 的小例子拆解 Attention 的 Q、K、V、打分和加权求和。"
category: "transformer"
section: "Transformer 专题"
tags: ["Transformer","Attention","QKV"]
difficulty: "入门"
order: 1
updated: "2026-07-16"
created: "2026-05-29"
draft: false
featured: true
cover: ""
readingTime: "约 7 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 本文解决什么问题

Attention 这个词很容易被讲玄：什么“关注重要信息”“动态加权上下文”。这些说法不算错，但不够具体。本文只用 4 个 token 走一遍计算过程，目标是让你知道 Q、K、V 分别在干什么，注意力分数从哪里来，最后为什么会得到新的表示。

## 核心结论

Self-Attention 做的事情是：让序列里的每个 token 根据自己当前的需求，从同一序列的其他 token 中取信息。它不是固定窗口，也不是简单平均，而是为每个 token 计算一组权重，再对 value 做加权求和。

一句话记法：Q 表示“我想找什么”，K 表示“我有什么特征可以被别人匹配”，V 表示“如果别人关注我，我能贡献什么信息”。

## 概念解释

假设一句话被切成 4 个 token：

```txt
[我, 喜欢, 机器, 学习]
```

每个 token 先通过 embedding 变成向量。随后，模型用三组可学习矩阵把每个向量投影成 Q、K、V。对“学习”这个 token 来说，它的 Q 会分别和 4 个 token 的 K 做点积，得到 4 个相关性分数。分数经过缩放和 softmax 后变成权重，再用这些权重加权 4 个 V，得到“学习”这个位置的新向量。

## 数学/工程直觉

矩阵形式通常写成：

```txt
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

这里的 `QK^T` 会得到一个 4 x 4 的分数矩阵。第 i 行表示第 i 个 token 看其他 token 的分数；第 j 列表示其他 token 对第 j 个 token 的关注程度。softmax 让每一行变成概率分布，最后乘以 V 就是在“取信息”。

工程上，Attention 的好处是并行计算方便，长距离依赖也能直接建模；代价是标准 self-attention 的计算和显存通常随序列长度平方增长。长上下文模型会围绕这个瓶颈做很多优化。

## 伪代码

```txt
输入：4 个 token 的 embedding，形状为 [4, d_model]
1. 线性投影得到 Q、K、V
2. 计算 scores = Q @ K.T
3. scores 除以 sqrt(d_k)
4. 对每一行做 softmax，得到 attention weights
5. output = weights @ V
输出：4 个带上下文信息的新向量
```

这一步只是一层 Attention。真实 Transformer 会叠很多层，并使用多头注意力，让不同 head 学不同类型的关系。

## 常见误区

第一，把注意力权重直接当解释。权重能提供线索，但不等于严格因果解释，尤其在多层多头结构里更要谨慎。

第二，以为 Attention 会自动理解位置。原始 self-attention 本身对顺序不敏感，需要位置编码或相对位置机制提供顺序信息。

第三，只记公式，不看形状。面试里把 `[seq_len, d_model]`、`[seq_len, seq_len]`、`[seq_len, d_v]` 这些形状说清楚，往往比背一句定义更有说服力。

## 面试追问

1. Q、K、V 的直观含义分别是什么？
2. 为什么 `QK^T` 会得到 token 两两之间的关系？
3. self-attention 和 cross-attention 有什么区别？
4. Attention 的计算复杂度为什么和序列长度平方相关？
5. 多头注意力相比单头注意力增加了什么表达能力？

## 推荐阅读

下一篇可以看 `为什么要除以 sqrt(d_k)`。它解释的是公式里最容易被忽略、但对训练稳定性很关键的一项。
