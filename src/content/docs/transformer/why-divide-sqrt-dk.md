---
title: "为什么 Attention 分数要除以 √dₖ"
description: "从点积方差、Softmax 饱和与梯度行为出发，推导 Attention 的缩放因子。"
category: "transformer"
section: "Transformer 专题"
tags: ["Attention","Transformer","数学直觉"]
difficulty: "入门"
order: 3
updated: "2026-11-23"
created: "2026-05-29"
draft: false
featured: false
cover: ""
readingTime: "约 10 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 缩放因子解决的是分数尺度问题

Attention 使用 Query 与 Key 的点积作为匹配分数。若向量维度增长，而各分量的典型大小保持相近，点积的数值尺度可能随维度增大。Softmax 将这些分数转成权重时，过大的差异容易形成极端分布。

Scaled Dot-Product Attention 在点积之后除以 √d_k，让进入 Softmax 的数值尺度更稳定。这是一个可从统计近似理解的设计，而不是必须背诵的经验常数。

## 从随机变量推导点积方差

假设 Query 与 Key 的每个分量都独立、均值为零、方差为一，并且两组分量彼此独立。点积分数为：

$$
S=q^\top k=\sum_{i=1}^{d_k}q_i k_i.
$$

普通写法：score = sum_i query_i * key_i。

由于每个乘积项的均值为零、方差为一，独立求和得到：

$$
\mathbb E[S]=0,\qquad \operatorname{Var}(S)=d_k.
$$

普通写法：E[score] = 0；Var(score) = key_dimension。

因此分数的标准差随 √d_k 增长。归一化后：

$$
\operatorname{Var}\left(\frac{S}{\sqrt{d_k}}\right)=1.
$$

普通写法：Var(score / sqrt(key_dimension)) = 1。

这些结论依赖上面的独立与方差假设。实际训练后的向量不一定满足该分布，但推导解释了缩放项的统计动机。

## Softmax 为什么对分数尺度敏感

Softmax 的第 i 项为：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

普通写法：p_i = exp(logit_i) / sum_j exp(logit_j)。

如果把所有分数乘以很大的正数，最大分数对应的概率接近 1，其余概率接近 0。分布趋向高度尖锐；在该区域，许多分量的梯度变得很小。

Softmax 对某个输入分量的导数为：

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).
$$

普通写法：d p_i / d z_j = p_i * (indicator(i=j) - p_j)。

当 p_i 接近 0 或 1 时，许多导数项减小，使训练难以充分利用不同 Key 的信号。

## 一个简单的尺度对比

考虑分数向量 (1,2,3)。Softmax 输出约为 (0.090,0.245,0.665)；如果改为 (4,8,12)，权重将更接近 (0,0.018,0.982)。

虽然两个输入的排序完全一样，Softmax 输出的分布却明显不同。这说明模型关注的不只是“哪个分数更大”，还包括分数之间的绝对尺度。

## 为什么不是除以 dₖ

在上述统计近似中，点积方差与 d_k 成正比，因此把标准差归一化需要 √d_k。若直接除以 d_k，随着维度增加，典型分数幅度又会变得过小，使权重分布更接近均匀。

实际现代模型还受到归一化、注意力温度、参数初始化和位置编码等因素影响。缩放机制与这些设计协同工作，但其核心目的仍是控制打分的数值尺度。

## 推导和工程测试如何衔接

可以在固定数据和初始化条件下，比较原始点积、除以 √d_k 与其他温度配置，并统计注意力分数方差、权重熵和梯度范数。

如果要讨论训练稳定性，还应报告学习率、头维度、归一化策略和模型容量，避免把所有差异归因于单个缩放常数。

## 延伸阅读

- [四个 Token 推导 Self-Attention](/docs/transformer/attention-four-tokens)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
