---
title: "为什么要除以 sqrt(d_k)"
description: "解释 scaled dot-product attention 中缩放项的数学直觉和工程意义。"
category: "transformer"
section: "Transformer 专题"
tags: ["Attention","Transformer","数学直觉"]
difficulty: "入门"
order: 3
updated: "2026-07-16"
created: "2026-05-29"
draft: false
featured: false
cover: ""
readingTime: "约 6 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 本文解决什么问题

Scaled dot-product attention 里有一个看起来很小的细节：`QK^T` 后面要除以 `sqrt(d_k)`。很多人记住了公式，却说不清为什么要缩放。本文把这个问题拆开：点积为什么会变大，变大后 softmax 会发生什么，缩放为什么能让训练更稳定。

## 核心结论

除以 `sqrt(d_k)` 是为了控制点积分数的尺度。如果不缩放，随着 key/query 维度变大，点积的方差也会变大，softmax 容易变得过于尖锐，梯度变小，训练不稳定。

它不是为了改变 Attention 的语义，而是为了让不同维度下的分数落在更合适的数值范围里。

## 概念解释

假设 query 和 key 的每个分量都近似均值为 0、方差为 1。两个向量做点积时，会把 `d_k` 个乘积加起来。加和项越多，结果的方差越大，大致会随 `d_k` 增长。

当这些分数进入 softmax，如果某个分数明显大于其他分数，softmax 输出会接近 one-hot。模型早早把注意力集中到少数位置，反向传播时很多位置的梯度就会很弱，训练容易变慢或不稳定。

## 数学/工程直觉

缩放项的直觉可以写成：

```txt
原始点积方差约为 d_k
除以 sqrt(d_k) 后，方差重新回到约 1
```

这和神经网络初始化里的尺度控制是同一类思想：不是让模型少学东西，而是避免数值范围在深层网络里失控。Transformer 层数越深、head 越多，这种稳定性越重要。

## 一个小例子

```txt
d_k = 64
sqrt(d_k) = 8

如果某一行原始 attention scores 是：
[12, 4, 2, -3]

缩放后变成：
[1.5, 0.5, 0.25, -0.375]
```

第一组分数进 softmax 后会非常尖锐；第二组仍然保留偏好，但不会过早把其他位置压到几乎没有梯度。

## 常见误区

第一，以为缩放是经验技巧。它确实来自工程实践，但背后有清晰的方差控制直觉。

第二，以为除以 `d_k` 也一样。除以 `d_k` 会缩得过狠，常见推导对应的是除以标准差 `sqrt(d_k)`。

第三，只从“防止数值过大”解释，不提 softmax 和梯度。面试里最好把“点积方差变大 -> softmax 饱和 -> 梯度变弱”这条链路说完整。

## 面试追问

1. `d_k` 指的是什么维度？
2. 为什么点积的方差会随维度增长？
3. softmax 饱和会对梯度造成什么影响？
4. 为什么是除以 `sqrt(d_k)` 而不是 `d_k`？
5. 这个缩放项和初始化、归一化有什么相似之处？

## 推荐阅读

建议和 Attention 计算流程一起读。先掌握 Q、K、V 的形状，再理解缩放项，会比单独背公式更稳。
