---
title: "FlashAttention：从 IO Complexity 理解精确注意力"
description: "从 HBM 与 SRAM 的数据搬运出发，解释分块计算和 Online Softmax 为什么能够加速精确 Attention。"
category: "papers"
section: "论文研读"
tags: ["FlashAttention", "Attention", "GPU"]
difficulty: "进阶"
order: 2
updated: "2026-11-23"
created: "2022-05-01"
draft: false
featured: true
readingTime: "约 13 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 研究问题：注意力的瓶颈不只是 FLOPs

标准 Attention 计算包含 QKᵀ、Softmax 和乘 V。对于长度 N 的序列，直接实现可能显式生成 N×N 的注意力矩阵。它不仅占用大量显存，还需要反复将中间张量写入和读出 GPU 的高带宽内存（HBM）。

FlashAttention 的出发点是 IO-aware algorithm：计算的实际代价受访存影响，不能只按乘加次数判断。通过将计算组织为适合片上 SRAM 的块，能够避免将完整的注意力分数矩阵写回 HBM。

## 从公式看哪些中间量需要保存

精确缩放点积注意力为：

$$
O=\operatorname{softmax}(QK^\top/\sqrt{d_k})V.
$$

普通写法：O = softmax(Q @ K.T / sqrt(d_k)) @ V。

直接实现习惯上先构造 S=QKᵀ，得到 P=softmax(S)，再计算 PV。对于长序列，S 和 P 的规模随 N² 增长。即便最终输出 O 只有 N×d 维，中间量仍然昂贵。

关键观察：Softmax 可以按块维护精确的归一化统计量，而不要求一次持有整行所有分数。

## Online Softmax 的核心递推

对已处理的一组分数，维护行最大值 m 与指数和 l。新块的最大值为 m_b，指数和为 l_b，合并时：

$$
m'=\max(m,m_b),\quad l'=e^{m-m'}l+e^{m_b-m'}l_b.
$$

普通写法：new_max = max(old_max, block_max)；new_normalizer = exp(old_max-new_max)*old_sum + exp(block_max-new_max)*block_sum。

若同时保存已累积的分子向量，还可以按相同的重标定机制更新输出。最终得到与全量 Softmax 相同的数学结果，浮点舍入路径可能不同。

## 分块（Tiling）与重计算

FlashAttention 将 Q、K、V 切分为小块。一个块进入 SRAM 后，尽可能完成打分、归一化统计更新与输出累积，然后再读取下一块。这减少中间矩阵在 HBM 的落盘次数。

反向传播通常不保存完整注意力矩阵，而是保存更紧凑的统计量，在反向过程中重计算部分中间值。计算量与内存访问之间由此出现一种有利的交换。

这也解释了一个常见误解：FlashAttention 不是将精确 Attention 改写成线性注意力近似。它优化的是计算顺序和访存模式，核心算子仍包含成对打分。

## 理论复杂度与工程性能如何对应

标准密集注意力的乘法计算在常见设置下仍随 N² 增长；FlashAttention 重点降低额外显存与 HBM 访问。实际速度取决于序列长度、头维度、硬件 SRAM 大小、矩阵指令和内核实现。

因此比较时应控制硬件、精度、形状、mask 类型和反向传播配置。单一数据点的加速比不能当作不同 GPU 和上下文长度上的普遍结论。

## 与 KV Cache、PagedAttention 的关系

FlashAttention 优化 Attention 内核本身；KV Cache 避免 Decode 中重复计算历史 K/V；PagedAttention 负责推理服务的 KV 内存管理。它们可能共同出现在同一个推理系统中，但处于不同层次。

例如长上下文 Prefill 可以从分块注意力计算获益；高并发 Decode 还会受到 KV 访存与调度策略限制。理解瓶颈后才能决定优先优化哪一层。

## 推荐的复现实验

准备多组 N、d、精度与 mask 类型，比较原生 SDPA 与目标内核的峰值显存、前向延迟、反向延迟和数值误差。每组先完成预热，再重复测量；对异步 GPU 运算应使用正确的同步或 Event 计时。

结果应附硬件型号、CUDA 版本、内核版本、batch、head 维度与测试脚本，否则很难解释差异。

## 原始论文与实现

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Dao et al., 2022。
- [FlashAttention 官方代码](https://github.com/Dao-AILab/flash-attention)。
- [KV Cache 技术解析](/docs/transformer/kv-cache)。
