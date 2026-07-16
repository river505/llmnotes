---
title: "Attention Is All You Need 精读"
description: "用组会可复述的方式拆解 Transformer 原始论文。"
category: "papers"
section: "论文精读"
tags: ["Transformer","论文精读","Attention"]
difficulty: "入门"
order: 1
updated: "2026-07-16"
created: "2026-05-29"
draft: false
featured: true
cover: ""
readingTime: "约 8 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 本文解决什么问题

《Attention Is All You Need》是 Transformer 的原始论文，也是大模型学习绕不开的一篇。精读这篇论文，不是为了记住每个实验数字，而是理解它为什么放弃循环结构，如何用 attention 建模序列，以及这个设计为什么后来能扩展到大语言模型。

## 论文要解决的问题

在 Transformer 之前，机器翻译常用 RNN、LSTM、GRU 或 CNN 结构。RNN 按时间步顺序处理序列，长距离依赖难学，并行效率也受限制。论文的核心问题是：能不能不用循环和卷积，只靠注意力机制完成序列到序列建模？

答案就是 Transformer：编码器读取源序列，解码器生成目标序列，中间主要依赖 self-attention、cross-attention、前馈网络、残差连接和层归一化。

## 模型结构

Transformer 由 encoder 和 decoder 组成。Encoder 每层包含 multi-head self-attention 和 feed-forward network；Decoder 每层多了 masked self-attention 和 encoder-decoder attention。

关键模块可以这样理解：

- Self-Attention：同一序列内部 token 之间互相取信息。
- Masked Self-Attention：生成时不能看到未来 token。
- Cross-Attention：解码器根据当前生成状态去读取编码器输出。
- Feed-Forward Network：对每个位置独立做非线性变换。
- Positional Encoding：补充序列顺序信息。

## 核心公式

论文中的 scaled dot-product attention 写作：

```txt
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

`QK^T` 用来计算匹配分数，`sqrt(d_k)` 控制数值尺度，softmax 把分数变成权重，最后乘以 `V` 得到加权后的上下文表示。Multi-head attention 则是并行做多组 attention，让不同 head 捕捉不同关系。

## 为什么重要

Transformer 的重要性不只是效果好，更在于它改写了序列建模的工程路径。Self-attention 可以并行计算，训练效率高；堆叠层数和扩大宽度相对直接；同一套结构可以迁移到语言、视觉、语音和多模态任务。

后来 BERT、GPT、T5 以及许多大模型都可以看作在 Transformer 基础上的不同训练目标、数据规模和系统设计选择。

## 读论文时要抓的点

第一，关注作者为什么不用 RNN。论文不是为了炫技，而是为了解决并行效率和长距离依赖问题。

第二，关注 attention 的形状和复杂度。它带来长距离建模能力，也带来序列长度平方级别的计算成本。

第三，关注 encoder-decoder 和 decoder-only 的区别。原论文是机器翻译架构，而很多 LLM 使用的是 decoder-only 变体。

## 常见误区

第一，把 Transformer 等同于 GPT。GPT 是 Transformer 的一种 decoder-only 使用方式，不是全部。

第二，把 attention 权重当成完整解释。权重有参考价值，但不能直接等同于模型推理原因。

第三，只背模型图，不理解数据流。精读时最好能从输入 token 一直讲到输出 logits。

## 组会追问

1. Transformer 为什么能比 RNN 更好并行？
2. Encoder self-attention 和 decoder masked self-attention 有什么区别？
3. 为什么需要 positional encoding？
4. Multi-head attention 增加了什么表达能力？
5. 原论文结构和现代 decoder-only LLM 有什么不同？

## 推荐阅读

读这篇论文前，建议先看本站 Attention 4 token 示例和 `sqrt(d_k)` 解释。先把公式背后的计算过程看明白，再回到论文会轻松很多。
