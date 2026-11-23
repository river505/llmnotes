---
title: "KV Cache：自回归推理中的内存与延迟"
description: "从逐 Token 解码、缓存张量形状和内存访问出发，理解 KV Cache 的作用、显存代价与优化方向。"
category: "transformer"
section: "Transformer 与注意力"
tags: ["KV Cache", "Transformer", "推理优化"]
difficulty: "进阶"
order: 3
updated: "2026-11-23"
created: "2026-11-23"
draft: false
featured: true
readingTime: "约 12 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 为什么逐 Token 生成会重复计算

在自回归生成中，模型每次只新增一个 Token。如果每一步都重新计算历史位置的 Key 和 Value，前面已经完成的投影就会被不断重复。KV Cache 在推理时保存各层历史 Token 的 K、V 表示，新 Token 到来时只计算其新的投影并追加缓存。

这里的缓存不是模型参数，也不是检索系统的文本缓存。它是一个与会话、序列长度和模型层数相关的中间激活状态。

## Prefill 与 Decode 是两种计算形态

**Prefill** 一次处理输入提示词，利用矩阵并行度形成完整上下文的 K/V，并为首个输出 Token 准备状态。这个阶段通常包含较大矩阵乘法和注意力计算，算术密度相对较高。

**Decode** 每轮只处理一个新 Token。尽管计算一个新 Query 的代价相对小，但必须读取历史 K/V 才能与已生成上下文建立注意力，因此常呈现明显的内存带宽约束。

这解释了为什么优化首 Token 延迟 TTFT 与每 Token 延迟 TPOT 往往需要不同的系统手段。

## 缓存张量形状和显存公式

对于标准分层缓存，设 B 是 batch size，L 是当前序列长度，H_kv 是 K/V 头数，d_h 是每头维度，N 是层数，b 是单元素字节数。

$$
M_{\mathrm{KV}}=2BLH_{\mathrm{kv}}d_hNb.
$$

普通写法：KV_bytes = 2 * batch * sequence_length * kv_heads * head_dim * layers * bytes_per_element。

系数 2 来自同时缓存 Key 和 Value。它描述主要缓存张量的理论体积，实际服务还会有块管理、对齐、碎片和其他运行时开销。

例如，B=1、L=8192、H_kv=8、d_h=128、N=32、b=2（FP16 或 BF16），则：

$$
M_{\mathrm{KV}}=2\times1\times8192\times8\times128\times32\times2=1\,073\,741\,824\ \mathrm{bytes}.
$$

普通写法：2 * 1 * 8192 * 8 * 128 * 32 * 2 = 1 GiB。

这仅是示例配置的缓存，不代表某个具体模型的总显存。若头数或上下文长度加倍，缓存规模按对应比例增加。

## MHA、MQA 与 GQA 如何影响缓存

多头注意力 MHA 通常让每个 Query 头拥有独立 K/V 头；MQA 将多个 Query 头共享同一组 K/V；GQA 在二者之间按组共享。

如果 Query 头数为 H_q，而 KV 头数从 H_q 降为 H_kv，缓存的主要张量在其他条件相同时按 H_kv/H_q 比例缩小。

代价并不是免费的：头共享改变了表示能力与训练结构，因此模型架构和训练配方需要同时考虑。相关细节见 [GQA 与 MLA](/docs/transformer/gqa-mla)。

## Paged KV Cache：为什么需要块管理

真实服务不只处理一个请求。不同请求长度不一、生成速度不同、可能提前结束，直接为每个请求分配最大连续上下文会浪费大量显存。

PagedAttention 一类方案把 KV Cache 管理成固定大小的逻辑块及其物理映射。请求按需增加块，块可以更灵活地复用与回收。这是内存管理策略，而不是修改 Attention 的数学定义。

理解系统时应分别回答：缓存如何计算、缓存如何分配、请求如何调度。它们对应不同的性能瓶颈。

## 评测：不要只测每秒 Token 数

最小评测应覆盖如下变量：

- 输入长度与生成长度：短提示词和长上下文对 Prefill、Decode 的影响不同。
- 并发数：请求数上升后，KV Cache 和调度器的压力不同。
- TTFT、TPOT、端到端延迟：区分首次响应和持续生成。
- GPU 显存与内存带宽：解释为何吞吐变化。
- 输出一致性：验证优化没有造成越界、错位或解码异常。

实验中固定模型权重、Tokenizer、精度和解码参数，才能把性能差异归因到缓存策略。

## 从两个生成步骤检查缓存正确性

假设 Prefill 输入 Token 为 [A,B,C]，生成模型第一次输出 D。初始 Prefill 已计算位置 0、1、2 的 K/V；下一次 Decode 只需要计算 D 在位置 3 的投影并追加缓存，而不应重新投影 A、B、C。

随后生成 E 时，Query 对历史 A、B、C、D 对应的 Key 进行匹配，输出的注意力权重仍覆盖整个可见上下文。KV Cache 复用的是已计算的历史状态，而不是降低当前 Query 需要读取的历史长度。

~~~text
Prefill:  A B C       → cache [K_A,K_B,K_C], [V_A,V_B,V_C]
Decode1: D (pos=3)   → read ABC, append K_D / V_D
Decode2: E (pos=4)   → read ABCD, append K_E / V_E
~~~

这里的 D、E 是新增 Token 的示意字母。真实张量还含 Batch、Layer、Head 和维度信息，且不同框架对缓存排列可能不同。

## 为什么缓存不能无限增长

对于固定模型，缓存字节数与当前上下文长度近似线性相关。当请求越来越长、并发数越来越高时，即使模型权重不变，可容纳的会话数也会下降。

服务系统通常需要在最大上下文长度、并发数和超时策略之间做容量规划。会话结束后，缓存块必须及时回收；Prefix Cache 等共享机制需要确保共享条件正确，避免让不应共享的上下文状态混入其他请求。

## 用测试证明缓存与全量推理一致

对一个短 Token 序列，分别采用每轮重新计算完整上下文和启用 KV Cache 的增量推理。保持相同权重、精度、位置编码、Causal Mask 和确定性采样，比较各步 Logits 的最大绝对差和最终生成序列。

如果出现明显偏差，优先排查：新增 Token 的位置索引、Cache 拼接维度、Batch 内序列长度管理，以及 RoPE 在当前轮次应用的位置。输出错误未必意味着缓存的数据本身损坏，也可能来自 Mask 或索引错位。


## 原始资料

- [Efficiently Scaling Transformer Inference: Multi-Query Attention](https://arxiv.org/abs/1911.02150)。
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。
