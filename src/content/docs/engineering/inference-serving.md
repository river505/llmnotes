---
title: "LLM 推理服务：延迟、吞吐与显存之间的关系"
description: "从 Prefill/Decode、请求批处理、KV Cache 和量化出发，设计能够解释性能变化的推理服务基准。"
category: "engineering"
section: "推理工程"
tags: ["推理部署", "vLLM", "性能评测"]
difficulty: "进阶"
order: 1
updated: "2026-11-23"
created: "2026-11-23"
draft: false
featured: true
readingTime: "约 13 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 同一模型，为什么部署表现可以完全不同

模型推理不是只有一次前向传播。真实服务同时面对多个请求、不同上下文长度、不同生成长度、首 Token 延迟、持续生成速度和显存配额。调度策略与内存布局的改变，可能显著影响用户体验，而不修改模型权重。

因此，应把推理优化理解为一个多目标系统问题：吞吐、响应时间、显存占用和输出一致性需要同时衡量。

## Prefill 与 Decode 的资源特点

Prefill 一次处理完整输入上下文，存在较强的矩阵并行性。Decode 每次新增一个 Token，运算规模更小，但需要读取历史 KV Cache，因此容易受到显存带宽与请求调度的影响。

单请求速度的提升不必然意味着高并发吞吐增加。例如，扩大 batch 可以更有效利用 GPU，但排队与调度会使部分请求的尾延迟升高。

## 四个必须区分的指标

- TTFT（Time to First Token）：收到请求到第一个可见 Token 的时间，包括排队和 Prefill。
- TPOT（Time per Output Token）：生成阶段平均每个 Token 的时间。
- 吞吐量（Throughput）：单位时间完成的请求数或生成 Token 数，应说明统计口径。
- 峰值显存（Peak Memory）：模型参数、KV Cache、激活和运行时管理的总成本。

对用户交互式服务，通常更关注 TTFT 和尾延迟；批量离线处理则可能优先优化整体吞吐。

## KV Cache 和批处理

历史 KV 的理论字节数与 batch、长度、KV 头数和层数成正比：

$$
M_{\mathrm{KV}}=2BLH_{\mathrm{kv}}d_hNb.
$$

普通写法：KV_memory = 2 * batch * context_length * kv_heads * head_dim * layers * bytes_per_element。

Continuous Batching 在 Token 生成步骤之间动态插入和移除请求；PagedAttention 等内存管理方案让长短不一的请求更灵活地分配缓存块。一个负责调度，一个负责缓存布局，两者共同影响服务效率。

## 量化与精度

模型参数量化会降低权重内存，但不同算子是否支持相应低精度内核决定最终速度。KV Cache 量化影响的是随上下文长度增长的状态内存；它与参数量化不是同一个优化问题。

应分别测参数精度、缓存精度和计算精度，不要仅用“INT8 比 FP16 快”这样的静态判断替代硬件基准。

## 最小实验协议

固定模型、Tokenizer、硬件、并发生成策略和随机种子，系统扫描以下维度：

1. 输入长度：短、中、长上下文三个量级。
2. 输出长度：短回答与长生成两类。
3. 并发数：1、4、16、32 等可负载范围。
4. 批处理策略：静态 batch 与持续批处理。
5. 缓存策略与精度：固定其他变量后分别测试。

建议报告 p50/p95 TTFT、p50/p95 TPOT、吞吐量、峰值显存和失败率，再根据应用场景建立约束下的优化目标。

## 避免把性能测试做成单一数字

推理服务的性能会随着输入、输出长度分布改变。仅展示某一批请求的 tokens/s，无法预测真实流量的结果。更有价值的是请求长度与并发的二维性能曲面，以及性能退化的解释。

## 延伸资料

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)。
- [vLLM Documentation](https://docs.vllm.ai/)。
