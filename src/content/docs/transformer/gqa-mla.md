---
title: "GQA 与 MLA：注意力中的 KV 压缩"
description: "比较分组共享与潜变量压缩两条路线，从缓存开销、计算结构和位置编码理解设计取舍。"
category: "transformer"
section: "Transformer 与注意力"
tags: ["GQA", "MLA", "KV Cache"]
difficulty: "高级"
order: 4
updated: "2026-11-23"
created: "2026-11-23"
draft: false
featured: true
readingTime: "约 12 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 核心矛盾不是 Query，而是历史 KV

在长上下文推理中，历史 Key 与 Value 需要持续存放并被新 Token 访问。标准多头注意力增加头数会带来更丰富的子空间，但也会提高缓存体积和内存读写量。

GQA（Grouped-Query Attention）和 MLA（Multi-head Latent Attention）从不同角度处理这个矛盾：前者减少 K/V 头数，后者通过低维潜变量表示实现压缩。

## GQA：让一组 Query 共享 K/V

设 Query 总头数 H_q，Key/Value 头数 H_kv，分组数 g=H_q/H_kv。GQA 中每组多个 Query 头共享同一组 Key 和 Value。

$$
g=\frac{H_q}{H_{\mathrm{kv}}}.
$$

普通写法：group_size = query_heads / kv_heads。

如果 H_q=32、H_kv=8，每四个 Query 头共享一组 K/V。相比 32 个独立 KV 头，其核心缓存体积缩小到原来的 1/4。具体实现仍需处理 head layout 和 kernel 对齐。

GQA 可视为 MHA 与 MQA 之间的折中：MHA 的 K/V 头数与 Query 一致，MQA 只使用一个共享 KV 头；GQA 允许更细的分组。

## MLA：用低秩潜变量缓存历史状态

MLA 将 K/V 的联合信息投影到低维潜变量 c_t，再通过投影恢复用于注意力的表示。一个抽象写法为：

$$
c_t=W_{\mathrm{down}}h_t,\qquad
k_t=W_k^{\mathrm{up}}c_t,\quad v_t=W_v^{\mathrm{up}}c_t.
$$

普通写法：latent_kv = down_projection(hidden); key/value = up_projection(latent_kv)。

实际 MLA 实现涉及矩阵吸收与旋转位置编码的分离设计。推理时并不一定真的显式重建完整 K/V 后再执行标准注意力；通过重排线性代数，可以减少运行时存取与重复计算。

因此比较 GQA 与 MLA 时，不能只看一个压缩比，还要检查 attention kernel、RoPE 的处理、解码开销和训练兼容性。

## 理论缓存量比较

GQA 的主要缓存随 H_kv × d_h 增长；潜变量式 KV 缓存则主要取决于压缩后维度 d_c 及必要的额外位置状态。

在忽略常数和附加状态的抽象下：

$$
M_{\mathrm{GQA}}\propto 2BLNH_{\mathrm{kv}}d_h,\qquad M_{\mathrm{MLA}}\propto BLNd_c.
$$

普通写法：GQA cache ~ 2 * batch * length * layers * kv_heads * head_dim；MLA cache ~ batch * length * layers * latent_dim。

这不是同一架构内的逐项恒等替换。特别是位置编码与核函数实现不同，必须用真实运行参数核算。

## 决策应由哪些实验支持

首先固定模型规模、上下文长度和硬件平台，分别记录 Prefill/Decode 的速度、峰值显存、不同并发下的吞吐与延迟分位数。其次需要语言建模或下游能力评测，以防缓存压缩造成有效容量下降。

另外要区分训练时的模型设计选择和已有模型的部署优化：GQA 的头配置通常属于模型架构，不能在不做适配的情况下随意修改权重。

## 延伸阅读

- [KV Cache：自回归推理中的内存与延迟](/docs/transformer/kv-cache)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
