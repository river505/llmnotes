---
title: "MoE 混合专家：Top-k 路由、负载均衡与计算成本"
description: "用可手算的 Top-2 路由例子解释专家混合、稀疏激活、负载均衡和通信成本。"
category: "llm-basics"
section: "基础与训练"
tags: ["MoE", "Top-k", "专家路由"]
difficulty: "进阶"
order: 4
updated: "2026-11-23"
created: "2026-11-23"
draft: false
featured: true
readingTime: "约 15 分钟"
unlockRequired: true
previewRatio: 0.35
---

## MoE 为什么不是简单增加网络层数

稠密 Transformer 中，每个 Token 都通过同一组 FFN 参数。如果增大 FFN 的宽度，模型参数和每 Token 计算量会同时增加。MoE（Mixture of Experts）通过多个专家网络与一个路由器，把“总参数容量”和“单次激活计算”拆开。

每个 Token 只选择部分专家执行，而非调用全部专家。MoE 因此有机会在提高模型容量的同时，控制每个 Token 的计算开销。但全部专家参数仍需储存，路由和通信还会引入新的系统成本。

## 路由器如何产生专家概率

给定 Token 表示 h，路由层计算 E 个专家的分数：

$$
z=W_rh,\qquad p_e=\frac{e^{z_e}}{\sum_{j=1}^{E}e^{z_j}}.
$$

普通写法：router_logits = W_router @ hidden_state；router_probs = softmax(router_logits)。

随后选取概率最大的 k 个专家组成集合 T(h)，并按所选权重聚合专家输出：

$$
y=\sum_{e\in T(h)}\widetilde p_e f_e(h),\qquad
\widetilde p_e=\frac{p_e}{\sum_{j\in T(h)}p_j}.
$$

普通写法：output = sum(normalized_selected_probability[e] * expert_e(hidden_state))。

式中的 f_e 通常表示某个非线性 FFN。实际架构可能使用不同的路由归一化规则、共享专家和分数偏置，这里给出稀疏专家混合的常用抽象。

## 四个专家的 Top-2 手算例子

假设路由器为某个 Token 输出概率 [0.50, 0.30, 0.15, 0.05]，Top-2 选择前两个专家。如果只对选中概率重新归一化，则新权重为：

$$
\widetilde p_1=0.50/0.80=0.625,\quad
\widetilde p_2=0.30/0.80=0.375.
$$

普通写法：selected_weights = [0.625, 0.375]。

最终输出为 0.625·f₁(h)+0.375·f₂(h)。第 3、4 个专家不会参与这个 Token 的 FFN 计算，但它们可能被其他 Token 选择，因此仍然需要训练与存储。

## 如何把一批 Token 发给不同专家

一批输入经过路由后，系统获得一个形如 [num_tokens, k] 的专家索引矩阵。执行器需要根据这个矩阵把 Token 分组、发送到对应专家，处理后再按照原始 Token 顺序把输出合并。

~~~text
Token representations → Router → Top-k indices
                           │
                     Dispatch tokens
                    /      |        \
                Expert 0 Expert 1  Expert 2 ...
                    \      |        /
                    Weighted combine
                           │
                     Output tokens
~~~

真正困难的是不均匀的路由：不同专家可能收到完全不同数量的 Token，造成设备空闲、队列拥堵和跨设备通信。稀疏激活不意味着系统性能会自动按专家数线性改善。

## 负载均衡与专家容量

如果大部分 Token 总是被分给少数专家，其余专家难以充分训练，热点专家的计算量也会成为瓶颈。一个直观指标是第 e 位专家的负载比例：

$$
f_e=\frac{n_e}{\sum_j n_j}.
$$

普通写法：expert_load_fraction = assigned_tokens_for_expert / total_expert_assignments。

一些 MoE 通过辅助均衡损失调整专家分配，也有架构通过动态 Bias 等方式减少额外损失的影响。不同设计的权衡需要结合具体模型论文说明。

容量因子决定每个专家为 Token 分配的缓冲上限。容量偏小会导致 Token 溢出或重路由；容量过大可能造成填充与显存浪费。这属于计算图之外的重要系统设计。

## 对比 Dense FFN 与稀疏 MoE

| 指标 | 稠密 FFN | Top-k MoE |
|---|---|---|
| 每个 Token 参与的网络 | 单一 FFN | 选中的 k 个专家 |
| 总参数 | 接近每 Token 激活参数 | 可以明显高于激活参数 |
| 门控与调度 | 不涉及专家路由 | 需要计算路由与 Dispatch |
| 负载不均衡 | 不涉及专家分配 | 需要专项处理 |
| 多设备通信 | 取决于并行方案 | 可能需要 All-to-All |

总参数量、激活参数量、显存占用、实际 FLOPs 和端到端吞吐是不同指标，不能只用“Top-k 较小”就推断部署费用。

## 可以运行的路由代码

~~~python
import numpy as np
logits = np.array([2.4, 1.9, 0.6, -0.5])
p = np.exp(logits - logits.max())
p = p / p.sum()
index = np.argsort(p)[-2:][::-1]
selected_weight = p[index] / p[index].sum()
print("selected expert ids:", index)
print("selected weights:", selected_weight)
~~~

这个示例展示门控选择，不涉及真正的分布式 Expert Dispatch。生产系统需要把路由结果映射到 GPU 内存与通信拓扑，才能得到高效率的专家执行。

## 研究与工程评估

最小对照实验应记录固定训练 Token 预算下的能力指标，同时比较稠密基线、Top-1、Top-2 和不同专家数量。系统指标至少包括最大专家负载、各专家分配的变异程度、溢出比例、实际吞吐、峰值显存与通信时间。

MoE 与跨模态专家模型也不完全等价。Transformer MoE 通常以 Token 为路由单位；时间序列或多模态系统还可能按样本、任务或模态进行专家选择，其门控粒度与训练信号不同。

## 原始资料

- [Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
