---
title: "预训练、SFT 与偏好优化：三种目标如何塑造模型行为"
description: "从损失函数和数据分布出发，区分语言建模、指令适配与偏好对齐的训练机制。"
category: "llm-basics"
section: "基础与训练"
tags: ["预训练", "SFT", "DPO", "RLHF"]
difficulty: "进阶"
order: 3
updated: "2026-11-23"
created: "2026-11-23"
draft: false
featured: true
readingTime: "约 13 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 三种训练目标，对应三种不同的问题

模型生成一段文本之前，先要能预测语言模式；但能够补全互联网文本，不等于能够可靠地执行用户指令；能够响应指令，也不等于在多个候选回答中倾向更有帮助、更符合偏好的那个。预训练、监督微调和偏好优化由此形成不同的目标。

工程上应先明确当前要改变什么：是扩充语言与领域知识，还是改进特定任务的输入输出格式，抑或调整回答的排序偏好。方法选择的依据是目标函数和数据类型，而非训练阶段的名称。

## 预训练：条件分布的最大似然估计

自回归模型对 Token 序列的概率分布作因子分解：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

普通写法：p(x1:T) = product over t of p(xt | x<t)。

对应的负对数似然损失：

$$
\mathcal L_{\mathrm{NLL}}=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

普通写法：NLL = -sum_t log p_theta(x_t | x_<t)。

模型学习的是条件预测分布，训练信号由语料中的下一个 Token 提供。规模增大改变了知识覆盖和泛化能力，但并不会自动把训练目标变成事实核验或工具执行。

预训练还涉及数据去重、文档混合、序列包装、学习率计划与并行策略。这些操作并非附属细节：数据分布会改变模型先验，批次和上下文长度则直接影响吞吐与数值稳定性。

## 监督微调：从补全文本变成条件响应

SFT 的训练样本由指令、上下文和目标回答组成。通常只对回答部分计算损失，避免模型把系统指令或用户输入当作需要原样复现的预测目标。

用掩码 m_t 标记参与优化的回答位置：

$$
\mathcal L_{\mathrm{SFT}}=-\sum_t m_t\log p_\theta(y_t\mid x,y_{<t}).
$$

普通写法：SFT loss = -sum_t mask_t * log p_theta(answer_token_t | instruction, previous answer)。

一个常见问题是将训练数据的模板格式直接混入普通文本，导致推理时角色边界不稳定。因此应核查 Tokenizer 的 Chat Template、特殊 Token、标签掩码和截断方式，而不只是观察训练损失是否下降。

## 偏好优化：学习两个回答的相对排序

偏好数据通常包含同一输入下的较优回答 y_w 与较差回答 y_l。排序信息与单一参考答案不同：它描述两种输出的相对选择，并不要求写出唯一正确的目标文本。

DPO 使用策略模型与参考模型的对数概率差构造目标，绕开显式训练奖励模型再使用 PPO 的在线循环：

$$
\mathcal L_{\mathrm{DPO}}=-\mathbb E\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right)\right].
$$

普通写法：DPO = -E log sigmoid(beta * [log policy/reference on preferred - log policy/reference on rejected])。

DPO 并不等于在所有场景都优于 PPO。两者的区别首先在于数据收集与优化流程：离线偏好对足够时，可用更简单的目标；需要与奖励信号在线交互时，策略优化仍有用途。

## 从数据到训练的实现检查

建议把训练管线拆为四个可审计部分：

1. **样本构造**：每条数据的角色、输入、目标、偏好标签与元数据是否明确。
2. **损失掩码**：padding、输入指令和无效位置是否错误参与梯度。
3. **训练稳定性**：混合精度、梯度累积、激活检查点、学习率与权重更新是否一致。
4. **行为评估**：在独立数据上检验任务成功率、格式合规率、事实准确性与偏好一致性，而不只比较训练损失。

## 什么证据能说明对齐有用

如果目标是提升指令遵循，应构造覆盖不同指令结构的测试集，比较训练前后模型对约束条件的遵循率。如果目标是提升回答质量，需定义评价维度并设计盲评或配对比较。

同一批数据同时用于生成偏好、训练策略与最终评价，会产生评价偏差。模型输出还应按数据子集、任务类型和失败模式拆开观察，避免一个平均分掩盖结构性退化。

## 与参数高效微调的关系

LoRA 改变的是参数更新的表示形式，不直接改变训练目标。LoRA 可以与 SFT、DPO 或其他损失组合使用；“微调方式”与“优化任务”必须分清。

继续阅读：[LoRA 论文研读](/papers/lora)、[DPO 论文研读](/papers/dpo) 与 [PPO 原理](/docs/rl/ppo)。

## 原始资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022。
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023。
