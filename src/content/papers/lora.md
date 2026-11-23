---
title: "LoRA：低秩更新如何改变微调成本"
description: "从矩阵增量、参数量与优化目标出发，分析 LoRA 的表达方式、合并部署和实验验证。"
category: "papers"
section: "论文研读"
tags: ["LoRA", "PEFT", "微调"]
difficulty: "进阶"
order: 3
updated: "2026-11-23"
created: "2021-06-01"
draft: false
featured: true
readingTime: "约 12 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 问题：大模型适配不一定需要更新全部参数

全参数微调会为每个可训练权重维护梯度与优化器状态。模型参数量很大时，训练显存、检查点和多任务版本管理成本随之上升。LoRA 通过低秩矩阵表达权重增量，让原有参数保持冻结，只训练一组小矩阵。

LoRA 是参数化方式，不是一种新的语言建模损失。它可以与监督微调、分类目标、偏好优化等任务组合使用。

## 权重增量的数学形式

给定线性层原权重 W∈ℝ^(d_out×d_in)，LoRA 将更新写为：

$$
W'=W+\Delta W,\qquad \Delta W=\frac{\alpha}{r}BA,
$$

其中 A∈ℝ^(r×d_in)、B∈ℝ^(d_out×r)，r 远小于输入输出维度。

普通写法：W_new = W_frozen + (alpha/r) * B @ A。

全参数更新需要 d_out×d_in 个参数；LoRA 增量需要 r(d_in+d_out) 个参数。压缩率随维度和秩改变，不能脱离目标层大小给固定比例。

## 为什么低秩是可行的假设

LoRA 假设具体任务的适配更新可以在较低维子空间中得到有效表达。它并没有声称所有可能权重变化都严格低秩，而是通过限制更新自由度换取参数效率。

当任务需要的表示变化超过当前秩能够表达的范围时，增大 r、调整目标模块或改用其他微调策略可能更合适。

## 选择哪些模块注入

注意力层中的 Q/K/V/O 投影，以及某些前馈层线性变换，都可能成为注入目标。不同模块对性能、显存、训练速度的影响不同；“在所有层放相同秩”只是一个基线选择。

实验记录应注明 target_modules、r、alpha、dropout、是否训练 bias、量化配置和优化器。否则两次标为 LoRA 的实验未必可以直接比较。

## 初始化、训练与权重合并

常见实现将增量路径初始化为零影响，从而在开始训练时保持原预训练模型的输出。训练中冻结 W，只更新 A 和 B。

部署时可以将 BA 合并到基础权重中，避免在线推理额外执行两个矩阵投影；但如果要在同一基础模型上动态切换多个 adapter，也可以选择保留独立适配器。

## LoRA、QLoRA 与任务损失的区别

QLoRA 利用量化基础权重降低训练显存，并通过低秩适配器学习增量。量化策略和 LoRA 矩阵本身承担不同功能。SFT 或 DPO 则决定优化哪些输出行为。

三者分属三个轴：参数存储形式、可训练参数范围、训练目标。讨论微调方案时最好同时给出这三项。

## 如何设计合理消融

固定数据切分、训练 Token 数、优化预算和模型评估集，比较全参数微调、不同秩的 LoRA、不同 target_modules 和必要的低资源基线。

除了最终任务指标，至少还应记录训练峰值显存、总训练时间、检查点大小和推理部署形式。一个配置即使效果接近，如果显存和版本管理成本显著更低，也有明确的工程价值。

## 原始资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), Hu et al., 2021。
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), Dettmers et al., 2023。
- [预训练、SFT 与偏好优化](/docs/llm-basics/training-alignment)。
