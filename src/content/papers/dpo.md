---
title: "DPO：从偏好对直接学习模型策略"
description: "解释偏好优化的分类目标、参考模型与温度参数，并比较 DPO 和 PPO/RLHF 的优化链路。"
category: "papers"
section: "论文研读"
tags: ["DPO", "RLHF", "偏好优化"]
difficulty: "进阶"
order: 4
updated: "2026-11-23"
created: "2023-05-01"
draft: false
featured: true
readingTime: "约 13 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 偏好学习的问题是什么

面对同一个提示，可能存在多个都通顺、却质量不同的回答。偏好数据提供的是一对回答的相对排序：被选择的回答 y_w 优于被拒绝的回答 y_l。

在传统的基于奖励模型的 RLHF 流程中，通常先用偏好数据拟合奖励模型，再通过策略优化调整语言模型。DPO 尝试将这一流程收缩为直接利用偏好对优化策略。

## 从 KL 约束的奖励最大化出发

考虑在参考策略 π_ref 附近优化奖励 r：

$$
\max_\pi\;\mathbb E_{y\sim\pi(\cdot|x)}[r(x,y)]-\beta D_{\mathrm{KL}}(\pi(\cdot|x)\Vert\pi_{\mathrm{ref}}(\cdot|x)).
$$

普通写法：maximize expected reward - beta * KL(policy || reference)。

在相应假设下，最优策略与奖励之间具有对数概率比关系。DPO 利用这一对应，将偏好比较改写为策略概率的分类目标。

## DPO 的训练目标

较优回答 y_w 与较差回答 y_l 的损失为：

$$
\mathcal L_{\mathrm{DPO}}=-\mathbb E\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right]\right).
$$

普通写法：DPO loss = -E log sigmoid(beta * [log policy/reference on chosen - log policy/reference on rejected])。

训练通过增加较优回答相对参考策略的概率，并降低较差回答的相对概率，推动偏好排序。β 控制对数概率差的尺度，实际训练表现取决于数据质量和实现配置。

## 为什么仍然需要参考模型

如果只最大化优选回答概率而不保留参照，模型可能过分拟合偏好数据中的风格或局部模式。参考策略在目标函数中提供了基准分布。

参考模型通常来自监督微调后的检查点。DPO 不需要显式在线生成样本并反复更新奖励模型，但这不意味着它可以忽略分布偏移或偏好标注噪声。

## 与 PPO 的区别

PPO 使用策略比和优势估计进行迭代更新，常涉及奖励模型、价值估计和在线或批量采样。DPO 更接近离线的偏好分类目标，训练实现通常更简单。

它们的比较不能只看训练代码量。需考察偏好数据是否固定、是否需要在线探索、奖励反馈的结构，以及最终评估方式。

## 数据处理中的关键细节

一对偏好数据必须共享相同输入上下文，否则概率差失去明确的对比语义。需要正确截断 prompt 与 answer，并保证 chosen/rejected 的 Token 概率求和口径一致。

在长回答场景中，长度差异可能影响序列对数概率。训练和评估都应检查回答长度分布，避免将“偏好提升”误判为模型更倾向生成某种长度。

## 评估偏好训练效果

可用独立偏好对测试排序准确率，但排序准确不完全代表真实助手质量。还应根据任务检查指令遵循、事实准确性、安全要求和泛化能力。

推荐对模型进行盲评并使用配对样本，展示整体结果及分任务结果。训练样本与最终评估集不应复用同一批偏好对。

## 原论文

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), Rafailov et al., 2023。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), Ouyang et al., 2022。
- [PPO 技术文档](/docs/rl/ppo)。
