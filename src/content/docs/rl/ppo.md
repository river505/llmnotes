---
title: "PPO：为什么策略更新需要裁剪目标"
description: "从策略概率比、优势估计与 Clipped Objective 推导 PPO，并解释其在 RLHF 中的使用方式。"
category: "rl"
section: "强化学习"
tags: ["PPO","RLHF","强化学习"]
difficulty: "入门"
order: 8
updated: "2026-11-23"
created: "2026-05-29"
draft: false
featured: true
cover: ""
readingTime: "约 14 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 问题起点：一次更新能把策略改变多少

策略梯度方法依据采样数据调整动作概率，但如果新策略相对旧策略变化太大，原先的数据分布就可能不能代表新的行为。更新过于激进时，训练会出现回报下降、性能波动和难以解释的行为漂移。

PPO（Proximal Policy Optimization）希望利用当前采样数据改善策略，同时抑制过大的策略变化。其中广泛使用的 PPO-Clip 通过一个裁剪目标限制离开参考概率比范围后继续获得的优化收益。

## 策略概率比从何而来

设旧策略为 π_old、新策略为 π_θ，在同一状态 s 下选择动作 a 的概率比为：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

普通写法：ratio = new_policy_probability(action|state) / old_policy_probability(action|state)。

r_t=1 表示该动作概率未变化；r_t>1 表示上升；r_t<1 表示下降。这个比值本身不说明动作好坏，因此还需要优势估计 A_t。

## Advantage：动作结果相对基线有多好

优势函数刻画采取动作 a 的预期价值相对状态价值的差异：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

普通写法：advantage = action_value - state_value。

A>0 意味着该动作相对基线更好，期望提升其概率；A<0 则倾向降低其概率。实际 PPO 训练会用 GAE 等方法根据奖励和价值预测估计优势，并可能对优势进行归一化。

## PPO-Clip 的优化目标

标准的裁剪目标为：

$$
L^{\mathrm{CLIP}}(\theta)
=\mathbb E_t\left[
\min\left(r_t(\theta)A_t,\,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)
\right].
$$

普通写法：objective = mean(min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage))。

训练通常最大化该目标，或最小化其负值。超出裁剪范围时，某些方向的继续变化不再增加目标收益；但裁剪并非数学上强制要求策略分布绝对不会越过该范围。

## 数值例子：正优势和负优势

假设 ε=0.2。对于 A=+1 的好动作，如果新旧概率比 r=1.3，未裁剪项为 1.3，裁剪项为 1.2，取较小值得到 1.2。这阻止该动作的概率过度上升带来额外收益。

对于 A=-1 的差动作，如果 r=0.7，未裁剪项为 -0.7，裁剪项为 -0.8，取较小值得到 -0.8。这样继续降低该动作概率也不再带来同样的优化激励。

理解这个例子要抓住符号：好动作和坏动作对应的“有利方向”不同，裁剪函数通过同一公式处理两者。

## 与 KL 惩罚有什么区别

PPO-Clip 约束的是概率比目标中的优化激励；KL 惩罚则直接衡量策略分布之间的差异。在 RLHF 中常使用参考模型的 KL 正则化，控制对齐策略偏离监督微调模型的程度。

两者可以同时存在：一个服务于稳定策略更新，另一个服务于参考分布附近的偏好优化。不能把裁剪参数 ε 和 KL 系数当作同一个超参数。

## 从强化学习转向语言模型

在语言模型对齐中，状态可以包含提示词和已生成的 Token 前缀，动作对应下一个 Token。奖励可能来自偏好模型或其他反馈，价值模型估计状态价值，PPO 根据采样输出更新策略参数。

这种映射会带来奖励稀疏、序列长度变化和高维动作空间等问题，因此实际训练还涉及长度规范化、参考模型、奖励归一化以及多批次优化策略。

## PPO、DPO 和 RLHF 的关系

RLHF 是偏好数据、奖励建模、策略优化与评估组成的流程。PPO 是其中一种策略优化方法。DPO 利用偏好回答对构造离线分类目标，在适用条件下可省去显式训练奖励模型并在线使用 PPO 的部分步骤。

比较时应关注数据收集方式、优化预算、输出质量与评估协议，不应简单将新旧方法的标签当成性能排序。

## 如何复现实验并排查

最小实验应记录奖励曲线、策略 KL、裁剪比例、优势分布、生成长度和独立任务成功率。若奖励升高而任务质量下降，应优先检查奖励模型偏差和数据分布，而不是只调学习率。

若训练不稳定，观察概率比的分布、极端优势值、价值函数误差与每轮更新次数，能够帮助区分优化问题和奖励设计问题。

## 原始资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al., 2017。
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)。
- [DPO 论文研读](/papers/dpo)。
