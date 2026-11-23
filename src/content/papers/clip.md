---
title: "CLIP：图文对比学习如何建立共享表示空间"
description: "通过图像—文本批内对比目标解释零样本分类、嵌入对齐以及多模态检索。"
category: "papers"
section: "论文研读"
tags: ["CLIP", "多模态", "对比学习"]
difficulty: "进阶"
order: 7
updated: "2026-11-23"
created: "2021-03-01"
draft: false
featured: true
readingTime: "约 12 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 研究问题：视觉表征能否直接从语言监督学习

传统图像分类需要预定义类别与人工标注。CLIP 尝试从互联网上的图像—文本配对数据中学习视觉与文本的共享语义空间，使模型能够用文本描述匹配图像，而不仅依赖封闭类别编号。

这种思路为零样本图像分类、跨模态检索和后续视觉语言模型提供了关键表征基础。

## 双编码器与共享空间

CLIP 包含图像编码器 f(x) 和文本编码器 g(t)。两侧输出向量被归一化，在共享表示空间中使用余弦相似度比较。

给定一个 batch 内 N 个图像—文本正配对，计算 N×N 的相似度矩阵；对角线对应正确配对，其他位置作为批内负例。

## 对称的对比学习目标

设相似度为 s_ij，温度为 τ，图像到文本方向的损失为：

$$
\mathcal L_{\mathrm{i2t}}=-\frac{1}{N}\sum_i \log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)}.
$$

普通写法：image-to-text loss = average negative log softmax of matching image-text score。

训练通常同时计算文本到图像方向，再将两个方向的损失组合。这样模型学习的是跨模态的相对匹配关系，而不是直接生成描述文本。

## 零样本分类为何可行

把候选类别写成自然语言文本模板，例如“a photo of a dog”，使用文本编码器得到类别嵌入。测试图像的向量与各类别嵌入比较，最高相似度对应预测类别。

分类标签因此转换为文本描述。不同模板可能影响结果，工程评测中需要固定或报告模板集合，否则分类性能不具有可比较性。

## CLIP 与生成式多模态模型的区别

CLIP 的主要任务是对齐图像和文本表示，不负责自回归生成长文本。许多视觉语言模型会利用视觉编码器抽取特征，再通过投影模块接入语言模型生成过程。

因此，双编码器检索、跨模态对齐和多模态生成属于不同任务层次，不宜把它们都笼统称为“图文大模型”。

## 复现实验建议

可以比较不同文本提示模板、负样本构造、批次大小和嵌入归一化策略，对图文检索报告 Recall@k，对零样本分类报告准确率并记录类别定义。

不同的数据域会导致语言监督覆盖不足，评测应包含预训练分布之外的图像类型，以识别真实泛化能力。

## 论文与链接

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), Radford et al., 2021。
- [多模态大模型技术入门](/docs/multimodal/what-is-multimodal)。
