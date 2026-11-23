---
title: "Retrieval-Augmented Generation：从参数记忆到外部知识"
description: "研读 RAG 原始论文中的参数化/非参数化记忆、潜变量文档以及知识密集型任务的生成机制。"
category: "papers"
section: "论文研读"
tags: ["RAG", "检索", "生成模型"]
difficulty: "进阶"
order: 6
updated: "2026-11-23"
created: "2020-05-01"
draft: false
featured: true
readingTime: "约 12 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 研究问题：知识不必都写进模型参数

大语言模型的参数中包含大量训练语料的统计模式，但参数知识更新成本高，对知识来源的追踪也不直接。检索增强生成提出把神经生成模型与外部文档索引结合：模型负责利用上下文生成，检索系统负责提供相关文档。

原始 RAG 论文围绕知识密集型 NLP 任务，探索参数化记忆与非参数化记忆的结合，而今天的企业知识库问答只是这一思路的后续工程应用。

## 两种记忆如何协作

参数化记忆是生成模型权重中编码的能力；非参数化记忆是可检索的外部文档集合。给定问题 x，检索器对候选文档 z 分配概率 p(z|x)，生成器再建模 p(y|x,z)。

$$
p(y|x)=\sum_z p_\eta(z|x)p_\theta(y|x,z).
$$

普通写法：answer_probability = sum_over_documents(retrieval_probability * generation_probability_given_document)。

上式是文档层潜变量视角。实际工程通常采用有限个 top-k 文档近似，不会对整个语料求和。

## RAG-Sequence 与 RAG-Token

原论文讨论不同的文档潜变量使用方式。RAG-Sequence 在生成整个目标序列时使用相同文档潜变量；RAG-Token 可以在不同输出 Token 上隐式依赖不同文档。

这些定义反映了一个重要设计轴：模型能否在生成过程中动态改变证据来源。现代应用中常见的“先检索，再拼接到提示词”是更直接的一种工程实现，并不与原论文的概率建模完全等价。

## 为什么需要联合考虑检索器与生成器

只有相关性较高的文档不代表答案一定正确。检索器的训练目标影响候选证据，生成器对文档噪声的鲁棒性影响最终回答。两个模块的误差可以相互放大。

因此，不应只问“向量库选哪一个”。更重要的是：文档是否被正确解析、相关证据是否进入 top-k、上下文有没有冲突，以及生成器是否能引用正确来源。

## 工程实现的四层结构

1. 数据层：文档解析、版本控制、权限与元数据。
2. 召回层：稀疏检索、稠密检索、查询扩展与混合召回。
3. 排序层：重排、过滤、去重以及上下文预算分配。
4. 生成层：引用、回答约束、无法回答的处理与结果审计。

这四层分别对应不同类型的失败分析，适合建立可复现的系统级消融。

## 评测：把最终回答拆成证据与生成

首先测 Recall@k、MRR、nDCG 等检索指标，之后测答案正确性、证据忠实度与引用可靠性。只用最终回答的人工评分会使排错困难。

对时间敏感的知识库，还必须固定文档版本和查询时间。否则模型给出旧答案时，很难判断是检索失败还是语料已过时。

## 原始资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), Lewis et al., 2020。
- [RAG 技术文档](/docs/rag/what-is-rag)。
- [RAG 分层评测](/docs/rag/evaluation)。
