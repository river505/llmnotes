---
title: "RAG 评测实验室：从标注证据到诊断报表"
description: "设计一个小规模但完整的检索增强系统评测项目，覆盖召回、重排、答案忠实度与成本。"
category: "projects"
section: "项目实践"
tags: ["RAG", "项目", "模型评估"]
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

## 项目定义

输入是一个版本固定的公开文档集和一组带标注证据的问题；输出是检索结果、生成回答以及能解释每次失败来自哪一层的评测报告。

项目不需要复杂前端。首先确保系统能复现：从相同文档版本构建索引、在相同问题集合上运行多个策略，并输出逐样本记录和总体指标。

## 数据与标注

建议准备 30–100 篇公开文档，包含短文、长文和结构化段落。标注 50 个左右的问题，每个问题至少包含正确证据的位置、问题类型、预期答案或不可回答标签。

这是项目规模示例，不是算法成立所需的统计样本阈值。若文档涉及时间版本，应固定采集日期并保留原文快照。

## 实现四组检索基线

先实现 BM25-only 和 Dense-only，再加入 Hybrid 和 Hybrid+Reranker。统一 top-k 值、文档解析结果与评价集。

~~~text
source documents
  -> parse and normalize
  -> passages with metadata
  -> lexical / vector indexes
  -> retrieve candidates
  -> rerank optional
  -> build evidence context
  -> generate with citations
  -> log evidence and answer
~~~

查询日志至少保留：query_id、候选文档 ID、排序位置、检索分数、对应版本和最终被放入上下文的证据。

## 两层离线评估

**检索层**：用 Recall@k、MRR、nDCG 衡量证据覆盖与排序。按问题类型展示分组结果，特别是精确实体、跨段问题和长尾知识。

**生成层**：测答案正确率、引用可支持性、证据忠实度和不可回答问题的拒答表现。尽量将答案拆成可核验陈述，而不是只使用单个模型打分。

## 错误追踪与可解释报表

对于失败样本，先查看正确证据是否被原始候选召回。如果没有，分类为召回问题；如果已召回但被排序或截断移除，分类为重排或上下文问题；如果证据在上下文中但回答错误，分类为生成问题。

这种分层分析比单独展示最终正确率更容易指导后续改进。

## 性能与成本

记录向量构建耗时、每请求检索延迟、重排开销、生成 Token 数和端到端响应时间。比较检索增强带来的准确性收益是否值得额外延迟。

当某组策略表现更好时，检查是否使用了更高的上下文预算或更昂贵的模型；必要时在相同成本约束下再次比较。

## 最终交付

提交一套实验配置、索引脚本、标注样本、逐条预测日志、指标计算脚本及评测报告。报告应明确数据版本、模型版本和测试时间，以便他人复现。

参考：[RAG 评测技术文档](/docs/rag/evaluation) 和 [原始 RAG 论文](/papers/retrieval-augmented-generation)。
