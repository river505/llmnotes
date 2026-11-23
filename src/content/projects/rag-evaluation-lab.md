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

## 一个完整样本的标注格式

评测报告能够可信，前提是样本与证据关系被清楚记录。可以用下列 JSON Lines 结构保存问题、相关证据以及不可回答标签：

~~~json
{
  "query_id": "q_001",
  "query": "该接口的默认超时时间是多少？",
  "relevant_doc_ids": ["api_v3_timeout"],
  "reference_answer": "30 秒",
  "answerable": true,
  "document_version": "v3"
}
~~~

这里的数值只是演示数据。真实项目应从公开且版本明确的文档中制作标注，并由至少一名审核者检查证据片段与目标答案是否一致。若一个问题需要多个段落共同回答，应把所有必要文档 ID 都标出。

## 不依赖评判模型的检索指标实现

先实现纯检索指标，能帮助排除生成模型带来的不确定性。下面给出 Recall@k 和 Reciprocal Rank 的参考函数：

~~~python
def recall_at_k(retrieved, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return None
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def reciprocal_rank(retrieved, relevant):
    relevant = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

docs = ["doc_8", "doc_3", "doc_5", "doc_1"]
print(recall_at_k(docs, ["doc_3"], 3))  # 1.0
print(reciprocal_rank(docs, ["doc_3"]))  # 0.5
~~~

对每个 Query 分别计算，再按查询平均；若某些问题没有相关文档，应事先定义这类问题的评价口径，而不是在分母为零时强行设为成功。

## 一份可直接使用的消融设计

| 实验组 | 召回器 | 重排器 | 生成模型 | 目的 |
|---|---|---|---|---|
| A | BM25 | 无 | 固定 | 精确词匹配基线 |
| B | Dense | 无 | 固定 | 语义检索基线 |
| C | BM25 + Dense | 无 | 固定 | 混合召回贡献 |
| D | BM25 + Dense | 有 | 固定 | 重排独立贡献 |

所有组共享文档版本、输入问题和生成模型配置。在这张表之后，可单独做 Chunk 长度或上下文预算的对照，避免把多个因素同时改变而无法解释收益来源。

报告不仅需要总体指标，也需要逐样本的检索列表、最终答案、错误归因和成本记录。最后把“查不到”“排序错误”“有证据但回答错误”三类失败分别汇总，形成明确的下一轮优化清单。


## 最终交付

提交一套实验配置、索引脚本、标注样本、逐条预测日志、指标计算脚本及评测报告。报告应明确数据版本、模型版本和测试时间，以便他人复现。

参考：[RAG 评测技术文档](/docs/rag/evaluation) 和 [原始 RAG 论文](/papers/retrieval-augmented-generation)。
