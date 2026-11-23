---
title: "RAG 评测：把召回、排序与生成误差拆开"
description: "使用 Recall@k、MRR、nDCG、证据忠实度和答案正确率，为 RAG 建立可定位问题的评测协议。"
category: "rag"
section: "检索增强生成"
tags: ["RAG", "检索评测", "nDCG"]
difficulty: "进阶"
order: 3
updated: "2026-11-23"
created: "2026-11-23"
draft: false
featured: true
readingTime: "约 12 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 一个最终回答的错误，可能来自三处

用户问题进入 RAG 系统后，首先被转换成查询；系统找出候选文档并排序；最后语言模型阅读证据生成答案。因此，“回答不准确”至少可能由三种机制产生：关键文档没被召回、正确文档排序太低、模型没有正确使用已召回的证据。

只评价最终回答，会把三个误差源混在一起。更可操作的方法是建立检索层、生成层和端到端层三个评测视角。

## 构建证据标注的数据集

每个测试样本至少包含 query、允许访问的文档集合、相关证据的标识，以及可核验的目标答案。对于多跳问题，应标注所有必要证据，而不只是某一个高相关文档。

注意两个边界：一是评测文档版本必须固定；二是训练集中的问答对与最终测试问题不应大量重叠。否则召回指标可能主要反映对已有样本的适配。

## Recall@k：正确证据是否进入候选

对于一个查询，相关文档集合为 R，前 k 条检索结果为 T_k：

$$
\mathrm{Recall@k}=\frac{|R\cap T_k|}{|R|}.
$$

普通写法：Recall@k = number of relevant retrieved documents / number of relevant documents。

Recall@k 关心覆盖率，不关心相关结果的内部排序。适合衡量候选阶段有没有漏掉关键证据。在某些只要求找到至少一个证据的问题中，也可以报告 Hit@k，但不能将其混同于多证据召回率。

## MRR 与 nDCG：正确证据排在何处

MRR 取首个相关结果排名的倒数，再对全部查询求均值：

$$
\mathrm{MRR}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\operatorname{rank}_i}.
$$

普通写法：MRR = mean(1 / rank of first relevant document)。

如果存在多级相关性标签，nDCG 可以通过折损累计增益评估整个列表：

$$
\mathrm{DCG@k}=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)},\quad
\mathrm{nDCG@k}=\frac{\mathrm{DCG@k}}{\mathrm{IDCG@k}}.
$$

普通写法：nDCG@k = discounted relevance gain / ideal discounted relevance gain。

MRR 更强调第一个正确结果；nDCG 可以体现多个相关证据及其等级。指标应按问答任务真正依赖的证据数量选择。

## 生成层评测：正确性不等于忠实度

答案正确性（answer correctness）与证据忠实度（faithfulness）是两个维度。模型可能给出正确答案但没有使用当前检索证据，也可能忠实总结了错误或过时的文档。

对关键事实，可以将生成答案拆成原子陈述，分别判断是否能被检索到的证据支持，再汇总为一致性或忠实度指标。自动评判结果应使用人工标注样本校准，避免把另一个模型的主观判断当作金标准。

引用评估要检查：引用是否存在、是否指向正确版本、引用段落能否直接支持相应陈述。仅在答案后面附加几个文档链接，不代表具备可追溯性。

## 端到端排错顺序

当某个问题失败时，可以按下面的次序调试：

1. 检查原始文档解析和版本是否正确；若缺少关键段落，先修知识库。
2. 查看候选召回能否找到标注的证据；若找不到，改查询、索引或召回算法。
3. 查看相关结果的实际排名；若排名靠后，检查 reranker 和混合检索策略。
4. 确认正确证据进入上下文后，检验模型能否基于证据给出正确回答。
5. 比较 token 成本、延迟和准确率，选择业务允许的 Pareto 折中。

## 评测实验矩阵

最小消融可设置：BM25-only、Dense-only、Hybrid、Hybrid+Rerank 四组，再跨多个 k 值比较。除整体平均值外，按实体检索、时间问题、长文档、多跳问答和无法回答的问题分别报告结果。

对于需要严格权限控制的系统，还应验证检索结果的访问权限过滤。未授权内容即使只进入模型上下文而没有显示给用户，也属于需要排查的错误。

## 一个小型排序评测手算

假设有两个查询，且每个查询都只有一条标注的最关键证据。查询 A 的正确证据排第 2 位，查询 B 排第 4 位。则这两条查询的 Reciprocal Rank 分别为 1/2 和 1/4，平均 MRR 为：

$$
\mathrm{MRR}=\frac{1/2+1/4}{2}=0.375.
$$

普通写法：MRR = (1/2 + 1/4) / 2 = 0.375。

如果计算 Recall@3，则查询 A 命中、查询 B 未命中，宏平均 Recall@3 为 0.5。两者给出不同信息：MRR 能区分正确证据出现在第 2 位和第 4 位，Recall@3 只判断它是否进入前三位。

## 怎样保存可复核的逐查询报告

推荐为每个查询保留 query_id、ground_truth_document_ids、retrieved_document_ids、rerank_score、模型生成答案、引用的 Document ID 和最终判定。这样某个平均指标变化时，可以直接定位改变最大的查询。

~~~text
query_id: q_014
gold_docs: [doc_03]
retrieved_top5: [doc_07, doc_03, doc_11, doc_02, doc_09]
recall_at_3: 1.0
reciprocal_rank: 0.5
answer_correct: true
citation_supported: true
~~~

如果评测表只保留总体准确率，就无法回答改进是来自更多证据被召回，还是模型在相同证据下生成得更好。

## 自动评判也需要校准

用另一个 LLM 判断答案是否忠实于证据，可以减少部分人工成本，但它本身也会误判。建议选一批人工认真标注的样本，先评估自动评判器的准确率与一致性，再将其应用到更大评测集。

此外，检索器、生成器和评判器若共享同一数据来源或模型家族，可能存在相关偏差。最终报告应说明哪些结果来自客观匹配，哪些依赖模型评判，并保留可供人工核对的样本。


## 推荐文献

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)。
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)。
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)。
