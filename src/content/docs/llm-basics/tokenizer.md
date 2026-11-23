---
title: "Token 与 Tokenizer：模型究竟接收什么"
description: "从文本编码、子词词表、BPE 合并策略与上下文预算理解 Tokenizer 的工程影响。"
category: "llm-basics"
section: "大模型基础"
tags: ["Token","Tokenizer","LLM"]
difficulty: "入门"
order: 2
updated: "2026-11-23"
created: "2026-05-29"
draft: false
featured: false
cover: ""
readingTime: "约 12 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 语言模型的输入不是字符串

用户看到的是自然语言字符串，模型实际处理的是离散 Token ID 序列。Tokenizer 完成从文本到 Token 序列的转换；Embedding 层再把 Token ID 转成连续向量，供后续网络计算。

分词并不是简单地按汉字、英文单词或空格切割。一个词可能拆成多个子词，一段数字、代码或包含特殊符号的文本，也可能产生不符合直观词语边界的 Token 序列。

## 词表与 Token ID 的关系

词表 V 包含分词器允许输出的单位集合。Tokenizer 将输入字符串映射为 Token ID 序列：

$$
T(x)=(t_1,t_2,\ldots,t_n),\qquad t_i\in\{0,\ldots,|V|-1\}.
$$

普通写法：tokenizer(text) -> [token_id_1, token_id_2, ...]，每个 ID 对应词表中的一个条目。

这些整数不是向量表示。只有经过 Embedding 查表后，才得到形如 X∈ℝ^(n×d_model) 的连续输入。对训练而言，目标 Token 同样以 ID 表示，损失函数通常计算其在词表分布中的对数概率。

## 为什么需要子词分解

若把所有完整词语都作为词表单位，开放词汇、稀有组合、拼写变化和新词会让词表无限膨胀。若每个字符或字节都独立建模，序列长度又可能显著增加。

子词方法尝试在词表规模与序列长度之间取得平衡。高频片段可以拥有独立 Token，低频词语则被拆成若干片段，因此兼顾组合覆盖与压缩效果。

常见路线包括 BPE、WordPiece、Unigram 等。不同训练目标与归一化策略会形成不同词表，并影响后续模型的序列长度分布。

## BPE 的合并过程

以字符或字节单元作为初始序列，BPE 统计相邻单元对的频率，并逐步把高频对合并为新单位。简化的训练过程可以理解为：

1. 从语料的初始符号序列开始，统计相邻 Token 对。
2. 选择当前高频的相邻对，建立一个合并规则。
3. 将对应符号对合并，更新语料表示与统计频率。
4. 重复直到达到目标词表规模或停止条件。

真实 Tokenizer 的预处理、空格处理与 Byte-level 编码更复杂。尤其当输入包含中文、Emoji、代码缩进时，字节层实现与 Unicode 规范化会影响最终结果。

## Token 数与成本不是字符数的简单换算

推理上下文通常以 Token 数限制长度。假设最大允许上下文长度为 L，输入已占 n_in，模型计划最多生成 n_out，则需要：

$$
n_{\mathrm{in}}+n_{\mathrm{out}}\le L.
$$

普通写法：input_tokens + maximum_output_tokens <= context_window。

对于包含工具结果、系统指令和多轮历史的应用，还必须把这些内容计入输入 Token。大量重复检索片段不仅提高费用，也会占用可供模型推断的上下文预算。

中文、英文、代码和表格的 Token/字符比例可能不同，不能使用同一个固定换算系数。实际容量规划应调用与目标模型一致的 Tokenizer 进行统计。

## 特殊 Token 与 Chat Template

聊天模型往往使用特殊 Token 区分对话角色、轮次结束、工具调用边界等。它们可能不是普通字符串中可见的字面内容，而是由模板和 Tokenizer 映射到专门的 ID。

当微调和部署使用不同 Chat Template，训练输入与推理输入就可能不一致。常见症状包括角色混淆、输出格式异常和模型无法按预期调用工具。

因此，训练数据预处理、推理服务与评测脚本最好复用同一套 Tokenizer 和模板定义。

## 分词器如何影响模型效果

词表不仅影响长度，也影响某些任务的表示难度。代码标识符、专有名词、数学符号以及多语言混合文本可能产生复杂的切分模式。更短的 Token 序列不必然意味着任务准确率更高，但会改变上下文承载量和计算成本。

在多语言系统中，建议至少比较各语言上的平均 Token 数、长尾分词长度和截断比例。对于代码生成，还应检查缩进、换行和特殊标记的处理是否稳定。

## 一个可执行的检查方案

选取涵盖中英文、代码、长数字、URL、Markdown 和 Emoji 的测试集合，记录原始字符数、Token 数、是否可逆解码以及不同输入长度的截断情况。

更换 Tokenizer 或模型版本后重复统计，以判断上下文预算和成本是否发生结构性变化。对于有训练需求的项目，还需要关注特殊 Token 是否出现在预期的位置。

## 用一个玩具语料推导 BPE

假设初始符号按字符划分，语料中频繁出现 apple 和 apply。它们共享前缀 app，但末尾字符不同。BPE 统计最常见的相邻符号组合，先后可能产生 a+p→ap、ap+p→app 这样的合并规则。

应用这些规则后，apple 可以被拆成 app + le，apply 可以被拆成 app + ly。这个例子展示了子词复用如何减少长尾完整词的数量；具体合并顺序仍由语料频率和预处理方式决定。

| 分词方案 | apple | apply | 特点 |
|---|---|---|---|
| 字符级 | a p p l e | a p p l y | 单位简单，但序列较长 |
| 子词级（示意） | app le | app ly | 高频片段可以复用 |
| 整词级 | apple | apply | 需要更大的完整词词表 |

不要把表中的子词方案视为任何特定 Tokenizer 的真实输出。不同模型的 Tokenizer、语言预处理和词表容量会得到不同的结果。

## 怎样检查截断是否破坏任务

将训练或推理输入交给目标模型的 Tokenizer，统计 Token 长度分布。特别检查 P50/P95/P99、最大长度、角色模板额外消耗以及发生截断的样本比例。

如果截断删除的是系统约束、代码函数尾部或检索证据中的关键结论，模型即便剩余上下文长度足够，最终表现也可能下降。与其盲目增大上下文窗口，更稳妥的是先确认哪些信息被丢弃。


## 推荐阅读

- [基础模型与 LLM](/docs/llm-basics/what-is-llm)
- [从四个 Token 推导 Self-Attention](/docs/transformer/attention-four-tokens)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
