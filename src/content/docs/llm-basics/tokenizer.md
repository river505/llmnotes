---
title: "Token、Tokenizer 与词表"
description: "理解 token、分词器和词表如何影响上下文长度、成本与模型效果。"
category: "llm-basics"
section: "大模型基础"
tags: ["Token","Tokenizer","LLM"]
difficulty: "入门"
order: 2
updated: "2026-07-16"
created: "2026-05-29"
draft: false
featured: false
cover: ""
readingTime: "约 6 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 本文解决什么问题

你在大模型文档里经常会看到“上下文 128K”“输入按 token 计费”“中文一个字不一定等于一个 token”。这些说法都绕不开 Tokenizer。理解它，能帮你判断提示词为什么超长、成本怎么算、为什么同一段文本在不同模型里长度不一样。

## 核心结论

Token 是模型真正处理的离散单位，Tokenizer 是把文本切成 token 并映射为整数 ID 的工具，词表就是这些 token 到 ID 的表。模型看不到“字”和“词”的自然边界，它看到的是一串数字 ID，再把这些 ID 转成向量参与计算。

同一句话，用不同 Tokenizer 切分，token 数可能不同。这会影响上下文容量、推理成本、长文召回效果和生成速度。

## 概念解释

现代 LLM 常用 BPE、WordPiece、SentencePiece 等子词切分方法。它们不会简单按空格或汉字切，而是根据训练语料里经常共同出现的字符片段来合并。高频词可能成为一个 token，低频词会被拆成多个片段，代码里的符号、缩进和变量名也会被切分。

例如，“Transformer”在某些词表里可能是一个 token，在另一些词表里可能被拆成 `Transform` 和 `er`。中文也类似，常见词可能整体保留，不常见组合则会拆得更细。

## 数学/工程直觉

Tokenizer 的目标是在词表大小和表达效率之间折中。词表太小，文本会被切得很碎，序列变长，计算成本上升；词表太大，嵌入矩阵变大，低频 token 学不好，还会浪费参数。

工程上，token 数决定了很多现实限制：一次请求能放多少材料、RAG 每个 chunk 应该多长、历史对话要不要压缩、批量推理能塞多少样本，以及最终账单大概是多少。

## 一个小例子

```txt
原文：我想学习 tokenizer 的原理。
可能的切分：我 / 想 / 学习 / token / izer / 的 / 原理 / 。
送入模型：对应的一串整数 ID。
模型输出：下一批 token ID，再被解码成文本。
```

注意这里的切分只是示意，不同模型的实际结果不一定一样。做成本估算或上下文裁剪时，应使用目标模型对应的 tokenizer。

## 常见误区

第一，把字符数当 token 数。英文、中文、代码、emoji、URL 的 token 密度差别很大，不能用固定比例精确估算。

第二，忽略特殊 token。系统消息、角色标记、工具调用结构、停止符都可能占 token，并影响模型对边界的理解。

第三，在 RAG 中只按字符切文档。更稳妥的做法是按 token 长度控制 chunk，同时保留标题、页码、段落层级等元数据。

## 面试追问

1. Token、Tokenizer、词表三者分别是什么？
2. 为什么子词切分比按词切分更适合开放词表场景？
3. Tokenizer 会如何影响上下文长度和推理成本？
4. 为什么代码和 URL 往往 token 密度更高？
5. 做长文 RAG 时，为什么需要按 token 控制 chunk？

## 推荐阅读

读完本文后，可以接着看 Attention 的 4 token 示例。理解 token 如何进入模型，再看 Q、K、V 和注意力矩阵，会顺很多。
