---
title: "ReAct 框架"
description: "解释 ReAct 如何把推理过程和工具行动组合成可迭代的智能体流程。"
category: "agent"
section: "Agent"
tags: ["Agent","ReAct","Tool Calling"]
difficulty: "入门"
order: 4
updated: "2026-07-16"
created: "2026-05-29"
draft: false
featured: false
cover: ""
readingTime: "约 7 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 本文解决什么问题

如果只让模型一次性给答案，它遇到不确定信息时容易猜；如果只让模型调用工具，又容易变成没有判断力的流程编排。ReAct 的核心思路是把推理和行动交替起来：先想清楚下一步，再调用工具，根据观察结果继续调整。

## 核心结论

ReAct 是 Reasoning and Acting 的组合。它让模型在“思考、行动、观察”之间循环，适合需要查资料、调用工具、逐步排查的问题。

一个简化流程是：

```txt
Thought：我需要知道当前仓库有哪些失败测试。
Action：运行测试命令。
Observation：某个测试因为空值处理失败。
Thought：需要查看对应函数。
Action：读取源文件。
Observation：函数没有处理 null 输入。
Answer：给出修复建议或修改代码。
```

ReAct 的价值不是把思考过程展示得更长，而是让工具使用更有目的。

## 概念解释

ReAct 解决的是两个问题：模型缺少外部信息，工具调用缺少推理控制。通过交替循环，模型可以先提出假设，再用工具验证，最后根据结果修正假设。

在真实系统里，不一定要把 `Thought` 原样展示给用户。很多产品会把内部推理隐藏起来，只保留工具调用记录、引用来源和最终答案。这样既能保留可审计性，也能避免暴露不必要的中间内容。

## 工程直觉

ReAct 系统的质量取决于工具定义是否清楚。工具名称、参数、返回格式、错误码和使用边界越明确，模型越不容易乱调。反过来，如果工具描述模糊，模型会把调用工具当成碰运气。

另一个关键是限制循环。最多调用几次工具？连续失败几次后停止？哪些操作需要用户确认？哪些结果必须二次验证？这些规则要写在控制器里，而不是完全交给模型自由发挥。

## 伪代码

```txt
while not done:
  model reads goal + history
  model chooses next action or final answer
  if action:
    validate tool name and arguments
    run tool
    append observation to history
  else:
    return answer
```

这个循环的难点不在代码本身，而在工具边界、状态管理和失败处理。

## 常见误区

第一，把 ReAct 当成固定 prompt 模板。它更像一种控制思想，具体实现可以是函数调用、工作流节点或代码里的状态机。

第二，让模型暴露全部思考文本。生产系统更应该展示证据、步骤摘要和工具结果，而不是完整内部推理。

第三，忽略工具返回质量。Observation 如果太长、太乱或没有结构，下一步推理就会被噪声带偏。

## 面试追问

1. ReAct 中 Reasoning 和 Acting 分别解决什么问题？
2. ReAct 和普通 tool calling 有什么区别？
3. 为什么 ReAct 需要观察结果再决定下一步？
4. 如何避免 Agent 工具调用陷入死循环？
5. 生产系统中应不应该展示模型的完整 Thought？

## 推荐阅读

建议和 Agent 总览一起读。先理解 Agent 的组件，再看 ReAct 循环，会更清楚它在系统里的位置。
