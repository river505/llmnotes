---
title: "Agent 工具执行：从提示词循环到受控状态机"
description: "将任务拆成状态、工具契约、观察与终止条件，讨论工具调用的可靠性和评测。"
category: "agent"
section: "Agent 系统"
tags: ["Agent", "工具调用", "状态机"]
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

## Agent 的关键能力是执行，而不是连续对话

普通聊天模型一次生成文本便结束；工具型 Agent 则需要在多轮决策中读取外部状态、调用函数、解释返回结果并选择下一步。系统是否可靠，取决于每一步是否有明确定义的状态迁移、输入输出契约和终止条件。

把模型看作决策器，把工具看作外部环境接口，把执行器看作受控状态机，比把 Agent 描述为“会自己思考的 AI”更准确。

## 一个最小状态机

设当前状态包含任务目标、历史工具观察、剩余调用额度、未完成子目标以及已授权能力。模型产生动作 a_t，工具返回观察 o_t，再转移到下一个状态。

$$
s_{t+1}=F(s_t,a_t,o_t).
$$

普通写法：next_state = transition(current_state, chosen_action, tool_observation)。

工程状态可以拆为：READY → PLAN → TOOL_CALL → OBSERVE → VERIFY → DONE / FAILED。每次状态切换都记录动作名称、参数、结果摘要和时延，便于回放。

## 工具契约应如何设计

一个可调用工具至少需要名称、参数 Schema、输入验证、返回结构、权限要求以及明确的失败语义。例如文件修改工具应要求路径、原内容匹配条件和替换内容，而不是允许执行任意系统命令。

推荐区分只读工具和写入工具。只读操作适合在更宽的权限范围自动执行；修改文件、发消息或部署生产服务等状态变更，应具备更严格的授权和确认。

## 观察与错误恢复

工具结果不是模型的系统指令，而是外部数据。执行器应将观察结果和控制指令区分处理，避免从网页、日志或仓库注释中直接接受“忽略之前规则”之类的文本。

失败时应明确区分：参数校验失败、权限不足、暂时性网络错误、业务条件不满足和执行超时。不同失败类型对应不同重试策略；不应无限重复同一个调用。

## 一个可以落地的执行循环

~~~text
state = initialize(task, allowed_tools, call_budget)
while state.status == RUNNING:
    proposal = model.decide(state.summary)
    action = validate(proposal, allowed_tools, budget)
    if action.requires_approval:
        action = get_authorized_approval(action)
    observation = execute(action)
    state = transition(state, action, observation)
    state = verify_progress(state)
return state.result
~~~

这段流程的核心是 verify_progress：模型说“已完成”不构成成功证据，必须使用外部可检查的条件，例如测试通过、生成文件存在、指定状态已更新。

## 评测体系

对代码任务可以记录任务成功率、测试通过率、实际修改范围、工具调用数量和成本。对信息检索任务，需要额外评估证据正确性、引用质量和结论可重复性。

系统级消融可比较：单次模型回答、固定流程、ReAct 式动态调用、加入检查点的状态机。固定同一批任务、工具权限和最大预算，才能解释架构差异的贡献。

## 相关论文与实践

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)。
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)。
- [Code Agent 项目实践](/projects/code-agent)。
