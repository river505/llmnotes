export const sidebar = [
  { title: '大模型基础', slug: 'llm-basics', items: [
    { title: '什么是大模型、基础模型与 LLM', slug: '/docs/llm-basics/what-is-llm' },
    { title: 'Token、Tokenizer 与词表', slug: '/docs/llm-basics/tokenizer' },
  ]},
  { title: 'Transformer 专题', slug: 'transformer', items: [
    { title: '用 4 个 Token 看懂 Attention', slug: '/docs/transformer/attention-four-tokens' },
    { title: '为什么要除以 sqrt(d_k)', slug: '/docs/transformer/why-divide-sqrt-dk' },
  ]},
  { title: 'RAG', slug: 'rag', items: [
    { title: 'RAG 是什么', slug: '/docs/rag/what-is-rag' },
    { title: '文档解析与切分 Chunking', slug: '/docs/rag/chunking' },
  ]},
  { title: 'Agent', slug: 'agent', items: [
    { title: 'Agent 是什么', slug: '/docs/agent/what-is-agent' },
    { title: 'ReAct 框架', slug: '/docs/agent/react' },
  ]},
  { title: '多模态大模型', slug: 'multimodal', items: [
    { title: '多模态大模型是什么', slug: '/docs/multimodal/what-is-multimodal' },
  ]},
  { title: '强化学习', slug: 'rl', items: [
    { title: '强化学习解决什么问题', slug: '/docs/rl/what-is-rl' },
    { title: 'PPO 原理', slug: '/docs/rl/ppo' },
  ]},
  { title: '推荐系统', slug: 'recsys', items: [
    { title: '推荐系统整体链路', slug: '/docs/recsys/recsys-pipeline' },
  ]},
];
