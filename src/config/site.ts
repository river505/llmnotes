export type NavItem = { label: string; href: string };

export const navItems: NavItem[] = [
  { label: '首页', href: '/' },
  { label: '学习路线', href: '/roadmaps' },
  { label: '文档', href: '/docs' },
  { label: '面试', href: '/interview' },
  { label: '项目实战', href: '/projects' },
  { label: '论文精读', href: '/papers' },
  { label: '资源', href: '/resources' },
  { label: '联系', href: '/contact' },
  { label: '关于', href: '/about' },
];

export const categories = [
  { title: '大模型基础', slug: 'llm-basics', href: '/docs/llm-basics/what-is-llm', description: 'Token、Embedding、Attention、Transformer、对齐训练与推理机制。' },
  { title: 'Transformer 专题', slug: 'transformer', href: '/docs/transformer/attention-four-tokens', description: '从 4 个 token 到完整 GPT 架构，拆透 Attention。' },
  { title: 'RAG', slug: 'rag', href: '/docs/rag/what-is-rag', description: '检索增强生成、文档切分、向量库、Rerank 与工程优化。' },
  { title: 'Agent', slug: 'agent', href: '/docs/agent/what-is-agent', description: 'ReAct、工具调用、记忆、规划、多智能体和 Code Agent。' },
  { title: '多模态', slug: 'multimodal', href: '/docs/multimodal/what-is-multimodal', description: 'CLIP、LLaVA、视觉 Token、多模态 RAG 与评测。' },
  { title: '强化学习', slug: 'rl', href: '/docs/rl/what-is-rl', description: 'MDP、Policy Gradient、PPO、RLHF、DPO 与 GRPO。' },
  { title: '推荐系统', slug: 'recsys', href: '/docs/recsys/recsys-pipeline', description: '召回、粗排、精排、重排、双塔与推荐系统面试。' },
  { title: 'AI 工程部署', slug: 'engineering', href: '/docs', description: '流式输出、推理加速、量化、vLLM、Cloudflare Workers AI。' },
];

export const roadmapCards = [
  { title: '零基础大模型学习路线', href: '/roadmaps/llm-beginner', description: '从机器学习基础到 Transformer、RAG、Agent 和项目输出。' },
  { title: 'AI 算法工程师路线', href: '/roadmaps/llm-beginner', description: '面向算法岗位的模型原理、评测、工程和面试路线。' },
  { title: '大模型应用开发路线', href: '/roadmaps/llm-beginner', description: 'API、RAG、Agent、部署与成本优化的工程路径。' },
  { title: 'AI 面试 30 天突击路线', href: '/interview/llm-interview', description: '高频题、项目追问和论文复述的集中训练。' },
];
