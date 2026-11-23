export type NavItem = { label: string; href: string };
export const navItems: NavItem[] = [
  { label:'首页',href:'/' },{ label:'技术文档',href:'/docs' },{ label:'学习路线',href:'/roadmaps' },
  { label:'论文研读',href:'/papers' },{ label:'项目实践',href:'/projects' },{ label:'技术面试',href:'/interview' }
];
export const categories = [
  {title:'模型结构与注意力',slug:'transformer',href:'/docs/transformer/attention-four-tokens',description:'从 Q/K/V 到 KV Cache、GQA 与长上下文计算。'},
  {title:'训练、微调与对齐',slug:'llm-basics',href:'/docs/llm-basics/training-alignment',description:'预训练目标、SFT、偏好优化与参数高效适配。'},
  {title:'检索与 RAG',slug:'rag',href:'/docs/rag/what-is-rag',description:'解析、混合检索、重排、引用与分层评测。'},
  {title:'Agent 与工具执行',slug:'agent',href:'/docs/agent/what-is-agent',description:'ReAct、工具调用、状态管理和执行验证。'},
  {title:'多模态表示',slug:'multimodal',href:'/docs/multimodal/what-is-multimodal',description:'跨模态对齐、视觉编码、特征融合与评测。'},
  {title:'强化学习与推理工程',slug:'rl',href:'/docs/rl/ppo',description:'策略优化、模型对齐、推理性能与部署成本。'}
];
export const roadmapCards = [
  {title:'从基础到 Transformer',href:'/roadmaps/llm-beginner',description:'由张量计算进入注意力、训练和原始论文。'},
  {title:'大模型应用工程',href:'/roadmaps/llm-engineering',description:'RAG、Agent、评测、服务部署的完整系统链路。'},
  {title:'算法研究与论文阅读',href:'/roadmaps/research-methods',description:'实验假设、消融设计、分组验证与论文证据。'},
  {title:'大模型技术面试',href:'/interview/llm-interview',description:'机制、复杂度、系统取舍与项目验证。'}
];
