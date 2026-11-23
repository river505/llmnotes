export const sidebar = [
  { title:'基础与训练',slug:'llm-basics',items:[
    {title:'基础模型与 LLM',slug:'/docs/llm-basics/what-is-llm'},
    {title:'Token 与分词系统',slug:'/docs/llm-basics/tokenizer'},
    {title:'预训练、SFT 与偏好优化',slug:'/docs/llm-basics/training-alignment'}]},
  { title:'Transformer 与注意力',slug:'transformer',items:[
    {title:'四个 Token 推导 Attention',slug:'/docs/transformer/attention-four-tokens'},
    {title:'缩放因子 √dₖ 的由来',slug:'/docs/transformer/why-divide-sqrt-dk'},
    {title:'KV Cache 的显存与延迟',slug:'/docs/transformer/kv-cache'},
    {title:'GQA 与 MLA 的设计取舍',slug:'/docs/transformer/gqa-mla'}]},
  { title:'检索增强生成',slug:'rag',items:[
    {title:'RAG 的系统链路',slug:'/docs/rag/what-is-rag'},
    {title:'文档解析与切分',slug:'/docs/rag/chunking'},
    {title:'检索与生成的分层评测',slug:'/docs/rag/evaluation'}]},
  { title:'Agent 系统',slug:'agent',items:[
    {title:'Agent 的状态与执行器',slug:'/docs/agent/what-is-agent'},
    {title:'ReAct 循环',slug:'/docs/agent/react'},
    {title:'工具调用与状态机',slug:'/docs/agent/tool-execution'}]},
  { title:'多模态学习',slug:'multimodal',items:[{title:'视觉与语言的融合',slug:'/docs/multimodal/what-is-multimodal'}]},
  { title:'强化学习',slug:'rl',items:[{title:'MDP 与策略',slug:'/docs/rl/what-is-rl'},{title:'PPO 裁剪目标',slug:'/docs/rl/ppo'}]},
  { title:'推理工程',slug:'engineering',items:[{title:'推理服务与性能',slug:'/docs/engineering/inference-serving'}]},
  { title:'推荐系统',slug:'recsys',items:[{title:'召回、排序与重排',slug:'/docs/recsys/recsys-pipeline'}]}
];