const tagSlugMap: Record<string, string> = {
  '工具调用': 'tool-calling',
  '规划': 'planning',
  '多模态': 'multimodal',
  '知识库': 'knowledge-base',
  '向量检索': 'vector-search',
  '工程实践': 'engineering-practice',
  '基础模型': 'foundation-model',
  '面试': 'interview',
  '推荐系统': 'recsys',
  '召回': 'retrieval',
  '排序': 'ranking',
  '强化学习': 'reinforcement-learning',
  '策略': 'policy',
  '数学直觉': 'math-intuition',
  '项目': 'project',
  '论文精读': 'paper-reading',
  '学习路线': 'roadmap',
  '入门': 'beginner',
};

export function tagSlug(tag: string) {
  return tagSlugMap[tag] ?? tag.toLowerCase().replace(/&/g, 'and').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}
