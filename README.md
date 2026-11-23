# LLMNotes

网站：**https://llmnotes.cn** · 仓库：**https://github.com/river505/llmnotes**

面向机器学习与大模型技术阅读的结构化文档站。以问题和计算机制组织 Transformer、训练与对齐、RAG、Agent、多模态、推理服务、原始论文与工程评估。本站采用独立的白色/蓝灰阅读界面，与 River Journal 技术博客保持视觉区分。

## 内容与维护

内容保存在 src/content 下，按 docs、papers、projects、roadmaps 和 interview 分类。每篇 Markdown 文件使用 YAML Frontmatter；技术文档的左侧目录由 src/config/sidebar.ts 维护。

增加和修改文章请直接编辑 GitHub 仓库文件；不需要每次通过聊天修改。关于篇幅、案例、公式、引用、审核及发布流程，见 **CONTENT_GUIDE.md**。

## 本地开发与构建

~~~bash
npm ci
npm run dev
npm run build
npm run preview
~~~

构建使用 Astro + Pagefind，并通过 Cloudflare Workers 部署。main 分支已连接 Cloudflare Workers Builds，提交后自动触发生产构建与部署；不要同时使用其它部署渠道写入同一 Worker。

## 配置与安全

~~~text
PUBLIC_TURNSTILE_SITE_KEY=
TURNSTILE_SECRET_KEY=
UNLOCK_COOKIE_SECRET=
~~~

只允许前端暴露 Public Site Key；其它密钥保存在 Cloudflare 绑定或本地安全环境中，不能加入 Git。站点源码仓库是公开的，draft: true 不意味着文件在 GitHub 上私密。
