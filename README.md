# LLM Notes

大模型学习与工程实践指南，面向 AI 研究生、算法工程师、开发者和面试准备者，系统梳理 LLM、RAG、Agent、多模态、强化学习、AI 面试、项目实战与论文精读。

## 本地开发

```bash
npm install
npm run dev
```

## 构建

```bash
npm run build
```

构建会先运行 Astro，再用 Pagefind 为 `dist` 生成静态搜索索引。

## 预览

```bash
npm run preview
```

## 环境变量

```txt
PUBLIC_TURNSTILE_SITE_KEY=
TURNSTILE_SECRET_KEY=
UNLOCK_COOKIE_SECRET=
```

`PUBLIC_TURNSTILE_SITE_KEY` 可暴露给前端；`TURNSTILE_SECRET_KEY` 只能在服务端使用。不要提交 `.env`。

## 部署

推荐通过 Cloudflare GitHub 集成自动部署，避免和本地 `npm run deploy` 混用。
