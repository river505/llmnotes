import { glob } from 'astro/loaders';
import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';

const difficulty = z.enum(['入门', '进阶', '高级']);

const articleSchema = z.object({
  title: z.string(),
  description: z.string(),
  category: z.string(),
  section: z.string().default(''),
  tags: z.array(z.string()).default([]),
  difficulty: difficulty.default('入门'),
  order: z.number().default(1),
  updated: z.coerce.date(),
  created: z.coerce.date(),
  draft: z.boolean().default(false),
  featured: z.boolean().default(false),
  cover: z.string().optional().default(''),
  readingTime: z.string().optional().default(''),
  unlockRequired: z.boolean().default(true),
  previewRatio: z.number().min(0.1).max(1).default(0.35),
});

export const collections = {
  docs: defineCollection({ loader: glob({ base: './src/content/docs', pattern: '**/*.{md,mdx}' }), schema: articleSchema }),
  interview: defineCollection({ loader: glob({ base: './src/content/interview', pattern: '**/*.{md,mdx}' }), schema: articleSchema }),
  projects: defineCollection({ loader: glob({ base: './src/content/projects', pattern: '**/*.{md,mdx}' }), schema: articleSchema }),
  papers: defineCollection({ loader: glob({ base: './src/content/papers', pattern: '**/*.{md,mdx}' }), schema: articleSchema }),
  roadmaps: defineCollection({ loader: glob({ base: './src/content/roadmaps', pattern: '**/*.{md,mdx}' }), schema: articleSchema }),
};
