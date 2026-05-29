export const prerender = false;

import type { APIRoute } from 'astro';

export const GET: APIRoute = async ({ locals }) => {
  const env = (locals as any).runtime?.env ?? import.meta.env;
  return Response.json({
    siteKey: env.PUBLIC_TURNSTILE_SITE_KEY ?? '',
  });
};
