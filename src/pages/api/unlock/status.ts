export const prerender = false;
import type { APIRoute } from 'astro';
export const GET: APIRoute = async ({ cookies }) => {
  return new Response(JSON.stringify({ unlocked: cookies.get('llmnotes_unlock')?.value === 'true' }), { headers: { 'content-type': 'application/json' } });
};
