export const prerender = false;
import type { APIRoute } from 'astro';
export const POST: APIRoute = async ({ request, cookies, locals }) => {
  const body = (await request.json().catch(() => ({}))) as { token?: string };
  const token = body.token;
  if (!token) return Response.json({ ok: false, message: '缺少验证 token' }, { status: 400 });
  const env = (locals as any).runtime?.env ?? import.meta.env;
  const secret = env.TURNSTILE_SECRET_KEY;
  if (!secret && import.meta.env.DEV) {
    cookies.set('llmnotes_unlock', 'true', { httpOnly: true, secure: false, sameSite: 'lax', maxAge: 43200, path: '/' });
    return Response.json({ ok: true });
  }
  if (!secret) return Response.json({ ok: false, message: '服务端未配置 Turnstile Secret' }, { status: 500 });
  const form = new FormData();
  form.append('secret', secret);
  form.append('response', token);
  const verify = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', { method: 'POST', body: form });
  const data = await verify.json() as { success?: boolean };
  if (!data.success) return Response.json({ ok: false, message: '验证失败' }, { status: 403 });
  cookies.set('llmnotes_unlock', 'true', { httpOnly: true, secure: true, sameSite: 'lax', maxAge: 43200, path: '/' });
  return Response.json({ ok: true });
};
