// Vercel Serverless Function (Node.js 18) that proxies requests to a Python backend
// Set BACKEND_URL in your Vercel Project Settings, e.g. https://your-backend.onrender.com

export default async function handler(req, res) {
  if (req.method === 'GET') {
    return res.status(200).json({ ok: true, message: 'Proxy is running. POST an image file to /api/predict.' });
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const backend = process.env.BACKEND_URL;
  if (!backend) {
    return res.status(500).json({ error: 'BACKEND_URL is not set in Vercel env vars.' });
  }

  const target = backend.replace(/\/$/, '') + '/predict';

  try {
    // Buffer the incoming request body (supports multipart/form-data)
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = Buffer.concat(chunks);

    // Forward original headers except host/connection-related ones
    const headers = Object.fromEntries(
      Object.entries(req.headers).filter(([k]) => !['host', 'connection', 'content-length'].includes(k.toLowerCase()))
    );

    const upstream = await fetch(target, {
      method: 'POST',
      headers,
      body
    });

    const buf = Buffer.from(await upstream.arrayBuffer());
    // Mirror status and content-type from upstream
    res.status(upstream.status);
    const ct = upstream.headers.get('content-type') || 'application/octet-stream';
    res.setHeader('content-type', ct);
    res.send(buf);
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: 'Upstream error', detail: err?.message || String(err) });
  }
}
