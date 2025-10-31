# Deploying to Vercel (Frontend + Proxy)

Vercel is great for hosting static frontends and lightweight serverless functions, but it is not suitable for running large Python ML stacks like PyTorch (package size and runtime limits). The recommended pattern is:

- Host the frontend (this repo) on Vercel.
- Host the Python inference API (FastAPI/Flask) on a Python-friendly host like Render, Railway, Fly.io, or Hugging Face Spaces.
- Use a Vercel function (`/api/predict`) as a small proxy to your backend. This keeps your API URL private and simplifies CORS.

## What was added

- `vercel.json` — Vercel config for Node.js serverless functions.
- `.vercelignore` — prevents large datasets and model weights from being uploaded.
- `public/index.html` — simple upload UI that calls `/api/predict`.
- `api/predict.js` — proxies requests to your real Python backend (set `BACKEND_URL`).
- `backend_skeleton/` — optional FastAPI starter you can deploy to Render/Railway.

## Quick start

1) Push this repo to GitHub (or GitLab/Bitbucket).

2) Create a Python backend:

   - Option A (FastAPI on Render):
     - `cd backend_skeleton`
     - Create a new repo for it and push.
     - On Render.com, create a new Web Service from that repo.
     - Start command: `uvicorn main:app --host 0.0.0.0 --port 8000`
     - Set a suitable build command (Render auto-installs from `requirements.txt`).
     - Add your real model loading/inference.

   - Option B (Hugging Face Spaces):
     - Convert your app to Gradio/FastAPI and deploy as a Space.

   - Option C (Streamlit Community Cloud):
     - If your app is Streamlit (`app.py`), deploy directly to Streamlit Cloud.

3) On Vercel:

   - Import this project from GitHub.
   - In Project Settings → Environment Variables, add:
     - `BACKEND_URL` = `https://your-backend.onrender.com` (no trailing slash)
   - Deploy.

4) Visit your Vercel URL. Upload an image; the request will be proxied to your backend `/predict` endpoint, and the result rendered as JSON.

## Notes

- `.vercelignore` ensures large files like `*.pth` and `AxialDataset/` are not uploaded. Keep model weights on your backend host.
- If you want a richer UI, you can replace `public/index.html` with a Next.js app.
- For CORS: The proxy avoids browser CORS preflights hitting your backend directly. If you prefer direct calls from the browser, configure CORS in your backend and call it from the frontend without the proxy.

## Windows PowerShell helpers

Initialize a new git repo and push:

```powershell
# From the project root
git init
git add .
git commit -m "chore: scaffold Vercel frontend + proxy"
# Create a GitHub repo first, then:
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

Deploy on Vercel (optional CLI):

```powershell
npm i -g vercel
vercel --prod
```

Configure env var via CLI:

```powershell
vercel env add BACKEND_URL production
# Then paste your backend URL when prompted
```
