# Streamlit Community Cloud — Easiest free hosting for this app

Your `app.py` is a Streamlit app that loads a PyTorch model and predicts a tumor class from an uploaded image. Streamlit Community Cloud is the simplest free way to get this running publicly.

## 1) What you need

- A GitHub repo containing at least:
  - `app.py`
  - `brain_tumor_model.pth` (if ≤ 100 MB — otherwise, see "Large model" below)
  - `requirements.txt` (already added here)

## 2) Deploy steps

1. Push this folder to GitHub.

   ```powershell
   git init
   git add .
   git commit -m "deploy: streamlit app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```

2. Go to https://share.streamlit.io
   - Sign in with GitHub.
   - Deploy a new app and select your repo.
   - Set the main file path to `app.py`.

3. Wait for build → open the URL → upload an image → see predictions.

## 3) Large model (>100 MB) options

GitHub blocks files over 100 MB by default. If your `.pth` is larger:

- Use Git LFS in your repo for the model file; or
- Host the model on a remote storage (e.g., Hugging Face Hub, Google Drive, Dropbox) and download it at startup if not present. Example pattern in `app.py`:

```python
import os, requests
MODEL_URL = os.environ.get("MODEL_URL")  # set in Streamlit Cloud Secrets
MODEL_PATH = "brain_tumor_model.pth"
if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
```

Then set `MODEL_URL` in Streamlit Cloud → Settings → Secrets.

## 4) Requirements

`requirements.txt` contains:

- streamlit, torch, torchvision, numpy, pillow, matplotlib

Streamlit Cloud installs CPU wheels for PyTorch by default. If you see build issues, pin versions compatible with your Python version.

## 5) Alternatives (also free)

- Hugging Face Spaces (Gradio UI): very easy, especially for demos; supports LFS and has good ML ergonomics.
- Vercel + Render (already scaffolded here): host a lightweight UI on Vercel and a FastAPI backend on Render.

Pick Streamlit Cloud if you want the fastest path with your existing `app.py`.
