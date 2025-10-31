from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

app = FastAPI(title="Brain Tumor API (skeleton)")

# Allow Vercel preview URLs and localhost by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"ok": True, "message": "Backend is running. POST an image to /predict"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    # TODO: Load your PyTorch model once at startup and reuse it here.
    # Example outline:
    #   - load bytes = await file.read()
    #   - image = PIL.Image.open(io.BytesIO(bytes)).convert("RGB")
    #   - preprocess -> tensor
    #   - with torch.no_grad(): logits = model(tensor)
    #   - map to class label
    #   - return {"label": label, "scores": {...}}
    # For now, just return a stub response.
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    return {"ok": True, "note": "Replace with real model inference", "filename": file.filename}


# To run locally:
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000
