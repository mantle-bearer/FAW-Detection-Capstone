import io
import os
from typing import Tuple
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# CLIP Integration (local)
from transformers import pipeline

# Initialize local CLIP model
try:
    print("🔄 Loading CLIP model...")
    clip_pipeline = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
    clip_enabled = True
    print("✅ CLIP model loaded successfully.")
except Exception as e:
    clip_enabled = False
    clip_pipeline = None
    print(f"⚠️ CLIP model failed to load: {e}")

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "DenseNet121_finetuned_final.onnx")
IMG_SIZE = (224, 224)
PREPROCESS = os.getenv("PREPROCESS", "0-1")  # options: '0-1' or 'imagenet'

# Candidate labels for CLIP pre-check
clip_labels = [
    "maize crop leaf",
    "plant with worms or insect damage",
    "healthy green leaf",
    "human face",
    "animal",
    "car",
    "text or drawing",
    "random object",
]

CLIP_THRESHOLD = 0.45

app = FastAPI(title="FAW Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_session(path: str) -> Tuple[ort.InferenceSession, dict]:
    """Load ONNX model and return session and input metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found at: {path}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    meta = {"name": inp.name, "shape": inp.shape, "dtype": inp.type}
    return sess, meta

# Load ONNX model
try:
    session, input_meta = load_session(MODEL_PATH)
    load_error = None
    print("✅ ONNX model loaded successfully.")
except Exception as e:
    session, input_meta, load_error = None, None, e
    print(f"⚠️ Model load error: {e}")

def _model_expects_channels_first(meta: dict) -> bool:
    shape = meta.get("shape")
    return len(shape) == 4 and shape[1] == 3

def preprocess_image(data: bytes) -> np.ndarray:
    """Preprocess image bytes to model input."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    arr = arr / 127.5 - 1.0 if PREPROCESS == "imagenet" else arr / 255.0
    if input_meta and _model_expects_channels_first(input_meta):
        arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def detect_context_locally(data: bytes):
    """Classify context using local CLIP zero-shot pipeline."""
    if not clip_enabled:
        return None
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        results = clip_pipeline(image, candidate_labels=clip_labels)
        best = max(results, key=lambda x: x["score"])
        label, confidence = best["label"], best["score"]
        relevant = label in clip_labels[:3] and confidence >= CLIP_THRESHOLD
        return {"label": label, "confidence": confidence, "relevant": relevant}
    except Exception as e:
        print("⚠️ CLIP failed:", e)
        return None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict FAW (Fall Armyworm) presence from uploaded image file."""
    global session, input_meta
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    data = await file.read()

    # Run CLIP pre-check
    ctx = detect_context_locally(data)
    if ctx and not ctx["relevant"]:
        return {
            "label": "Not relevant",
            "reason": ctx["label"],
            "confidence": ctx["confidence"],
        }

    # Run FAW detection
    try:
        model_input = preprocess_image(data)
        input_name = input_meta.get("name") if input_meta else session.get_inputs()[0].name
        outputs = session.run(None, {input_name: model_input})
        score = float(np.asarray(outputs[0]).ravel()[0])
        label = "FAW Detected" if score >= 0.5 else "Healthy Crop"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {
        "label": label,
        "score": round(score, 4),
        "context": ctx if ctx else "CLIP not available",
    }
