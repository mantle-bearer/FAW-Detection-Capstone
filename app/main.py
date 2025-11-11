import io
import os
import requests
from typing import Tuple

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "DenseNet121_finetuned_final.onnx")
IMG_SIZE = (224, 224)
PREPROCESS = os.getenv("PREPROCESS", "0-1")  # options: '0-1' or 'imagenet'

# Hugging Face CLIP configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_CLIP_MODEL = os.getenv("HF_CLIP_MODEL", "openai/clip-vit-base-patch32")
ENABLE_CLIP = os.getenv("ENABLE_CLIP", "1") in ("1", "true", "True")
CLIP_THRESHOLD = float(os.getenv("CLIP_THRESHOLD", "0.45"))

clip_labels = [
    "a close-up photo of a maize crop leaf",
    "a plant with worms or insect damage",
    "a healthy green leaf",
    "a human face",
    "an animal",
    "a car",
    "a random object",
    "text or drawing",
]

app = FastAPI(title="FAW Detection API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- MODEL LOAD ---------------------- #
def load_session(path: str) -> Tuple[ort.InferenceSession, dict]:
    """Load ONNX model and return session and input metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found at: {path}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    meta = {"name": inp.name, "shape": inp.shape, "dtype": inp.type}
    return sess, meta


try:
    session, input_meta = load_session(MODEL_PATH)
except Exception as e:
    session, input_meta, load_error = None, None, e
else:
    load_error = None


# ---------------------- HELPERS ---------------------- #
def _model_expects_channels_first(meta: dict) -> bool:
    shape = meta.get("shape")
    return len(shape) == 4 and shape[1] == 3


def preprocess_image(data: bytes) -> np.ndarray:
    """Resize, scale, and batch image input."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)

    if PREPROCESS == "imagenet":
        arr = arr / 127.5 - 1.0
    else:
        arr = arr / 255.0

    if input_meta and _model_expects_channels_first(input_meta):
        arr = np.transpose(arr, (2, 0, 1))

    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def detect_context_via_hf(data: bytes):
    """Use Hugging Face Inference API to detect context using CLIP."""
    if not (ENABLE_CLIP and HF_API_TOKEN):
        return None

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{HF_CLIP_MODEL}"

    payload = {
        "inputs": {
            "image": data,
            "parameters": {"candidate_labels": clip_labels}
        }
    }

    try:
        response = requests.post(api_url, headers=headers, files={"image": data}, timeout=15)
        if response.status_code != 200:
            print("⚠️ HF API error:", response.text)
            return None

        result = response.json()
        # Some models return a list of dicts
        if isinstance(result, list) and len(result) > 0 and "score" in result[0]:
            best = max(result, key=lambda x: x["score"])
            label, confidence = best["label"], best["score"]
        else:
            return None

        relevant = label in clip_labels[:3] and confidence >= CLIP_THRESHOLD
        return {"label": label, "confidence": confidence, "relevant": relevant}

    except Exception as e:
        print("HF CLIP inference failed:", e)
        return None


# ---------------------- ROUTES ---------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict FAW presence, after pre-filtering irrelevant images."""
    global session, input_meta
    if load_error and session is None:
        try:
            session, input_meta = load_session(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    data = await file.read()

    # 1️⃣ CLIP Pre-check via Hugging Face
    ctx = detect_context_via_hf(data)
    if ctx and not ctx.get("relevant"):
        return JSONResponse(
            status_code=200,
            content={
                "label": "Unrelated",
                "reason": ctx.get("label"),
                "confidence": ctx.get("confidence"),
                "message": "Image appears unrelated to crops or leaves.",
            },
        )

    # 2️⃣ Proceed with FAW ONNX Model
    try:
        model_input = preprocess_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    input_name = input_meta.get("name") if input_meta else session.get_inputs()[0].name
    outputs = session.run(None, {input_name: model_input})
    score = float(np.asarray(outputs[0]).ravel()[0])
    label = "FAW" if score >= 0.5 else "No_FAW"

    return {"label": label, "score": score, "context": ctx or {}}
