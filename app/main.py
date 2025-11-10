import io
import os
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

app = FastAPI(title="FAW Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify: ["https://your-domain.com"])
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


# Load model at startup
try:
    session, input_meta = load_session(MODEL_PATH)
except Exception as e:
    # Defer raising until first call if model not available at import-time in some environments
    session = None
    input_meta = None
    load_error = e
else:
    load_error = None


def _model_expects_channels_first(meta: dict) -> bool:
    shape = meta.get("shape")
    # shape usually like [None, 3, 224, 224] for NCHW or [None, 224, 224, 3] for NHWC
    if not shape or len(shape) != 4:
        return False
    # if second dim == 3 => channels first
    return shape[1] == 3


def preprocess_image(data: bytes) -> np.ndarray:
    """Preprocess image bytes to model input.

    Steps:
    - open with PIL, convert to RGB
    - resize to IMG_SIZE
    - convert to float32
    - scale according to PREPROCESS env var
    - add batch dim and transpose if model expects channels-first
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)

    if PREPROCESS == "imagenet":
        # ImageNet-style TF preprocessing: scale to [-1, 1]
        arr = arr / 127.5 - 1.0
    else:
        # default: scale to [0, 1]
        arr = arr / 255.0

    # determine channel order from loaded model metadata
    if input_meta and _model_expects_channels_first(input_meta):
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))

    # add batch dim
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": session is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict FAW (Fall Armyworm) presence from uploaded image file.

    Returns JSON with 'label' (FAW | No_FAW) and 'score' (float between 0 and 1).
    """
    global session, input_meta
    if load_error and session is None:
        # try lazy load if possible
        try:
            session, input_meta = load_session(MODEL_PATH)
            # clear load_error
            # (if this fails, we'll raise below)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    data = await file.read()
    try:
        model_input = preprocess_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    input_name = input_meta.get("name") if input_meta else session.get_inputs()[0].name

    try:
        outputs = session.run(None, {input_name: model_input})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # assume model outputs a single score (sigmoid) or probability
    score = float(np.asarray(outputs[0]).ravel()[0])
    label = "FAW" if score >= 0.5 else "No_FAW"

    return {"label": label, "score": score}
