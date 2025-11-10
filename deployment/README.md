# Deployment: FAW Detection ONNX Service

This folder contains instructions to build and run the lightweight FastAPI + ONNX Runtime service that performs FAW detection.

Recommended flow (local / Docker):

1) Build the Docker image (PowerShell / CMD):

   The Docker image uses the lightweight `requirements.inference.txt` for runtime packages.

   docker build -t faw-detector .

2) Run the container and map port 8000. If your ONNX model is in the repo root it will be copied into the image during build. Otherwise mount it at runtime:

   docker run -p 8000:8000 faw-detector

   # or mount model from host
   docker run -p 8000:8000 -v C:\path\to\DenseNet121_finetuned_final.onnx:/app/DenseNet121_finetuned_final.onnx faw-detector

3) Health check:

   GET http://localhost:8000/health

4) Predict (multipart/form-data)

   POST http://localhost:8000/predict
   Form field: file (image file)

Response:

  {
    "label": "FAW" | "No_FAW",
    "score": 0.8234
  }

Notes:
- The Docker build uses `requirements.inference.txt` (a small, focused set of packages) so you won't install the full project development dependencies when building the image.
- The service defaults to a simple preprocessing pipeline that rescales images to 224x224 and normalizes to [0,1]. If you need ImageNet-style preprocessing (scale to [-1,1]) set environment variable PREPROCESS=imagenet.
- Model path can be overridden with MODEL_PATH environment variable.
- To validate the ONNX conversion, compare predictions from the Keras model and this service on a few samples.
