FROM python:3.11-bullseye

# Set working directory
WORKDIR /app

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Copy requirements and install
# Use a small, dedicated requirements file for the inference service
COPY requirements.inference.txt ./requirements.inference.txt

# Install small set of system libs required by some binary wheels (onnxruntime uses OpenMP)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates \
	&& rm -rf /var/lib/apt/lists/* \
	&& pip install --no-cache-dir -r requirements.inference.txt

# Copy application files and model (if present in repo root)
COPY . /app

# Expose port
EXPOSE 8000

# Default model path inside container
ENV MODEL_PATH=/app/DenseNet121_finetuned_final.onnx

# Run the app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
