# ============================================================
# TP53 RAG Platform — container image (multi-arch: amd64 + arm64)
# Dedicated to tp53_rag (NOT the root/classic-pipeline Dockerfiles).
# python:3.11-slim is multi-arch, so the same Dockerfile builds for
# Raspberry Pi (arm64) via:  docker buildx build --platform linux/arm64
# ============================================================

# ---- builder: compile/install deps (build tools stay out of final image) ----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime: slim image with only runtime libs ----
FROM python:3.11-slim

# libgomp1 = OpenMP for onnxruntime/numpy; ffmpeg = optional voice transcription
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Bring in the installed Python packages from the builder
COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

# Non-root user; HOME points into /app so the ONNX/model caches are writable
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/logs \
    && chown -R appuser:appuser /app
USER appuser

ENV HOME=/app \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    INFERENCE_MODE=api

EXPOSE 8501

# Version-agnostic health check: is the server accepting connections on 8501?
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import socket,sys; s=socket.create_connection(('127.0.0.1',8501),timeout=5); s.close()" \
        || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
