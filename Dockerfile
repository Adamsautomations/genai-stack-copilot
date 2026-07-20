# syntax=docker/dockerfile:1

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # fastembed caches the ONNX model here; baked at build time so a cold
    # container does not spend its first request downloading 130 MB.
    FASTEMBED_CACHE_PATH=/opt/fastembed

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Warm the embedding model into the image. Without this, the first query on a
# freshly scaled instance pays the model download before it can even embed.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY src/ ./src/
COPY web/ ./web/

# Pre-packed CAG prefix. Optional — the RAG path works without it; the CAG
# toggle degrades to a clear error rather than shipping a 120 MB corpus.
COPY data/cag_context.jso[n] ./data/

# Container Apps sets PORT; default matches local `uvicorn` usage.
ENV PORT=8000
EXPOSE 8000

# Single worker on purpose: the session spend cap is process-local state.
# Scaling out needs that moved to a shared store first — see README.
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
