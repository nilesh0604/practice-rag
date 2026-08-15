# Multi-stage Dockerfile for the practice-rag FastAPI backend (Step 7).
#
# Builds a slim Python 3.12 image that serves the API with uvicorn. The
# container connects to the host's Ollama (deviation D1) and the Qdrant
# container on the compose network. Per the project rule, the container is
# capped at 1 CPU / 1 GB RAM via docker-compose deploy.resources.limits.
#
# The seed corpus is copied in so the /api/v1/ingest endpoint can re-index
# without a bind mount. The conda env is NOT reproduced here — pip3 installs
# requirements.txt into the image's system Python (slim base, no conda needed
# in the container).

FROM python:3.12-slim AS base

# Avoid .pyc files + unbuffered stdout so logs appear immediately in docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build deps for any source wheels (pypdf, etc.), then remove after.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the application code + seed corpus.
COPY api/ ./api/
COPY rag/ ./rag/
COPY schemas/ ./schemas/
COPY ingestion/ ./ingestion/
COPY eval/ ./eval/
COPY data/corpus/ ./data/corpus/
COPY pyproject.toml ./

EXPOSE 8000

# Healthcheck uses the liveness endpoint (cheap, dependency-free).
HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health',timeout=3).status==200 else 1)"

# uvicorn with a single worker (single-user practice project) + the warm-up
# lifespan handles Ollama model pre-load on startup.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
