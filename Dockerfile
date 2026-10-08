FROM python:3.14.6-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/model-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app

# CPU wheels avoid downloading the Linux CUDA runtime for this small demo.
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && python -m pip check

# .dockerignore is an allowlist. No local .env, vectors, logs or model cache.
COPY acl_config.py audit.py auth.py authorization.py graph_rag.py ingest.py llm.py main.py \
    operations.py query_router.py search.py semantic_cache.py streamlit_app.py ./
COPY .env.example .gitignore README.md GITHUB.md FINAL_RESULTS.md CLOUD.md ./
COPY deploy/ ./deploy/
COPY ui/ ./ui/
COPY .streamlit/ ./.streamlit/
COPY data/acme/ ./data/acme/
COPY data/globex/ ./data/globex/
RUN python -m deploy.lock_dependencies \
    && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=False)'

# Only the four public fictional fixtures enter this fresh image's index.
# No provider key is used, and ingestion never touches the developer's store.
RUN HF_HUB_OFFLINE=1 python ingest.py

RUN useradd --create-home --uid 1000 rag \
    && mkdir -p /app/logs \
    && chown -R rag:rag /app /opt/model-cache
USER rag
ENV HF_HUB_OFFLINE=1 PORT=10000 MAX_INFLIGHT_ASK=1 MAX_INFLIGHT_LOGIN=1
EXPOSE 10000
CMD ["python", "-m", "deploy.hosted"]

# CI-only target. Runtime image never needs Git or the test sources.
FROM base AS validation
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --chown=rag:rag tests/ ./tests/
COPY --chown=rag:rag eval/ ./eval/
USER rag
RUN python -m deploy.ci backend || (tail -n 100 dist/ci/backend/unit_tests.log && exit 1)

# Keep the default target identical to the image deployed on the host.
FROM base AS runtime
