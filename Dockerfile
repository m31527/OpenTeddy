# OpenTeddy — production-style container image.
#
# Build:
#   docker build -t openteddy .
#
# Run:
#   docker run --rm -p 8000:8000 -v openteddy-data:/data openteddy
#
# The container is stateless w.r.t. the OpenTeddy install; runtime
# state (tracker.db, memory_db, learned skills, agent workspace) lives
# under /data, which the caller should mount as a named volume or
# host-bind so it survives container restarts.
#
# Ollama is NOT bundled — the container assumes either:
#   (a) an Ollama daemon reachable on the host (set OLLAMA_BASE_URL),
#   (b) cloud LLM keys (Anthropic / OpenAI / Gemini / Deepseek /
#       OpenRouter) configured via Settings UI for a fully cloud run.
# Bundling Ollama into the image would double the image size and box
# the user into one model — both fights we don't want to pick.

FROM python:3.12-slim

# git: needed for the in-app updater path
# curl: gives us a healthcheck primitive without pulling in busybox
# build-essential: a few wheels (chromadb's hnswlib, pydantic_core
#   fallback path) compile from source on slim images that don't ship
#   prebuilt wheels for every arch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in a separate layer so changing source code
# doesn't bust the deps cache. requirements.txt landing first means
# 90% of rebuilds only re-copy the source layer (~10 MB) instead of
# re-installing the full stack (~400 MB).
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application source — copied last so iterative dev rebuilds are fast.
COPY . .

# Persistent data dir. tracker.db + memory_db + skills + workspace
# all redirected here so a container restart loses no state.
VOLUME ["/data"]
ENV TRACKER_DB_PATH=/data/openteddy.db \
    MEMORY_DB_PATH=/data/memory_db \
    SKILLS_DIR=/data/skills \
    AGENT_WORKSPACE_DIR=/data/agent-workspace

EXPOSE 8000

# Healthcheck — uvicorn's /health route returns 200 once lifespan
# startup finished. Useful for compose / k8s readiness gates.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
