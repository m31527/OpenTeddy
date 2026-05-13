#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OpenTeddy backend launcher.
#
# Quick start:
#   ./run.sh           # default port 8000
#   ./run.sh --open    # also open browser when ready
#   ./run.sh --port 8001
#
# Re-entrant: if you forgot to activate .venv, this script sources it for
# you. Aborts loudly if install.sh hasn't been run yet (no .venv).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${OPENTEDDY_PORT:-8000}"
HOST="${OPENTEDDY_HOST:-127.0.0.1}"
DO_OPEN=0
RELOAD=1   # --reload by default so editing main.py hot-reloads uvicorn

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)      PORT="$2"; shift ;;
    --host)      HOST="$2"; shift ;;
    --open)      DO_OPEN=1 ;;
    --no-reload) RELOAD=0 ;;
    -h|--help)
      cat <<EOF
OpenTeddy backend launcher.

Usage:
  ./run.sh [--port N] [--host H] [--open] [--no-reload]

Flags:
  --port N      bind port (default 8000, override via OPENTEDDY_PORT env)
  --host H      bind host (default 127.0.0.1, override via OPENTEDDY_HOST)
  --open        open http://host:port in your default browser when ready
  --no-reload   don't pass --reload to uvicorn (useful for production-style runs)
EOF
      exit 0 ;;
    *) echo "✗ unknown flag: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# Activate venv if the user didn't already. Most people just want
# `./run.sh` to work without remembering the source-activate dance.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -f .venv/bin/activate ]]; then
    cat <<EOF >&2
✗ .venv not found.

OpenTeddy needs a Python virtualenv with deps installed. Run install.sh
first:

    curl -fsSL https://openteddy.net/install | bash

Or if you cloned manually:

    cd $SCRIPT_DIR
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
EOF
    exit 1
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Friendly heads-up if Ollama is installed but the daemon isn't running.
# Doesn't block startup — user might have Anthropic key configured for
# pure-cloud mode.
if command -v ollama >/dev/null 2>&1; then
  if ! curl -fsSL -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "  ! Ollama installed but daemon isn't responding."
    echo "    Start Ollama.app from /Applications, or run 'ollama serve' in another tab."
    echo "    (Continuing — cloud LLMs still work without Ollama.)"
    echo
  fi
fi

# Background browser-open: fire after a short sleep so uvicorn has a beat
# to bind. Tolerates both macOS (`open`) and Linux (`xdg-open`); silently
# skips on systems that have neither.
if [[ "$DO_OPEN" = "1" ]]; then
  (
    sleep 1.5
    if command -v open >/dev/null 2>&1; then
      open "http://$HOST:$PORT" 2>/dev/null || true
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "http://$HOST:$PORT" 2>/dev/null || true
    fi
  ) &
fi

# exec so signals (Ctrl+C) hit uvicorn directly, not this wrapper.
RELOAD_FLAG=""
[[ "$RELOAD" = "1" ]] && RELOAD_FLAG="--reload"
exec uvicorn main:app --host "$HOST" --port "$PORT" $RELOAD_FLAG
