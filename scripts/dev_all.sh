#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dev_all.sh — start uvicorn + tauri dev together, kill both on Ctrl-C.
#
# Replaces the "open two terminals every morning" ritual:
#   Terminal 1: uvicorn main:app --reload
#   Terminal 2: cd desktop && npx tauri dev
#
# Usage:
#   scripts/dev_all.sh
#
# Both processes share this terminal's stdout. Ctrl-C terminates both
# cleanly via the trap below — no orphan uvicorn / tauri-dev processes
# left listening on ports.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP_DIR="$ROOT_DIR/desktop"

# ── Pre-flight ────────────────────────────────────────────────────────────
if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "✗ .venv not found at $ROOT_DIR/.venv — set up your Python venv first." >&2
  echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi
if [[ ! -d "$DESKTOP_DIR/node_modules" ]]; then
  echo "! desktop/node_modules missing — running npm install once…"
  ( cd "$DESKTOP_DIR" && npm install )
fi

# ── Process management ────────────────────────────────────────────────────
# Track child PIDs so the trap below can kill them all on Ctrl-C.
PIDS=()

cleanup() {
  echo
  echo "▸ Stopping dev processes…"
  # Kill all children. -SIGTERM first; if anything refuses we follow
  # up with -KILL after a beat.
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# ── Start uvicorn ─────────────────────────────────────────────────────────
echo "▸ Starting uvicorn on :8000 …"
(
  cd "$ROOT_DIR"
  # Activate venv inside the subshell so this script doesn't pollute
  # the caller's environment.
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exec uvicorn main:app --reload
) &
PIDS+=("$!")

# Wait until /health responds before kicking off tauri — the desktop's
# port-probe runs on iframe load, and racing against uvicorn boot
# would leave the user staring at "Backend not running" for 10 s.
echo "▸ Waiting for backend /health …"
for _ in {1..30}; do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "  ✓ backend up"
    break
  fi
  sleep 0.5
done

# ── Start tauri dev ───────────────────────────────────────────────────────
echo "▸ Starting tauri dev …"
(
  cd "$DESKTOP_DIR"
  exec npx tauri dev
) &
PIDS+=("$!")

# ── Wait for either child to exit ─────────────────────────────────────────
# `wait -n` returns when ANY child terminates. We then trip the
# cleanup trap to take down the survivor too — keeps the dev loop
# atomic (one ctrl-c kills both, one crash kills the other).
wait -n
echo "▸ One of the dev processes exited; tearing down the other."
