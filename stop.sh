#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OpenTeddy backend stopper — the counterpart of run.sh.
#
# Quick start:
#   ./stop.sh            # stop the server on the default port 8000
#   ./stop.sh --port 8001
#   ./stop.sh --all      # stop every OpenTeddy uvicorn on this machine
#
# Why not just `pkill uvicorn`: run.sh starts uvicorn with --reload, which
# means TWO processes — a supervisor and the worker that actually holds
# the port. Kill only the worker and the supervisor respawns it a second
# later, which is exactly the "I killed it but the port is still busy"
# loop this script exists to end. We signal the supervisor first (it
# shuts its worker down cleanly), then the port holder, and only escalate
# to SIGKILL if the port is still bound after a grace period.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${OPENTEDDY_PORT:-8000}"
ALL=0
GRACE="${OPENTEDDY_STOP_GRACE:-5}"   # seconds to wait for a clean exit

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift ;;
    --all)  ALL=1 ;;
    -h|--help)
      cat <<EOF
OpenTeddy backend stopper.

Usage:
  ./stop.sh [--port N] [--all]

Flags:
  --port N   stop the server bound to port N (default 8000, or OPENTEDDY_PORT)
  --all      stop every 'uvicorn main:app' started from this checkout,
             whatever port it is on
EOF
      exit 0 ;;
    *) echo "✗ unknown flag: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# ── helpers ────────────────────────────────────────────────────────────────

# PID(s) currently LISTENING on the port. lsof is on macOS and most Linux
# boxes; fall back to ss (iproute2) where it isn't.
port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true
  fi
}

# uvicorn supervisors/workers for THIS app. With --all, every port.
#
# pgrep -f matches the whole command line, so a shell that merely CONTAINS
# the text "uvicorn main:app --port 8000" (a `bash -c` wrapper, an editor
# task, the terminal that launched run.sh) matches too — and terminating
# that means terminating the caller. Keep only real server processes
# (python/uvicorn executables) and never ourselves or our parent.
uvicorn_pids() {
  local pat pid name
  if [[ "$ALL" = "1" ]]; then
    pat="uvicorn main:app"
  else
    pat="uvicorn main:app.*--port[= ]$PORT(\$|[^0-9])"
  fi
  for pid in $(pgrep -f "$pat" || true); do
    [[ "$pid" = "$$" || "$pid" = "$PPID" ]] && continue
    name="$(basename "$(ps -o comm= -p "$pid" 2>/dev/null || true)")"
    case "$name" in
      python*|uvicorn*) echo "$pid" ;;
    esac
  done
}

alive() { kill -0 "$1" 2>/dev/null; }

# ── systemd hint ──────────────────────────────────────────────────────────
# If a unit manages the server, killing the process just makes systemd
# restart it. Say so instead of fighting it.
if command -v systemctl >/dev/null 2>&1; then
  unit="$(systemctl list-units --type=service --state=running --no-legend 2>/dev/null \
          | awk '{print $1}' | grep -i -m1 'teddy' || true)"
  if [[ -n "$unit" ]]; then
    echo "  ! $unit is managed by systemd — stop it there instead:"
    echo "      sudo systemctl stop $unit"
    exit 1
  fi
fi

# ── collect targets ───────────────────────────────────────────────────────
targets="$( { uvicorn_pids; [[ "$ALL" = "1" ]] || port_pids; } | sort -u | grep -v "^$$\$" || true)"

if [[ -z "$targets" ]]; then
  if [[ "$ALL" = "1" ]]; then
    echo "✓ no OpenTeddy server running"
  else
    echo "✓ nothing listening on port $PORT"
  fi
  exit 0
fi

echo "  stopping OpenTeddy (pids: $(echo "$targets" | tr '\n' ' '))"

# ── SIGTERM, then wait ────────────────────────────────────────────────────
for pid in $targets; do
  kill -TERM "$pid" 2>/dev/null || true
done

deadline=$(( $(date +%s) + GRACE ))
while [[ $(date +%s) -lt $deadline ]]; do
  remaining=""
  for pid in $targets; do alive "$pid" && remaining="$remaining $pid"; done
  if [[ "$ALL" = "0" ]] && [[ -n "$(port_pids)" ]]; then remaining="$remaining port"; fi
  [[ -z "$remaining" ]] && break
  sleep 0.25
done

# ── escalate if anything survived ─────────────────────────────────────────
leftovers="$( { for pid in $targets; do alive "$pid" && echo "$pid"; done; [[ "$ALL" = "1" ]] || port_pids; } | sort -u || true)"
if [[ -n "$leftovers" ]]; then
  echo "  ! still running after ${GRACE}s — sending SIGKILL (pids: $(echo "$leftovers" | tr '\n' ' '))"
  for pid in $leftovers; do kill -KILL "$pid" 2>/dev/null || true; done
  sleep 0.5
fi

# ── verify ────────────────────────────────────────────────────────────────
if [[ "$ALL" = "0" ]] && [[ -n "$(port_pids)" ]]; then
  echo "✗ port $PORT is still bound by pid(s): $(port_pids | tr '\n' ' ')" >&2
  echo "  (not an OpenTeddy process? try: lsof -iTCP:$PORT -sTCP:LISTEN)" >&2
  exit 1
fi

if [[ "$ALL" = "1" ]]; then
  echo "✓ all OpenTeddy servers stopped"
else
  echo "✓ port $PORT is free — ./run.sh to start again"
fi
