#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenTeddy quickstart — first-time setup + start on a fresh DGX/server
# ─────────────────────────────────────────────────────────────────────────────
#
# Run this ONCE on a freshly cloned OpenTeddy on a fresh node. It will:
#
#   1. Set up the CDP browser stack (Brave on ARM64, Edge on amd64) +
#      systemd unit, via setup-edge-cdp.sh.
#   2. Optionally open the GUI login helper (if you have $DISPLAY) so
#      you can sign in to X / LinkedIn / etc. before the first scrape.
#   3. Create the Python venv if missing + install requirements.
#   4. Start the OpenTeddy backend (uvicorn) detached in the background.
#   5. Verify everything is alive and tell you the URL.
#
# After this script, you can:
#   - Open http://<host>:8000 in any browser → chat with OpenTeddy
#   - Re-run scripts/login-helper.sh whenever a site's cookies expire
#
# Re-run safe (idempotent). If anything is already done, the script skips it.
#
# Usage:
#   bash scripts/quickstart.sh             # without login (assume already done)
#   bash scripts/quickstart.sh --login     # also run login-helper before backend
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

DO_LOGIN=false
for arg in "$@"; do
  case "${arg}" in
    --login)    DO_LOGIN=true ;;
    --no-login) DO_LOGIN=false ;;
    -h|--help)
      sed -n '4,30p' "$0"; exit 0 ;;
  esac
done

# ── 1. CDP browser stack ─────────────────────────────────────────────────────
if systemctl is-enabled --quiet openteddy-cdp.service 2>/dev/null \
   && systemctl is-active --quiet openteddy-cdp.service 2>/dev/null; then
  echo "▶ [1/4] CDP service already running — skipping setup-edge-cdp.sh."
else
  echo "▶ [1/4] Setting up CDP browser stack…"
  sudo bash scripts/setup-edge-cdp.sh
fi

# ── 2. Optional interactive login ───────────────────────────────────────────
if ${DO_LOGIN}; then
  if [ -n "${DISPLAY:-}" ]; then
    echo "▶ [2/4] Opening login helper. A browser window will appear."
    bash scripts/login-helper.sh
  else
    echo "▶ [2/4] --login requested but no \$DISPLAY available."
    echo "    Either:"
    echo "      - Reconnect via AnyDesk / physical monitor / ssh -X, then re-run."
    echo "      - Run scripts/setup-novnc-login.sh for a browser-based VNC."
    echo "    Skipping login for now — you can run scripts/login-helper.sh later."
  fi
else
  echo "▶ [2/4] Skipping login step (use --login or run scripts/login-helper.sh later)."
fi

# ── 3. Python venv + requirements ────────────────────────────────────────────
if [ ! -x ".venv/bin/python" ]; then
  echo "▶ [3/4] Creating Python venv…"
  python3 -m venv .venv
fi
echo "▶ [3/4] Installing Python requirements…"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
echo "  ✓ Python deps ready"

# ── 4. Start backend ─────────────────────────────────────────────────────────
# Already running? Skip.
if curl -sSf http://127.0.0.1:8000/health >/dev/null 2>&1 \
   || curl -sSf http://127.0.0.1:8000/version >/dev/null 2>&1; then
  echo "▶ [4/4] OpenTeddy backend already responding on :8000 — skipping start."
else
  echo "▶ [4/4] Starting OpenTeddy backend (uvicorn)…"
  # Kill any stale uvicorn from a previous boot.
  pkill -f "uvicorn.*main:app" 2>/dev/null || true
  sleep 1
  nohup .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 \
    > /tmp/openteddy.log 2>&1 &
  sleep 5
  if curl -sSf http://127.0.0.1:8000/version >/dev/null 2>&1; then
    echo "  ✓ Backend responding on http://127.0.0.1:8000"
  else
    echo "  ✗ Backend didn't come up. Tail of log:"
    tail -20 /tmp/openteddy.log
    exit 1
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo ""
echo "✅ OpenTeddy is up."
echo ""
echo "Chat in any browser:"
echo "    http://localhost:8000           (same machine)"
echo "    http://${HOST_IP}:8000          (LAN — adjust firewall as needed)"
echo ""
echo "Live backend log:"
echo "    tail -f /tmp/openteddy.log"
echo ""
echo "When a site's cookies expire (X every ~30 days, LinkedIn ~90):"
echo "    bash scripts/login-helper.sh"
