#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenTeddy CDP login helper — simplest possible flow
# ─────────────────────────────────────────────────────────────────────────────
#
# For environments where you already have a desktop session on the host:
#   - AnyDesk / TeamViewer / RustDesk to a Linux desktop
#   - Physical monitor + keyboard plugged into the DGX / server
#   - VNC into the GNOME / KDE session
#   - SSH with X11 forwarding (ssh -X)
#
# All you need is `echo $DISPLAY` to return something (e.g. ":0", ":1").
#
# What this script does (3 steps):
#   1. Pause the headless CDP service so the browser can take the profile.
#   2. Launch Brave / Edge / Chromium with the same profile dir,
#      WITH a visible window. You log in to whatever sites you need.
#   3. When you close the browser window, automatically restart the
#      headless CDP service. Cookies persist in the profile dir, so the
#      headless scraper picks up the new login on the next call.
#
# No noVNC, no SSH tunnels, no Mac involvement. Just run it on the
# desktop session you already have.
#
# Usage:
#   bash scripts/login-helper.sh
#
# Or one-liner for the impatient:
#   curl -sSL https://raw.githubusercontent.com/m31527/OpenTeddy/main/scripts/login-helper.sh | bash
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROFILE_DIR="${OPENTEDDY_PROFILE_DIR:-/var/lib/openteddy/edge-profile}"
LISTEN_PORT="${OPENTEDDY_CDP_PORT:-9222}"

# ── Pre-flight: do we have a desktop? ────────────────────────────────────────
if [ -z "${DISPLAY:-}" ]; then
  cat <<'EOF'
✗ No $DISPLAY set — this script needs a GUI desktop session.

Easy ways to get one on a headless server:
  • AnyDesk / TeamViewer / RustDesk into the desktop
  • Physical monitor + keyboard plugged in
  • If you're SSH-ing now, reconnect with `ssh -X admin@<host>` so
    Brave's window forwards to YOUR local screen
  • Run scripts/setup-novnc-login.sh for a browser-based VNC setup

Or, if you're not really headless (you're inside AnyDesk):
  echo $DISPLAY
  # should print something like :0 or :1
  # if empty, your AnyDesk session lost its X11 connection — reconnect
EOF
  exit 1
fi

# ── Detect which browser is installed (same priority as setup-edge-cdp.sh) ──
if command -v brave-browser >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v brave-browser)"
  BROWSER_LABEL="Brave"
elif command -v microsoft-edge >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v microsoft-edge)"
  BROWSER_LABEL="Microsoft Edge"
elif command -v chromium >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v chromium)"
  BROWSER_LABEL="Chromium"
elif command -v chromium-browser >/dev/null 2>&1; then
  BROWSER_BIN="$(command -v chromium-browser)"
  BROWSER_LABEL="Chromium"
else
  echo "✗ No Chromium-based browser found."
  echo "  Run first: sudo bash scripts/setup-edge-cdp.sh"
  exit 1
fi

# ── Need sudo for systemctl. Check before we get into the browser flow. ──────
if ! sudo -n true 2>/dev/null; then
  echo "▶ sudo password required to pause/resume openteddy-cdp.service…"
  sudo -v   # cache credentials
fi

echo "▶ OpenTeddy login helper"
echo "    browser     : ${BROWSER_LABEL} (${BROWSER_BIN})"
echo "    profile dir : ${PROFILE_DIR}"
echo "    DISPLAY     : ${DISPLAY}"
echo ""

# ── 1. Pause headless ────────────────────────────────────────────────────────
if systemctl is-active --quiet openteddy-cdp.service; then
  echo "▶ Pausing headless openteddy-cdp.service (so the GUI browser can"
  echo "  take the profile dir's ProcessSingleton lock)…"
  sudo systemctl stop openteddy-cdp.service
  sleep 1
fi

# Belt-and-braces: kill any stray browser process holding the profile lock.
# Happens when a previous login helper run exited uncleanly.
pkill -f "remote-debugging-port=${LISTEN_PORT}" 2>/dev/null || true
sleep 1
rm -f "${PROFILE_DIR}/SingletonLock" "${PROFILE_DIR}/SingletonCookie" 2>/dev/null || true

# ── 2. Launch GUI browser ────────────────────────────────────────────────────
cat <<EOF
─────────────────────────────────────────────────────────────────────
  ▶ Opening ${BROWSER_LABEL} now. A window will appear on your
    desktop. Inside it:

      1. Navigate to the site(s) you need:
           https://x.com/login
           https://www.linkedin.com/login
           (or any other login-gated site)

      2. Sign in normally.

      3. CLOSE THE BROWSER WINDOW when done. This script will
         automatically restart the headless service.

  Cookies will be saved to:
      ${PROFILE_DIR}

  Press Ctrl-C in this terminal to abort instead of closing the
  browser window — the headless service will still come back.
─────────────────────────────────────────────────────────────────────
EOF

# Trap so the headless service comes back even on Ctrl-C / SIGHUP.
trap '_restart_headless' EXIT INT TERM HUP
_restart_headless() {
  echo ""
  echo "▶ Restarting headless openteddy-cdp.service…"
  pkill -f "remote-debugging-port=${LISTEN_PORT}" 2>/dev/null || true
  sleep 1
  rm -f "${PROFILE_DIR}/SingletonLock" "${PROFILE_DIR}/SingletonCookie" 2>/dev/null || true
  sudo systemctl start openteddy-cdp.service
  sleep 3
  if curl -sSf "http://127.0.0.1:${LISTEN_PORT}/json/version" >/dev/null 2>&1; then
    echo "  ✓ Headless CDP back up on 127.0.0.1:${LISTEN_PORT}"
    echo ""
    echo "✅ Login complete. Verify from OpenTeddy:"
    echo "    cd ~/OpenTeddy && .venv/bin/python -c \\"
    echo "      'import asyncio,sys; sys.path.insert(0,\".\");"
    echo "       from tools.chrome_attached_tool import x_search;"
    echo "       import json; print(json.dumps(asyncio.run("
    echo "         x_search(query=\"黴菌\", top_n=3)),"
    echo "         indent=2, ensure_ascii=False))'"
  else
    echo "  ⚠ Headless didn't come up — diagnose with:"
    echo "      sudo journalctl -u openteddy-cdp.service -n 30"
  fi
}

# Run the GUI browser in foreground. Script blocks here until the user
# closes the window (or Ctrl-C aborts).
"${BROWSER_BIN}" \
  --user-data-dir="${PROFILE_DIR}" \
  --no-sandbox \
  --remote-debugging-port="${LISTEN_PORT}" \
  about:blank https://x.com/login

# Trap will fire here as the script exits cleanly.
