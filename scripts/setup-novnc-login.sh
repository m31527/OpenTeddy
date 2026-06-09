#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenTeddy CDP login helper — Xvfb + noVNC for headless servers
# ─────────────────────────────────────────────────────────────────────────────
#
# Companion to setup-edge-cdp.sh. The headless browser that script sets up is
# perfect for SCRAPING but cannot accept an interactive LOGIN — there's no
# GUI for you to type a password into. This script bolts on an interactive
# login path for headless servers (DGX Spark, Jetson, rack machines without
# a monitor) so the entire workflow stays on the server itself, no Mac /
# laptop in the loop.
#
# Architecture:
#
#                       ┌────────────────────────────────────────────┐
#                       │  Ubuntu DGX (this host)                    │
#                       │                                            │
#                       │  systemd: openteddy-cdp.service            │
#                       │    └─ headless Brave on :9222              │
#                       │       reads /var/lib/openteddy/edge-profile│
#                       │                                            │
#                       │  systemd: openteddy-novnc-login.service    │
#                       │    └─ Xvfb :99 (virtual display)           │
#                       │       └─ x11vnc → :5900                    │
#                       │          └─ websockify → :6080 (web)       │
#                       └────────────────────────────────────────────┘
#                                            │
#                                            │  Operator opens
#                                            ▼  http://<host>:6080
#                       ┌────────────────────────────────────────────┐
#                       │  Any browser (phone, laptop, another box)  │
#                       │  Sees a chromium window running on DGX     │
#                       │  Logs in to X / LinkedIn / etc.            │
#                       │  Closes the tab when done                  │
#                       └────────────────────────────────────────────┘
#
# The login happens in a brave-browser that runs on the SAME DGX, uses the
# SAME profile dir, as the headless service. So once the operator logs in,
# the headless CDP-attached scraper immediately picks up those cookies on
# the next call — no scp, no storage_state.json juggling.
#
# This is what fleet-grade browser automation looks like: each node owns
# its own login state, set up once via web-based VNC, reused indefinitely.
#
# Usage:
#   sudo bash scripts/setup-novnc-login.sh                    # install
#   sudo systemctl start openteddy-novnc-login                # start the
#                                                              # virtual
#                                                              # display
#   # then point any browser at http://<this-host>:6080
#   # log in to whatever sites you need
#   sudo systemctl stop openteddy-novnc-login                 # stop when
#                                                              # done — the
#                                                              # cookies
#                                                              # persist in
#                                                              # the profile
#                                                              # dir
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

TARGET_USER="${OPENTEDDY_USER:-${SUDO_USER:-${USER:-openteddy}}}"
if ! id -u "${TARGET_USER}" >/dev/null 2>&1; then
  echo "✗ Target user '${TARGET_USER}' does not exist."
  exit 1
fi

PROFILE_DIR="/var/lib/openteddy/edge-profile"
SERVICE_FILE="/etc/systemd/system/openteddy-novnc-login.service"
NOVNC_PORT="${OPENTEDDY_NOVNC_PORT:-6080}"
VNC_DISPLAY=":99"
VNC_PORT="5900"
SCREEN_RES="1440x900x24"

if [ "$(id -u)" -ne 0 ]; then
  echo "✗ Run with sudo: sudo bash $0"
  exit 1
fi

# Detect browser binary the way setup-edge-cdp.sh chose it.
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
  echo "✗ No Chromium-based browser found. Run setup-edge-cdp.sh first."
  exit 1
fi

echo "▶ OpenTeddy noVNC login helper"
echo "    target user : ${TARGET_USER}"
echo "    profile dir : ${PROFILE_DIR}"
echo "    browser     : ${BROWSER_LABEL} (${BROWSER_BIN})"
echo "    noVNC port  : ${NOVNC_PORT}"
echo ""

# ── 1. Dependencies ──────────────────────────────────────────────────────────
echo "▶ Installing Xvfb / x11vnc / noVNC…"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  xvfb x11vnc novnc websockify \
  >/dev/null
echo "  ✓ Installed"

# ── 2. Wrapper script that starts the whole login stack ──────────────────────
# We use a wrapper instead of jamming the whole pipeline into ExecStart so the
# wait/sleep ordering between Xvfb → browser → x11vnc → websockify is sane.
# Without that, the browser sometimes spawns before Xvfb has bound :99 and
# silently falls back to its own internal display, which doesn't appear in
# the VNC session at all.
WRAPPER=/usr/local/bin/openteddy-novnc-login-runner
echo "▶ Writing wrapper to ${WRAPPER}…"
cat > "${WRAPPER}" <<EOF
#!/usr/bin/env bash
set -e

# Pause the headless CDP service so brave can take over the profile dir
# (Chromium-based browsers refuse to start a second instance on the same
# --user-data-dir; the "ProcessSingleton" lock is hard).
systemctl stop openteddy-cdp.service 2>/dev/null || true

# 1. Virtual display
Xvfb ${VNC_DISPLAY} -screen 0 ${SCREEN_RES} &
XVFB_PID=\$!
sleep 1

# 2. Browser with --remote-debugging-port=9222 so OpenTeddy can still attach
#    during the login session if needed. Headful (no --headless flag) so
#    the user can interact with it via VNC.
DISPLAY=${VNC_DISPLAY} "${BROWSER_BIN}" \\
  --remote-debugging-port=9222 \\
  --user-data-dir=${PROFILE_DIR} \\
  --no-sandbox \\
  --disable-gpu \\
  --window-size=1440,900 \\
  --window-position=0,0 \\
  about:blank &
BROWSER_PID=\$!
sleep 2

# 3. x11vnc on VNC_PORT, no password (we only bind to localhost; websockify
#    is what listens publicly on NOVNC_PORT).
x11vnc -display ${VNC_DISPLAY} -nopw -listen 0.0.0.0 -rfbport ${VNC_PORT} -forever -shared -bg

# 4. websockify wraps the VNC stream in a WebSocket so noVNC's vnc.html
#    can connect from any browser. Foreground so systemd can supervise.
exec websockify --web /usr/share/novnc ${NOVNC_PORT} localhost:${VNC_PORT}
EOF
chmod +x "${WRAPPER}"
echo "  ✓ Wrapper written"

# Companion cleanup script — restores the headless CDP service when the
# operator stops the login service.
CLEANUP=/usr/local/bin/openteddy-novnc-login-cleanup
cat > "${CLEANUP}" <<'EOF'
#!/usr/bin/env bash
# Kill the headful brave + x11vnc + Xvfb so port 9222 frees up, then bring
# the headless CDP service back. Best-effort: if any step fails the next
# `systemctl start openteddy-cdp.service` retry will fix it.
pkill -f "x11vnc -display :99" 2>/dev/null || true
pkill -f "Xvfb :99"            2>/dev/null || true
pkill -f "brave-browser.*remote-debugging-port=9222" 2>/dev/null || true
pkill -f "microsoft-edge.*remote-debugging-port=9222" 2>/dev/null || true
pkill -f "chromium.*remote-debugging-port=9222" 2>/dev/null || true
sleep 1
systemctl start openteddy-cdp.service 2>/dev/null || true
EOF
chmod +x "${CLEANUP}"

# ── 3. systemd unit ──────────────────────────────────────────────────────────
echo "▶ Writing systemd unit to ${SERVICE_FILE}…"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=OpenTeddy interactive login via noVNC (port ${NOVNC_PORT})
Documentation=https://github.com/m31527/OpenTeddy
After=network-online.target
Conflicts=openteddy-cdp.service

[Service]
Type=simple
User=${TARGET_USER}
Restart=no
ExecStart=${WRAPPER}
ExecStopPost=${CLEANUP}

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
echo "  ✓ Unit written. NOT auto-started — start it manually when you want"
echo "    to log in to a site."

# ── 4. Friendly summary ──────────────────────────────────────────────────────
HOST_IP="$(hostname -I | awk '{print $1}')"
echo ""
echo "✅ noVNC login helper installed."
echo ""
echo "When you need to log in to a site (X, LinkedIn, corp wiki, …):"
echo ""
echo "  1. Start the helper (this stops the headless CDP service):"
echo "       sudo systemctl start openteddy-novnc-login.service"
echo ""
echo "  2. From any browser (phone, laptop, even this DGX over SSH X-fwd):"
echo "       http://${HOST_IP}:${NOVNC_PORT}/vnc.html"
echo "     Click 'Connect' (no password) → you'll see a ${BROWSER_LABEL}"
echo "     window running on this host."
echo ""
echo "  3. In that window: open the site, log in, close the tab when done."
echo "     Cookies are saved to ${PROFILE_DIR}."
echo ""
echo "  4. Stop the helper (this restarts the headless CDP service):"
echo "       sudo systemctl stop openteddy-novnc-login.service"
echo ""
echo "  5. Verify OpenTeddy now sees the login:"
echo "       cd ~/OpenTeddy && .venv/bin/python -c \\"
echo "         'import asyncio,sys; sys.path.insert(0,\".\");"
echo "          from tools.chrome_attached_tool import chrome_attach_check;"
echo "          import json; print(json.dumps(asyncio.run("
echo "            chrome_attach_check()), indent=2))'"
echo ""
echo "Security note: noVNC port ${NOVNC_PORT} is bound to 0.0.0.0 so other"
echo "machines on your network can reach it (the whole point — VNC from"
echo "your phone). If you don't want that, edit ${WRAPPER} and change the"
echo "websockify line from '${NOVNC_PORT}' to '127.0.0.1:${NOVNC_PORT}',"
echo "then access via SSH tunnel only."
