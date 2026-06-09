#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenTeddy CDP browser setup — one-shot deploy script
# ─────────────────────────────────────────────────────────────────────────────
#
# Installs Microsoft Edge (the only Chromium-based browser with an official
# Linux ARM64 + amd64 apt repo) and wires it up as a systemd service that
# listens on 127.0.0.1:9222 for Chrome DevTools Protocol connections. After
# this runs, OpenTeddy's chrome_attached_tool can borrow the browser to
# scrape login-gated sites (X / Twitter, LinkedIn, paid SaaS, corp wikis)
# using the operator's actual authenticated session.
#
# Why Edge (and not snap-chromium / chrome / playwright bundled chromium):
#   - Official apt repo from Microsoft → no PPA / no snap sandbox / no manual
#     installer juggling. Same management story as nginx / postgres.
#   - Official Linux ARM64 build → works on DGX Spark, Raspberry Pi 5,
#     NVIDIA Jetson, Ampere boxes. Google Chrome has no ARM64 Linux build.
#   - Auto-updates via standard `apt upgrade`. Pin / hold packages the same
#     way you'd pin any other apt package on the fleet.
#   - It's Chromium under the hood — every CDP method, every DevTools API
#     call works identically. OpenTeddy's connect_over_cdp("http://...:9222")
#     doesn't care which Chromium-flavor it talks to.
#
# What the script does:
#   1. Adds the Microsoft apt repo + GPG key (skipped if already present).
#   2. apt installs microsoft-edge-stable.
#   3. Creates /var/lib/openteddy/edge-profile owned by the target user.
#   4. Writes /etc/systemd/system/openteddy-cdp.service with Restart=always
#      so a crash doesn't leave the fleet node browser-less.
#   5. Enables + starts the service. Edge is now listening on :9222.
#
# Re-run safe (idempotent): all writes are guarded. Running it again on an
# already-configured node updates the systemd unit but leaves the profile
# intact (so persistent logins survive a re-deploy).
#
# Usage:
#   sudo bash scripts/setup-edge-cdp.sh                 # run as root
#   sudo OPENTEDDY_USER=admin bash scripts/setup-edge-cdp.sh  # override user
#
# After setup:
#   1. Capture a logged-in storage_state.json once on any machine with a
#      display (see scripts/capture-edge-state.md for the playwright
#      codegen recipe) and `scp` it to /var/lib/openteddy/storage_state.json
#      on each fleet node.
#   2. OpenTeddy's chrome_attached_tool auto-detects + injects those
#      cookies on every attach. Each fleet node now scrapes X as you.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
# Target user — defaults to the SUDO_USER who invoked us, falling back to
# the current $USER, falling back to "openteddy". Override with OPENTEDDY_USER=.
TARGET_USER="${OPENTEDDY_USER:-${SUDO_USER:-${USER:-openteddy}}}"
if ! id -u "${TARGET_USER}" >/dev/null 2>&1; then
  echo "✗ Target user '${TARGET_USER}' does not exist on this host."
  echo "  Set OPENTEDDY_USER=<existing-user> and re-run."
  exit 1
fi

PROFILE_DIR="/var/lib/openteddy/edge-profile"
STATE_FILE="/var/lib/openteddy/storage_state.json"
SERVICE_FILE="/etc/systemd/system/openteddy-cdp.service"
LISTEN_PORT="${OPENTEDDY_CDP_PORT:-9222}"

# Refuse to run unless we're root — needs apt + systemctl + /etc writes.
if [ "$(id -u)" -ne 0 ]; then
  echo "✗ Run with sudo: sudo bash $0"
  exit 1
fi

echo "▶ OpenTeddy CDP setup"
echo "    target user : ${TARGET_USER}"
echo "    profile dir : ${PROFILE_DIR}"
echo "    listen port : ${LISTEN_PORT}"
echo ""

# ── 1. Microsoft apt repo ────────────────────────────────────────────────────
# Always (re)write the list and keyring so a previously-added entry with
# the wrong arch (e.g. arch=amd64 on an arm64 host, which apt-get install
# then "fixes" by reporting `Unable to locate package microsoft-edge-stable`)
# gets cleanly overwritten. Cheap operation; safer than an existence check.
echo "▶ (Re)writing Microsoft apt repo + GPG key…"
curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
  | gpg --dearmor --yes -o /usr/share/keyrings/microsoft.gpg
ARCH="$(dpkg --print-architecture)"
cat > /etc/apt/sources.list.d/microsoft-edge.list <<EOF
deb [arch=${ARCH} signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/edge stable main
EOF
echo "  ✓ Repo written (arch=${ARCH})"

# ── 2. Install Edge ──────────────────────────────────────────────────────────
echo "▶ Installing microsoft-edge-stable (apt update + install)…"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  microsoft-edge-stable \
  >/dev/null
EDGE_VERSION="$(/usr/bin/microsoft-edge --version 2>/dev/null | head -1 || echo '?')"
echo "  ✓ Installed: ${EDGE_VERSION}"

# ── 3. Profile directory ─────────────────────────────────────────────────────
echo "▶ Provisioning profile dir at ${PROFILE_DIR}…"
mkdir -p "${PROFILE_DIR}"
chown -R "${TARGET_USER}:${TARGET_USER}" "/var/lib/openteddy"
chmod 700 "${PROFILE_DIR}"
echo "  ✓ Profile dir ready (owner: ${TARGET_USER}, mode 700)"

# ── 4. Storage-state hint ────────────────────────────────────────────────────
if [ -f "${STATE_FILE}" ]; then
  CNT="$(python3 -c 'import json,sys; d=json.load(open("'${STATE_FILE}'")); print(len(d.get("cookies",[])))' 2>/dev/null || echo '?')"
  echo "▶ Found ${STATE_FILE} (${CNT} cookies) — will be auto-injected."
else
  echo "▶ No ${STATE_FILE} yet. After setup, capture one on a workstation:"
  echo ""
  echo "    pip install playwright"
  echo "    playwright install chromium"
  echo "    playwright codegen https://x.com/login --save-storage=storage_state.json"
  echo "    # ↑ log in once in the window that pops up, close it, then:"
  echo "    scp storage_state.json admin@<this-host>:/tmp/"
  echo "    sudo mv /tmp/storage_state.json ${STATE_FILE}"
  echo "    sudo chown ${TARGET_USER}:${TARGET_USER} ${STATE_FILE}"
  echo "    sudo chmod 600 ${STATE_FILE}"
  echo ""
fi

# ── 5. systemd unit ──────────────────────────────────────────────────────────
echo "▶ Writing systemd unit to ${SERVICE_FILE}…"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=OpenTeddy CDP browser (Microsoft Edge headless on 127.0.0.1:${LISTEN_PORT})
Documentation=https://github.com/m31527/OpenTeddy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${TARGET_USER}
Restart=always
RestartSec=5
# 127.0.0.1 only — never expose CDP to the LAN. Anyone who can reach 9222
# gets full DevTools access to every cookie / page in this profile, which
# includes the operator's X / LinkedIn / etc. sessions. Treat it like the
# Docker daemon socket: localhost trust only.
ExecStart=/usr/bin/microsoft-edge \\
  --headless=new \\
  --remote-debugging-port=${LISTEN_PORT} \\
  --remote-debugging-address=127.0.0.1 \\
  --user-data-dir=${PROFILE_DIR} \\
  --no-sandbox \\
  --disable-gpu \\
  --disable-features=UseOzonePlatform \\
  --window-size=1440,900 \\
  about:blank

# Some hardening — Edge doesn't need to write outside its profile dir.
ProtectSystem=full
ProtectHome=no
PrivateTmp=yes
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
EOF
echo "  ✓ Unit written"

# ── 6. Enable + start ────────────────────────────────────────────────────────
echo "▶ Reloading systemd + (re)starting service…"
systemctl daemon-reload
systemctl enable --now openteddy-cdp.service

# Give Edge a few seconds to bind.
sleep 3

# ── 7. Verify ────────────────────────────────────────────────────────────────
echo ""
echo "▶ Verifying…"
if curl -sSf "http://127.0.0.1:${LISTEN_PORT}/json/version" >/dev/null 2>&1; then
  BROWSER="$(curl -s http://127.0.0.1:${LISTEN_PORT}/json/version \
              | python3 -c 'import json,sys; print(json.load(sys.stdin).get("Browser",""))' 2>/dev/null)"
  echo "  ✓ CDP up on 127.0.0.1:${LISTEN_PORT} (${BROWSER})"
else
  echo "  ✗ CDP NOT responding. Diagnose:"
  echo "      systemctl status openteddy-cdp.service"
  echo "      journalctl -u openteddy-cdp.service -n 50"
  exit 1
fi

echo ""
echo "✅ Setup complete."
echo ""
echo "Service control:"
echo "    sudo systemctl status   openteddy-cdp.service"
echo "    sudo systemctl restart  openteddy-cdp.service"
echo "    sudo journalctl -u openteddy-cdp.service -f"
echo ""
echo "Test from OpenTeddy (as ${TARGET_USER}):"
echo "    cd ~/OpenTeddy && .venv/bin/python -c \\"
echo "      \"import asyncio, sys; sys.path.insert(0,'.'); \\"
echo "       from tools.chrome_attached_tool import chrome_attach_check; \\"
echo "       print(asyncio.run(chrome_attach_check()))\""
