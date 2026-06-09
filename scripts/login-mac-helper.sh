#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenTeddy CDP login helper — macOS
# ─────────────────────────────────────────────────────────────────────────────
#
# Mac counterpart to login-helper.sh (Linux). Same pattern, different
# service manager:
#
#   1. Stop the LaunchAgent (releases the profile dir's ProcessSingleton).
#   2. Launch the same browser WITH a visible window (no --headless) but
#      pointing at the same --user-data-dir, so cookies the operator sets
#      land in the profile the headless service uses.
#   3. When the browser window closes (or operator Ctrl-Cs), restart the
#      LaunchAgent. Cookies persist across the swap.
#
# Use whenever a site's cookies expire (X every ~30 days, LinkedIn ~90),
# or for the very first login on a fresh install.
#
# Usage:
#   bash scripts/login-mac-helper.sh
#
# Pre-opens the X login page in the new window; navigate elsewhere if you
# want to log in to a different site.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

LISTEN_PORT="${OPENTEDDY_CDP_PORT:-9222}"
OPENTEDDY_HOME="${HOME}/Library/Application Support/OpenTeddy"
PROFILE_DIR="${OPENTEDDY_HOME}/Chrome-CDP"
LAUNCH_AGENT_LABEL="net.openteddy.cdp"
LAUNCH_AGENT_PLIST="${HOME}/Library/LaunchAgents/${LAUNCH_AGENT_LABEL}.plist"

if [ ! -f "${LAUNCH_AGENT_PLIST}" ]; then
  cat <<EOF
✗ The CDP LaunchAgent isn't installed yet.
  Run: bash scripts/setup-mac-chrome.sh
EOF
  exit 1
fi

# Detect the browser the LaunchAgent uses (so we open the same one for
# login — different browsers don't share profile dirs).
BROWSER_BIN="$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' "${LAUNCH_AGENT_PLIST}" 2>/dev/null || true)"
if [ -z "${BROWSER_BIN}" ] || [ ! -x "${BROWSER_BIN}" ]; then
  echo "✗ Couldn't parse the browser path from ${LAUNCH_AGENT_PLIST}."
  echo "  Re-run: bash scripts/setup-mac-chrome.sh"
  exit 1
fi

BROWSER_LABEL="$(basename "${BROWSER_BIN}")"

echo "▶ OpenTeddy macOS login helper"
echo "    browser     : ${BROWSER_LABEL}"
echo "    profile dir : ${PROFILE_DIR}"
echo ""

# ── 1. Pause the headless LaunchAgent ────────────────────────────────────────
if launchctl list "${LAUNCH_AGENT_LABEL}" >/dev/null 2>&1; then
  echo "▶ Pausing headless ${LAUNCH_AGENT_LABEL}…"
  launchctl unload "${LAUNCH_AGENT_PLIST}"
fi

# Belt-and-braces: kill any stray browser on the debug port + clear the
# profile dir's ProcessSingleton lock. Otherwise the new headful launch
# below dies with "another instance is using this profile".
pkill -f "remote-debugging-port=${LISTEN_PORT}" 2>/dev/null || true
sleep 1
rm -f "${PROFILE_DIR}/SingletonLock" \
      "${PROFILE_DIR}/SingletonCookie" \
      "${PROFILE_DIR}/SingletonSocket" 2>/dev/null || true

# ── 2. Restart-headless trap (fires on close / Ctrl-C / hangup) ─────────────
_restart_headless() {
  echo ""
  echo "▶ Restarting headless LaunchAgent…"
  pkill -f "remote-debugging-port=${LISTEN_PORT}" 2>/dev/null || true
  sleep 1
  rm -f "${PROFILE_DIR}/SingletonLock" \
        "${PROFILE_DIR}/SingletonCookie" \
        "${PROFILE_DIR}/SingletonSocket" 2>/dev/null || true
  launchctl load -w "${LAUNCH_AGENT_PLIST}"
  sleep 3
  if curl -sSf "http://127.0.0.1:${LISTEN_PORT}/json/version" >/dev/null 2>&1; then
    echo "  ✓ Headless CDP back up on 127.0.0.1:${LISTEN_PORT}"
    echo ""
    echo "Verify the new login from OpenTeddy:"
    echo "    cd ~/OpenTeddy && .venv/bin/python -c \\"
    echo "      'import asyncio,sys; sys.path.insert(0,\".\");"
    echo "       from tools.chrome_attached_tool import x_search;"
    echo "       import json; print(json.dumps(asyncio.run("
    echo "         x_search(query=\"黴菌\", top_n=3)),"
    echo "         indent=2, ensure_ascii=False))'"
  else
    echo "  ⚠ LaunchAgent didn't come back. Check:"
    echo "      tail -50 \"${OPENTEDDY_HOME}/cdp.log\""
  fi
}
trap _restart_headless EXIT INT TERM HUP

# ── 3. Launch headful browser pointed at the same profile ────────────────────
cat <<EOF
─────────────────────────────────────────────────────────────────────
  ▶ Opening ${BROWSER_LABEL} now. A window will appear. Inside it:

      1. Open the site(s) you need to log in to:
           https://x.com/login
           https://www.threads.com/login
           https://www.linkedin.com/login

      2. Sign in normally (password + 2FA / OAuth).

      3. CLOSE THE BROWSER WINDOW when done. This script will
         automatically swap back to the headless LaunchAgent.

  Cookies will be saved to:
      ${PROFILE_DIR}
─────────────────────────────────────────────────────────────────────
EOF

# Foreground — script blocks until the browser window is closed.
"${BROWSER_BIN}" \
  --user-data-dir="${PROFILE_DIR}" \
  --remote-debugging-port="${LISTEN_PORT}" \
  --no-first-run \
  --no-default-browser-check \
  about:blank \
  https://x.com/login

# trap fires here on clean exit.
