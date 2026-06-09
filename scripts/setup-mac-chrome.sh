#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenTeddy CDP browser setup — macOS
# ─────────────────────────────────────────────────────────────────────────────
#
# Sister script to setup-edge-cdp.sh (Linux). Same idea: install a managed
# Chromium-based browser that listens on 127.0.0.1:9222 for CDP attaches,
# so OpenTeddy's chrome_attached_tool / x_search / threads_search work
# out of the box on macOS desktop installs.
#
# Mac specifics:
#
#   - Google Chrome is the default. Has native Apple Silicon AND Intel
#     builds — no fallback dance needed (unlike Linux ARM64 which has
#     no Chrome). Falls back to Brave / Edge / Chromium / Chromium-
#     equivalent if Chrome is missing.
#
#   - SEPARATE profile dir under ~/Library/Application Support/OpenTeddy.
#     Reason: macOS users are almost always actively using their normal
#     Chrome day-to-day. Sharing their profile would mean we'd need to
#     kill their working browser every time OpenTeddy wants to scrape,
#     which is unacceptable. With a separate profile, OpenTeddy's
#     scraping Chrome runs alongside the user's normal one — different
#     window, different cookies, different bookmarks. Trade-off: the
#     operator has to log in to X / Threads / etc. separately inside
#     the OpenTeddy profile (one-time, via login-mac-helper.sh).
#
#   - LaunchAgent is the macOS systemd equivalent. We write
#     ~/Library/LaunchAgents/net.openteddy.cdp.plist with KeepAlive=true
#     + RunAtLoad=true so the CDP browser survives reboots + crashes,
#     same lifecycle as Linux's openteddy-cdp.service.
#
#   - Headless by default. The CDP browser doesn't need to be visible
#     for scraping — and a visible window in the user's Dock would be
#     constant noise. For login, scripts/login-mac-helper.sh swaps it
#     for a headful instance, then back.
#
# Usage:
#   bash scripts/setup-mac-chrome.sh                  # install + start
#   bash scripts/setup-mac-chrome.sh --uninstall      # remove LaunchAgent
#
# Re-run safe (idempotent).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

UNINSTALL=false
for arg in "$@"; do
  case "${arg}" in
    --uninstall|-u) UNINSTALL=true ;;
    -h|--help) sed -n '4,40p' "$0"; exit 0 ;;
  esac
done

LISTEN_PORT="${OPENTEDDY_CDP_PORT:-9222}"
OPENTEDDY_HOME="${HOME}/Library/Application Support/OpenTeddy"
PROFILE_DIR="${OPENTEDDY_HOME}/Chrome-CDP"
LAUNCH_AGENT_LABEL="net.openteddy.cdp"
LAUNCH_AGENT_PLIST="${HOME}/Library/LaunchAgents/${LAUNCH_AGENT_LABEL}.plist"
LOG_PATH="${OPENTEDDY_HOME}/cdp.log"

# ── Uninstall path ───────────────────────────────────────────────────────────
if ${UNINSTALL}; then
  echo "▶ Uninstalling OpenTeddy CDP LaunchAgent…"
  launchctl unload "${LAUNCH_AGENT_PLIST}" 2>/dev/null || true
  rm -f "${LAUNCH_AGENT_PLIST}"
  pkill -f "remote-debugging-port=${LISTEN_PORT}" 2>/dev/null || true
  echo "  ✓ LaunchAgent removed."
  echo "  ℹ Profile dir kept at ${PROFILE_DIR} (delete manually if you also"
  echo "    want to discard logged-in sessions)."
  exit 0
fi

# ── Detect browser ───────────────────────────────────────────────────────────
# Priority: Chrome → Brave → Edge → Chromium. Chrome first because most
# macOS users already have it; the alternatives are only relevant for
# users with strong privacy / corporate preferences.
declare -a CANDIDATES=(
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome|Chrome"
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser|Brave"
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge|Edge"
  "/Applications/Chromium.app/Contents/MacOS/Chromium|Chromium"
)

BROWSER_BIN=""
BROWSER_LABEL=""
for entry in "${CANDIDATES[@]}"; do
  path="${entry%|*}"
  label="${entry#*|}"
  if [ -x "${path}" ]; then
    BROWSER_BIN="${path}"
    BROWSER_LABEL="${label}"
    break
  fi
done

if [ -z "${BROWSER_BIN}" ]; then
  cat <<'EOF'
✗ No Chromium-based browser found in /Applications/.

Install Chrome (the default OpenTeddy expects on macOS):
  https://www.google.com/chrome/

Or any of these — OpenTeddy speaks vendor-neutral CDP, so all work:
  Brave    https://brave.com/download/
  Edge     https://www.microsoft.com/edge
  Chromium https://www.chromium.org/getting-involved/download-chromium/

Then re-run this script.
EOF
  exit 1
fi

echo "▶ OpenTeddy macOS CDP setup"
echo "    browser     : ${BROWSER_LABEL} (${BROWSER_BIN})"
echo "    profile dir : ${PROFILE_DIR}"
echo "    listen port : 127.0.0.1:${LISTEN_PORT}"
echo "    log         : ${LOG_PATH}"
echo ""

# ── Provision profile + log dirs ─────────────────────────────────────────────
mkdir -p "${PROFILE_DIR}"
mkdir -p "$(dirname "${LOG_PATH}")"

# ── Write the LaunchAgent ────────────────────────────────────────────────────
# KeepAlive=true → if Chrome dies (rare), launchd respawns it within a few
# seconds. Mirrors Linux's systemd Restart=always behaviour.
# RunAtLoad=true → starts on first login after install + every subsequent
# user-login boot. Mirrors systemctl enable + start.
echo "▶ Writing LaunchAgent to ${LAUNCH_AGENT_PLIST}…"
mkdir -p "$(dirname "${LAUNCH_AGENT_PLIST}")"
cat > "${LAUNCH_AGENT_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LAUNCH_AGENT_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${BROWSER_BIN}</string>
        <string>--headless=new</string>
        <string>--remote-debugging-port=${LISTEN_PORT}</string>
        <string>--remote-debugging-address=127.0.0.1</string>
        <string>--user-data-dir=${PROFILE_DIR}</string>
        <string>--no-first-run</string>
        <string>--no-default-browser-check</string>
        <string>--disable-features=UseOzonePlatform</string>
        <string>--window-size=1440,900</string>
        <string>about:blank</string>
    </array>

    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>

    <key>StandardOutPath</key><string>${LOG_PATH}</string>
    <key>StandardErrorPath</key><string>${LOG_PATH}</string>

    <key>ProcessType</key><string>Background</string>
</dict>
</plist>
EOF
echo "  ✓ LaunchAgent written"

# ── (Re)load the LaunchAgent ─────────────────────────────────────────────────
echo "▶ (Re)loading LaunchAgent…"
launchctl unload "${LAUNCH_AGENT_PLIST}" 2>/dev/null || true
launchctl load -w "${LAUNCH_AGENT_PLIST}"
echo "  ✓ Loaded"

# ── Verify ───────────────────────────────────────────────────────────────────
sleep 4
echo ""
echo "▶ Verifying…"
if curl -sSf "http://127.0.0.1:${LISTEN_PORT}/json/version" >/dev/null 2>&1; then
  BROWSER_VER="$(curl -s http://127.0.0.1:${LISTEN_PORT}/json/version \
                  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("Browser",""))' 2>/dev/null)"
  echo "  ✓ CDP up on 127.0.0.1:${LISTEN_PORT} (${BROWSER_VER})"
else
  echo "  ✗ CDP NOT responding yet. Could be cold-start delay — wait 10s and check:"
  echo "      curl -sS http://127.0.0.1:${LISTEN_PORT}/json/version"
  echo "  Tail the log for errors:"
  echo "      tail -50 \"${LOG_PATH}\""
  exit 1
fi

echo ""
echo "✅ Setup complete."
echo ""
echo "Lifecycle:"
echo "    launchctl list ${LAUNCH_AGENT_LABEL}     # is it running?"
echo "    launchctl unload ${LAUNCH_AGENT_PLIST}   # stop it"
echo "    launchctl load   ${LAUNCH_AGENT_PLIST}   # start it"
echo "    tail -f \"${LOG_PATH}\"                  # live log"
echo ""
echo "Login to scraping sites (X / Threads / LinkedIn / corp wiki / etc.):"
echo "    bash scripts/login-mac-helper.sh"
echo ""
echo "Test from OpenTeddy:"
echo "    cd ~/OpenTeddy && .venv/bin/python -c \\"
echo "      'import asyncio,sys; sys.path.insert(0,\".\");"
echo "       from tools.chrome_attached_tool import chrome_attach_check;"
echo "       import json; print(json.dumps(asyncio.run("
echo "         chrome_attach_check()), indent=2))'"
echo ""
echo "Uninstall:"
echo "    bash scripts/setup-mac-chrome.sh --uninstall"
