#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Install the vision model OpenTeddy uses to see pages (page_look, the
# browser extension's icon / canvas lookups).
# ─────────────────────────────────────────────────────────────────────────────
#
#   ./scripts/install-vision-model.sh              # the model Settings names
#   ./scripts/install-vision-model.sh qwen3-vl:8b-instruct
#
# From another machine, without copying anything there first:
#   ssh user@box 'bash -s' < scripts/install-vision-model.sh
#
# What it does:
#   1. Picks the model: the argument, else OpenTeddy's Settings → Vision
#      Model, else qwen3-vl:8b-instruct.
#   2. Pulls it with Ollama unless it's already there (~6 GB, once).
#   3. Shows it a small red square and checks it answers, so you know it
#      actually works, not just that it downloaded.
#   4. Says whether OpenTeddy's setting points at it.
#
# No sudo, no restart, and safe to run again.
#
# Env: OLLAMA_HOST (default http://127.0.0.1:11434),
#      OPENTEDDY_URL (default http://127.0.0.1:8000).
set -euo pipefail

DEFAULT_MODEL="qwen3-vl:8b-instruct"
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"
[[ "$OLLAMA_URL" == http* ]] || OLLAMA_URL="http://$OLLAMA_URL"
OPENTEDDY_URL="${OPENTEDDY_URL:-http://127.0.0.1:8000}"

ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
info() { printf '  • %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
  -h|--help) sed -n '2,25p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

command -v curl >/dev/null    || die "curl is missing"
command -v python3 >/dev/null || die "python3 is missing"
command -v ollama >/dev/null  || die "Ollama isn't installed — https://ollama.com/download"
curl -sf -m 5 "$OLLAMA_URL/api/version" >/dev/null \
  || die "Ollama isn't answering at $OLLAMA_URL — start it (systemctl start ollama, or open Ollama.app)"

# ── 1. Which model ───────────────────────────────────────────────────────────
# What OpenTeddy is set to use. "-" = OpenTeddy isn't running here (or
# didn't answer); "" = the setting is empty, which turns vision off.
setting="$(curl -sf -m 5 "$OPENTEDDY_URL/settings" 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin)["data"]["vision_model"]["value"] or "")
except Exception:
    print("-")
' 2>/dev/null)" || true    # curl failing is fine: python already printed "-"

model="${1:-}"
if [[ -z "$model" ]]; then
  if [[ "$setting" != "-" && -n "$setting" ]]; then model="$setting"; else model="$DEFAULT_MODEL"; fi
fi
printf '\n\033[1m▶ Vision model: %s\033[0m\n' "$model"

# ── 2. Pull ──────────────────────────────────────────────────────────────────
have() {
  curl -sf -m 10 "$OLLAMA_URL/api/tags" | python3 -c '
import json, sys
want = sys.argv[1]
names = {m["name"] for m in json.load(sys.stdin).get("models", [])}
sys.exit(0 if want in names or want + ":latest" in names else 1)
' "$1"
}
if have "$model"; then
  ok "already installed"
else
  info "downloading with ollama pull (about 6 GB for the 8b; only this once)…"
  # </dev/null: under `ssh … 'bash -s' < script` stdin IS the rest of this script.
  OLLAMA_HOST="$OLLAMA_URL" ollama pull "$model" </dev/null || die "ollama pull $model failed — check the name and the network, then run this again"
  have "$model" || die "pull finished but Ollama doesn't list $model"
  ok "installed"
fi

# ── 3. Does it see? ──────────────────────────────────────────────────────────
info "test: showing it a red square (the first load takes a while)…"
answer="$(python3 - "$OLLAMA_URL" "$model" <<'PY'
import base64, json, struct, sys, time, urllib.request, zlib

url, model = sys.argv[1], sys.argv[2]
w = h = 64
row = b"\x00" + b"\xd0\x20\x20" * w                  # filter byte + RGB pixels
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(row * h))
       + chunk(b"IEND", b""))
body = json.dumps({
    "model": model, "stream": False,
    "prompt": "What color is this image? Answer with one English word.",
    "images": [base64.b64encode(png).decode()],
    "options": {"num_predict": 16, "temperature": 0},
}).encode()
t0 = time.time()
try:
    req = urllib.request.Request(url + "/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
except Exception as exc:                              # noqa: BLE001
    print(f"ERROR {exc}")
    sys.exit(0)
print(f"{(out.get('response') or '').strip()[:80]}|{time.time() - t0:.1f}")
PY
)"
case "$answer" in
  ERROR*) die "the model didn't answer: ${answer#ERROR }" ;;
esac
said="${answer%|*}"; secs="${answer##*|}"
if [[ -z "$said" ]]; then
  die "the model answered with nothing — if this is a thinking build, use the -instruct one ($DEFAULT_MODEL)"
elif printf '%s' "$said" | grep -qi red; then
  ok "it sees: \"$said\" (${secs} s)"
else
  warn "it answered \"$said\" (${secs} s) — it runs, but not the expected \"red\""
fi

# ── 4. OpenTeddy's setting ───────────────────────────────────────────────────
if [[ "$setting" == "-" ]]; then
  info "OpenTeddy isn't answering at $OPENTEDDY_URL — when it runs, check Settings → Vision Model is \"$model\""
elif [[ "$setting" == "$model" ]]; then
  ok "OpenTeddy's Settings → Vision Model is already \"$model\" — nothing to restart"
elif [[ -z "$setting" ]]; then
  warn "Settings → Vision Model is empty (vision is off). Set it to \"$model\" to use it"
else
  warn "Settings → Vision Model is \"$setting\", not \"$model\". Change it there if you want this one"
fi
echo
