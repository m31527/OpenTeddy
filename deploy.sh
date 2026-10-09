#!/usr/bin/env bash
# deploy.sh — ship what's on GitHub main to a machine that runs OpenTeddy.
#
#   ./deploy.sh home                    # m31527@192.168.50.86:/home/m31527/OpenTeddy
#   ./deploy.sh user@host[:/path]       # any other machine (default path ~/OpenTeddy)
#   ./deploy.sh home --system           # the service was installed with --system
#   ./deploy.sh home --restart          # restart even when nothing changed
#   ./deploy.sh home --autostash        # target has local edits in files the update touches
#
# The target pulls from GitHub, so this first checks that local main has
# been pushed. On the target: git pull --ff-only, pip install if
# requirements.txt changed, ./openteddy service restart (waits for
# /health). Local edits on the target are kept; git refuses a pull that
# would overwrite them, and --autostash sets them aside and re-applies
# them. A missing checkout is cloned and the script stops there: a first
# install needs ./install.sh run by hand.
#
# Add a target: one more line in the case block below.
set -euo pipefail

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage 1
case "$1" in
  -h|--help) usage 0 ;;
  home) HOST="m31527@192.168.50.86"; REMOTE_DIR="/home/m31527/OpenTeddy" ;;
  *@*)
    HOST="${1%%:*}"
    REMOTE_DIR="${1#*:}"
    [[ "$REMOTE_DIR" == "$1" ]] && REMOTE_DIR="OpenTeddy"   # no ":path" given
    ;;
  *) echo "unknown target '$1' (use home, or user@host[:/path])" >&2; exit 1 ;;
esac
shift
SYSTEM=0; RESTART=0; AUTOSTASH=0
for arg in "$@"; do
  case "$arg" in
    --system) SYSTEM=1 ;;
    --restart) RESTART=1 ;;
    --autostash) AUTOSTASH=1 ;;
    *) echo "unknown option $arg" >&2; exit 1 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
SSH=(ssh -o ConnectTimeout=8 "$HOST")

# ── local checks: the target can only pull what GitHub has ──────────────────
branch="$(git rev-parse --abbrev-ref HEAD)"
[[ "$branch" == "main" ]] || echo "⚠ you're on '$branch'; the target deploys origin/main"
git fetch -q origin main
ahead="$(git rev-list --count origin/main..main)"
if [[ "$ahead" -gt 0 ]]; then
  echo "✗ local main has $ahead commit(s) not on GitHub — push first, then deploy:" >&2
  git log --oneline origin/main..main | sed 's/^/    /' >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "⚠ uncommitted local changes are NOT part of this deploy"
fi
want="$(git rev-parse --short origin/main)"
echo "→ deploying origin/main ($want: $(git log -1 --format=%s origin/main)) to $HOST:$REMOTE_DIR"

# ── remote: clone if missing, otherwise pull → deps → restart → health ──────
# `bash -s` reads the script from stdin; arguments arrive as $1, $2, …
"${SSH[@]}" bash -s -- "$REMOTE_DIR" "$(git remote get-url origin)" "$SYSTEM" "$RESTART" "$AUTOSTASH" <<'REMOTE'
set -euo pipefail
dir="$1"; url="$2"; system="$3"; restart="$4"; autostash="$5"
if [[ ! -d "$dir/.git" ]]; then
  echo "• no checkout at $dir — cloning"
  git clone -q "$url" "$dir"
  echo "✓ cloned. First install on this machine:"
  echo "    cd $dir && ./install.sh && ./openteddy service install --host 0.0.0.0"
  exit 0
fi
cd "$dir"
dirty="$(git status --porcelain --untracked-files=no)"
if [[ -n "$dirty" ]]; then
  echo "• $(hostname) has local edits (kept):"
  echo "$dirty" | sed 's/^/    /'
fi
before="$(git rev-parse HEAD)"
pull=(git pull --ff-only -q)
[[ "$autostash" == 1 ]] && pull+=(--autostash)
if ! "${pull[@]}"; then
  echo "✗ git pull refused on $(hostname)." >&2
  [[ -n "$dirty" && "$autostash" != 1 ]] && \
    echo "  The update touches a file edited there. Re-run with --autostash to set the edits aside and re-apply them." >&2
  exit 1
fi
if [[ -n "$(git diff --name-only --diff-filter=U)" ]]; then
  echo "✗ re-applying local edits conflicted; fix on $(hostname) (git status), service NOT restarted" >&2
  exit 1
fi
after="$(git rev-parse HEAD)"
if [[ "$before" == "$after" ]]; then
  echo "✓ already up to date ($(git rev-parse --short HEAD))"
  [[ "$restart" == 1 ]] || exit 0
else
  echo "✓ $(git rev-parse --short "$before") → $(git rev-parse --short "$after")  ($(git diff --name-only "$before" "$after" | wc -l | tr -d ' ') files)"
  if git diff --name-only "$before" "$after" | grep -qx requirements.txt && [[ -x .venv/bin/pip ]]; then
    echo "• requirements.txt changed → installing"
    .venv/bin/pip install -q -r requirements.txt
  fi
fi
svc=(./openteddy service restart)
[[ "$system" == 1 ]] && svc+=(--system)
echo "↻ restarting service"
if ! "${svc[@]}"; then
  echo "✗ restart or health check failed. If the service isn't installed yet:" >&2
  echo "    cd $dir && ./openteddy service install --host 0.0.0.0" >&2
  echo "  logs: ./openteddy service logs" >&2
  exit 1
fi
echo "✓ $(hostname) is running $(git rev-parse --short HEAD)"
REMOTE
