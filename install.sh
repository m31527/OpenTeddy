#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OpenTeddy one-line installer
#
# Usage:
#   curl -fsSL https://openteddy.net/install | bash
#
# With flags:
#   curl -fsSL https://...install.sh | bash -s -- --dir ~/projects/openteddy --skip-models
#
# Idempotent — re-run to update an existing install (git pull + pip refresh).
#
# Cautious by design:
#   - never sudo anything; all files scope to $HOME/OpenTeddy (or --dir)
#   - never fetches a secondary script; the whole flow is in this file so
#     a paranoid user can `curl -O` and audit before running
#   - --dry-run prints what would happen without changing anything
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────────
REPO_URL="https://github.com/m31527/OpenTeddy.git"
DEFAULT_DIR="$HOME/OpenTeddy"
MIN_PY_MAJOR=3
MIN_PY_MINOR=10
# Defaults match config.py: gemma_model + qwen_model. Pulling them so a
# fresh install can run tasks out-of-the-box without the user having to
# guess which models to download.
DEFAULT_MODELS=("gemma3:4b" "qwen2.5:3b")

# ── Flags ─────────────────────────────────────────────────────────────────
INSTALL_DIR="${OPENTEDDY_DIR:-$DEFAULT_DIR}"
FORCE=0
SKIP_MODELS=0
DRY_RUN=0
DEBUG=0

# ── Color output (auto-off when not a tty) ────────────────────────────────
if [[ -t 1 ]]; then
  CYAN='\033[36m'; GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'
  BOLD='\033[1m';  RESET='\033[0m'
else
  CYAN=''; GREEN=''; YELLOW=''; RED=''; BOLD=''; RESET=''
fi
bold()  { printf "${BOLD}%s${RESET}\n" "$*"; }
info()  { printf "  ${CYAN}▸${RESET} %s\n" "$*"; }
ok()    { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn()  { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }
die()   { printf "  ${RED}✗${RESET} %s\n" "$*" >&2; exit 1; }

# ── Arg parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)         INSTALL_DIR="$2"; shift ;;
    --force)       FORCE=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --dry-run)     DRY_RUN=1 ;;
    --debug)       DEBUG=1; set -x ;;
    -h|--help)
      cat <<EOF
OpenTeddy one-line installer.

Quick start:
  curl -fsSL https://openteddy.net/install | bash

Options (pass via: ... | bash -s -- --flag):
  --dir <path>      Install location (default: \$HOME/OpenTeddy)
  --force           Overwrite an existing non-git directory
  --skip-models     Don't pull default Ollama models (gemma3:4b, qwen2.5:3b)
  --dry-run         Print what would happen, change nothing
  --debug           Verbose shell trace (set -x)
  -h, --help        This message

Environment:
  OPENTEDDY_DIR     Alternative to --dir (same default)

The installer is idempotent: re-running on an existing install updates
to latest main, refreshes pip deps, and skips already-pulled models.
EOF
      exit 0 ;;
    *) die "unknown flag: $1 (try --help)" ;;
  esac
  shift
done

# ── Pre-flight checks ─────────────────────────────────────────────────────
bold "▶ Pre-flight"

# Python 3.10+ — chromadb async + sqlalchemy[asyncio] need a recent runtime.
command -v python3 >/dev/null \
  || die "Python 3 not found. Install from https://www.python.org/downloads/ or via brew install python."
PYVER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
PY_OK=$(python3 -c "import sys; print(1 if sys.version_info >= ($MIN_PY_MAJOR, $MIN_PY_MINOR) else 0)")
[[ "$PY_OK" = "1" ]] \
  || die "Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ required (you have $PYVER). Upgrade via pyenv / brew install python@3.12."
ok "Python $PYVER"

command -v git  >/dev/null || die "git not found. Install via brew install git / apt-get install git."
ok "git $(git --version | awk '{print $3}')"

# curl is what got us here, but defensive: some shells run curl from /bin
# without putting it on PATH for child processes.
command -v curl >/dev/null || die "curl not found in PATH."

case "$(uname -s)" in
  Darwin) OS_LABEL="macOS $(sw_vers -productVersion 2>/dev/null)" ;;
  Linux)  OS_LABEL="Linux $(uname -r)" ;;
  *)      OS_LABEL="$(uname -s)" ;;
esac
ok "$OS_LABEL"

# ── Ollama presence (warn-but-continue) ───────────────────────────────────
# Local-model use needs Ollama. If absent, OpenTeddy still installs cleanly
# and the user can use it in cloud-only mode (Anthropic / OpenAI / etc.
# keys configured via Settings UI) — so we don't fail the install here.
bold "▶ Ollama"
if command -v ollama >/dev/null; then
  ok "$(ollama --version 2>&1 | head -1)"
  if curl -fsSL -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    ok "daemon responding on 127.0.0.1:11434"
  else
    warn "daemon not running — start Ollama.app or run 'ollama serve' before using local models"
  fi
else
  warn "Ollama not installed — local-model features will be disabled"
  info "install later from https://ollama.com/download (or brew install ollama)"
  info "OpenTeddy will still install; cloud LLMs (Claude / OpenAI / Gemini /"
  info "Deepseek / OpenRouter) work without Ollama via Settings → Cloud LLM Provider"
fi

# ── Install location ──────────────────────────────────────────────────────
bold "▶ Install location"
info "target: $INSTALL_DIR"

if [[ -e "$INSTALL_DIR" ]]; then
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "existing OpenTeddy clone found — will update via git pull"
  elif [[ -z "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]]; then
    info "directory exists but empty — proceeding"
  elif [[ "$FORCE" = "1" ]]; then
    warn "non-empty directory — --force passed, will overwrite"
    [[ "$DRY_RUN" = "0" ]] && rm -rf "$INSTALL_DIR"
  else
    die "'$INSTALL_DIR' exists and is non-empty. Either delete it, pass --force, or pick a different path with --dir <other-path>."
  fi
fi

# ── Clone / pull source ───────────────────────────────────────────────────
bold "▶ Source"
if [[ "$DRY_RUN" = "1" ]]; then
  info "[dry-run] would git-clone $REPO_URL → $INSTALL_DIR"
elif [[ -d "$INSTALL_DIR/.git" ]]; then
  # Update path. --ff-only refuses to merge — if the user made local
  # changes we want to halt loudly rather than silently rebasing them.
  ( cd "$INSTALL_DIR" && git pull --ff-only --depth 1 origin main ) \
    || die "git pull failed. Resolve conflicts manually in $INSTALL_DIR, or rm -rf it and re-run install.sh."
  ok "updated to $(cd "$INSTALL_DIR" && git rev-parse --short HEAD)"
else
  # --depth 1: shallow clone, ~3x faster + smaller download for users
  # who just want to run it (vs hack on it).
  git clone --depth 1 "$REPO_URL" "$INSTALL_DIR" \
    || die "git clone failed. Check network + GitHub access."
  ok "cloned to $INSTALL_DIR ($(cd "$INSTALL_DIR" && git rev-parse --short HEAD))"
fi

# ── Python venv + deps ────────────────────────────────────────────────────
bold "▶ Python environment"
if [[ "$DRY_RUN" = "1" ]]; then
  info "[dry-run] would create .venv + pip install -r requirements.txt"
else
  cd "$INSTALL_DIR"
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    ok "created .venv"
  else
    ok ".venv exists — reusing"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  # -q reduces install noise to one line per package; pip's verbose
  # mode dwarfs everything else in the install log.
  pip install -q --upgrade pip
  if [[ -f requirements.txt ]]; then
    pip install -q -r requirements.txt
    ok "deps installed"
  else
    warn "requirements.txt not found at repo root — manual setup needed"
  fi
fi

# ── Default models (Ollama-only step) ─────────────────────────────────────
# Pulling the two defaults config.py expects (gemma3:4b + qwen2.5:3b) means
# `./run.sh` after install can serve real tasks without the user having to
# guess what to pull. Skipped when Ollama isn't installed (warned already)
# or --skip-models was passed (for users who already have other models).
bold "▶ Default models"
if [[ "$SKIP_MODELS" = "1" ]]; then
  info "skipped (--skip-models)"
elif ! command -v ollama >/dev/null; then
  info "Ollama not installed — skipping default-model pull"
elif [[ "$DRY_RUN" = "1" ]]; then
  info "[dry-run] would pull: ${DEFAULT_MODELS[*]}"
else
  for m in "${DEFAULT_MODELS[@]}"; do
    # `ollama list` columns: NAME  ID  SIZE  MODIFIED. Match on column 1.
    if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$m"; then
      ok "$m already pulled"
    else
      info "pulling $m (one-time, may take a few minutes)…"
      # Tail the last 3 lines so users see progress but the log stays
      # readable. Don't die() if pull fails — network blips happen and
      # the user can retry manually.
      if ollama pull "$m" 2>&1 | tail -3; then
        ok "$m"
      else
        warn "$m pull failed — retry later with: ollama pull $m"
      fi
    fi
  done
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo
bold "✅ OpenTeddy installed at $INSTALL_DIR"
echo
echo "  Start the backend:"
echo "    cd $INSTALL_DIR && ./run.sh --open"
echo
echo "  Or, if you want it on a different port:"
echo "    cd $INSTALL_DIR && ./run.sh --port 8001 --open"
echo
echo "  Then open:"
echo "    http://localhost:8000"
echo
echo "  Using cloud LLMs (no Ollama needed):"
echo "    Settings → Cloud LLM Provider → pick Anthropic / OpenAI / Gemini / Deepseek / OpenRouter"
echo
echo "  Updating later:"
echo "    cd $INSTALL_DIR && git pull && source .venv/bin/activate && pip install -r requirements.txt"
echo "    Or just re-run: curl -fsSL https://openteddy.net/install | bash"
echo
