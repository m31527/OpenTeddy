#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenTeddy vLLM engine setup — Linux / CUDA only
# ─────────────────────────────────────────────────────────────────────────────
#
# Installs vLLM into OpenTeddy's venv and registers a systemd service that
# serves the executor model over an OpenAI-compatible endpoint on
# 127.0.0.1:8001, so OpenTeddy's local_engine=vllm path has a persistent
# backend that survives reboots (mirrors setup-edge-cdp.sh for the CDP
# browser).
#
# WHEN TO USE THIS:
#   vLLM helps OpenTeddy under CONCURRENT load — multiple operators +
#   watcher loops + scheduled tasks hitting the same node. On a single-
#   stream personal install it does NOT beat Ollama (both are memory-
#   bandwidth-bound on a GB10-class box), and it's more operationally
#   fragile. So: enable vLLM on the "hot" fleet nodes that serve real
#   concurrency; leave the rest (and every Mac) on Ollama.
#
# REQUIREMENTS:
#   - Linux + NVIDIA CUDA. vLLM has no macOS / Metal build — running this
#     on a Mac is pointless (local_engine.py hard-gates Darwin to Ollama).
#   - A model in HuggingFace format (vLLM does NOT read Ollama's GGUF).
#     Default below is Qwen/Qwen2.5-7B-Instruct; override with --model.
#   - Enough free unified memory. BF16 needs ~2 bytes/param (7B≈15GB,
#     32B≈64GB). Quantize (--quantization fp8) to roughly halve that and
#     to be bandwidth-competitive with Ollama's Q4 GGUF.
#
# Usage:
#   sudo bash scripts/setup-vllm.sh                       # defaults
#   sudo bash scripts/setup-vllm.sh --model Qwen/Qwen2.5-32B-Instruct \
#        --max-model-len 16384 --gpu-mem 0.6 --quantization fp8
#   sudo bash scripts/setup-vllm.sh --uninstall
#
# After setup, point OpenTeddy at it (or set these in the Settings UI):
#   OPENTEDDY_LOCAL_ENGINE=vllm
#   VLLM_BASE_URL=http://127.0.0.1:8001
#   QWEN_MODEL=<the same --model you served>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults (override via flags) ────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT="8001"
MAX_MODEL_LEN="16384"
GPU_MEM="0.5"
QUANTIZATION=""              # "", "fp8", "awq", "gptq", ...
TOOL_PARSER="hermes"         # hermes for Qwen; llama3_json for Llama, etc.
ENFORCE_EAGER="false"        # true = skip torch.compile (faster start, ~15% slower run)
UNINSTALL="false"

while [ $# -gt 0 ]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2 ;;
    --port)           PORT="$2"; shift 2 ;;
    --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem)        GPU_MEM="$2"; shift 2 ;;
    --quantization)   QUANTIZATION="$2"; shift 2 ;;
    --tool-parser)    TOOL_PARSER="$2"; shift 2 ;;
    --enforce-eager)  ENFORCE_EAGER="true"; shift ;;
    --uninstall|-u)   UNINSTALL="true"; shift ;;
    -h|--help)        sed -n '4,46p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TARGET_USER="${OPENTEDDY_USER:-${SUDO_USER:-${USER:-openteddy}}}"
if ! id -u "${TARGET_USER}" >/dev/null 2>&1; then
  echo "✗ Target user '${TARGET_USER}' does not exist."; exit 1
fi
# Resolve OpenTeddy home (only used for the verify-vllm.py hint).
OT_HOME="$(eval echo "~${TARGET_USER}")/OpenTeddy"
# vLLM gets its OWN dedicated venv — NOT OpenTeddy's. vLLM pins
# aggressive dependency versions (it downgraded httpx and broke
# chromadb when installed into OpenTeddy's .venv). Isolating it means
# OpenTeddy's deps are never touched; the two processes talk over HTTP
# only, which is the correct architecture anyway.
VLLM_VENV="/var/lib/openteddy/vllm-venv"
VENV_PY="${VLLM_VENV}/bin/python"
SERVICE_FILE="/etc/systemd/system/openteddy-vllm.service"
LOG_PATH="/var/lib/openteddy/vllm.log"

if [ "$(id -u)" -ne 0 ]; then
  echo "✗ Run with sudo: sudo bash $0"; exit 1
fi

# ── Uninstall ────────────────────────────────────────────────────────────────
if [ "${UNINSTALL}" = "true" ]; then
  echo "▶ Uninstalling openteddy-vllm.service…"
  systemctl disable --now openteddy-vllm.service 2>/dev/null || true
  rm -f "${SERVICE_FILE}"
  systemctl daemon-reload
  echo "  ✓ Service removed. The dedicated vLLM venv (${VLLM_VENV}) + HF"
  echo "    model cache are left in place. To fully reclaim the disk:"
  echo "      sudo rm -rf ${VLLM_VENV}"
  echo "      rm -rf ~/.cache/huggingface   # the downloaded model weights"
  exit 0
fi

# ── Platform guard ───────────────────────────────────────────────────────────
if [ "$(uname -s)" != "Linux" ]; then
  echo "✗ vLLM is Linux/CUDA only. This host is $(uname -s)."
  echo "  On macOS, OpenTeddy uses Ollama (local_engine hard-gates Darwin)."
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠ nvidia-smi not found — vLLM needs an NVIDIA GPU + CUDA. Continuing"
  echo "  anyway in case this is an unusual setup, but expect failure if"
  echo "  there's no CUDA device."
fi

echo "▶ OpenTeddy vLLM setup"
echo "    user         : ${TARGET_USER}"
echo "    model        : ${MODEL}"
echo "    port         : 127.0.0.1:${PORT}"
echo "    max-model-len: ${MAX_MODEL_LEN}"
echo "    gpu-mem-util : ${GPU_MEM}"
echo "    quantization : ${QUANTIZATION:-none (bf16)}"
echo "    tool-parser  : ${TOOL_PARSER}"
echo "    enforce-eager: ${ENFORCE_EAGER}"
echo "    vllm venv    : ${VLLM_VENV}  (isolated from OpenTeddy's .venv)"
echo ""

# ── 0. System build deps for vLLM's runtime JIT ──────────────────────────────
# vLLM JIT-compiles CUDA kernels at first startup via TWO compilers:
#   - FlashInfer sampler → needs `ninja`. Missing → "FileNotFoundError:
#     ... 'ninja'" during the profile run.
#   - Triton sampler     → needs a C compiler + the Python dev headers
#     (Python.h). Missing → "fatal error: Python.h: No such file or
#     directory" → CalledProcessError from gcc.
# Both surface AFTER a full ~90s model load (the JIT only fires during
# the post-load profile run), so they look like late mysterious crashes.
# Install the whole build toolchain up front. Best-effort on apt; clear
# manual hint on non-Debian distros.
echo "▶ Ensuring vLLM JIT build deps (ninja, python3-dev, gcc)…"
if command -v apt-get >/dev/null 2>&1; then
  apt-get install -y -qq ninja-build python3-dev build-essential >/dev/null 2>&1 \
    && echo "  ✓ build deps installed (ninja-build, python3-dev, build-essential)" \
    || echo "  ⚠ apt install of build deps failed — install ninja-build + python3-dev + gcc by hand if vLLM crashes during startup"
else
  echo "  ⚠ No apt. Install these for your distro or vLLM will crash at"
  echo "    startup: ninja (ninja-build), Python dev headers (python3-dev /"
  echo "    python3-devel), and a C compiler (gcc / build-essential)."
fi

# ── 1. Create the DEDICATED vLLM venv + install vLLM into it ─────────────────
# Critical: vLLM goes in its OWN venv, never OpenTeddy's. vLLM pins
# conflicting versions (e.g. it downgrades httpx, which breaks chromadb
# → OpenTeddy won't even boot). Isolation = OpenTeddy's deps are
# untouched. The two only ever talk over HTTP on :PORT.
mkdir -p "$(dirname "${VLLM_VENV}")"
if [ ! -x "${VENV_PY}" ]; then
  echo "▶ Creating dedicated vLLM venv at ${VLLM_VENV}…"
  sudo -u "${TARGET_USER}" python3 -m venv "${VLLM_VENV}"
  chown -R "${TARGET_USER}:${TARGET_USER}" "$(dirname "${VLLM_VENV}")" 2>/dev/null || true
fi
echo "▶ Installing vLLM into the dedicated venv (this can take several minutes)…"
sudo -u "${TARGET_USER}" "${VENV_PY}" -m pip install --quiet --upgrade pip
sudo -u "${TARGET_USER}" "${VENV_PY}" -m pip install --quiet --upgrade vllm
VLLM_VER="$(sudo -u "${TARGET_USER}" "${VENV_PY}" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?')"
echo "  ✓ vLLM ${VLLM_VER} (in ${VLLM_VENV}, OpenTeddy's .venv untouched)"

# ── 2. Log dir ───────────────────────────────────────────────────────────────
mkdir -p "$(dirname "${LOG_PATH}")"
chown -R "${TARGET_USER}:${TARGET_USER}" /var/lib/openteddy 2>/dev/null || true

# ── 3. Build the ExecStart command ───────────────────────────────────────────
# Assembled as an array so optional flags (quantization / enforce-eager)
# only appear when set — vLLM rejects empty-string flag values.
EXEC="${VENV_PY} -m vllm.entrypoints.openai.api_server"
EXEC="${EXEC} --model ${MODEL}"
EXEC="${EXEC} --port ${PORT}"
EXEC="${EXEC} --host 127.0.0.1"
EXEC="${EXEC} --max-model-len ${MAX_MODEL_LEN}"
EXEC="${EXEC} --gpu-memory-utilization ${GPU_MEM}"
EXEC="${EXEC} --enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER}"
if [ -n "${QUANTIZATION}" ]; then
  EXEC="${EXEC} --quantization ${QUANTIZATION}"
fi
if [ "${ENFORCE_EAGER}" = "true" ]; then
  EXEC="${EXEC} --enforce-eager"
fi

# ── 4. systemd unit ──────────────────────────────────────────────────────────
echo "▶ Writing systemd unit to ${SERVICE_FILE}…"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=OpenTeddy vLLM inference server (${MODEL} on 127.0.0.1:${PORT})
Documentation=https://github.com/m31527/OpenTeddy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${TARGET_USER}
WorkingDirectory=${OT_HOME}
Restart=always
RestartSec=10
# vLLM cold-start (weight load + optional torch.compile) can take minutes;
# give systemd a long startup grace so it doesn't kill a healthy-but-slow
# boot.
TimeoutStartSec=900
Environment=HF_HUB_ENABLE_HF_TRANSFER=1
ExecStart=${EXEC}
StandardOutput=append:${LOG_PATH}
StandardError=append:${LOG_PATH}

[Install]
WantedBy=multi-user.target
EOF
echo "  ✓ Unit written"

# ── 5. Enable + start ────────────────────────────────────────────────────────
echo "▶ Enabling + starting (first boot downloads the model — can be slow)…"
systemctl daemon-reload
systemctl enable --now openteddy-vllm.service

# ── 6. Wait for readiness ────────────────────────────────────────────────────
echo "▶ Waiting for vLLM to come up (polling /v1/models, up to 10 min)…"
DEADLINE=$(( $(date +%s) + 600 ))
UP="false"
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
  if curl -sSf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    UP="true"; break
  fi
  sleep 5
done

echo ""
if [ "${UP}" = "true" ]; then
  echo "  ✓ vLLM serving on 127.0.0.1:${PORT}"
else
  echo "  ⚠ Not up after 10 min. Could be a slow first-boot model download,"
  echo "    OR a startup crash that systemd is silently restart-looping."
  echo ""
  echo "    To see the REAL error, run it in the foreground (systemd's"
  echo "    Restart=always hides the root cause behind a reload loop):"
  echo "        sudo systemctl stop openteddy-vllm.service"
  echo "        sudo pkill -9 -f vllm ; sleep 3"
  echo "        ${VENV_PY} -m vllm.entrypoints.openai.api_server \\"
  echo "          --model ${MODEL} --port ${PORT} --max-model-len ${MAX_MODEL_LEN} \\"
  echo "          --gpu-memory-utilization ${GPU_MEM} --enforce-eager \\"
  echo "          --enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER}"
  echo ""
  echo "    Common causes (full guide: docs/vllm-deployment.md):"
  echo "      • 'Free memory ... less than desired' → OOM. Stop Ollama"
  echo "        (sudo systemctl stop ollama) or lower --gpu-mem to 0.35."
  echo "      • 'ninja' / 'Python.h' not found → missing build deps"
  echo "        (this script installs them; re-run if apt was offline)."
  echo "      • Hangs at 'FlashAttention version 2' → torch.compile;"
  echo "        --enforce-eager (already set) skips it."
fi

echo ""
echo "✅ Setup done."
echo ""
echo "Point OpenTeddy at vLLM (env vars, or the Settings UI):"
echo "    OPENTEDDY_LOCAL_ENGINE=vllm"
echo "    VLLM_BASE_URL=http://127.0.0.1:${PORT}"
echo "    QWEN_MODEL=${MODEL}"
echo ""
echo "Verify the round-trip (uses OpenTeddy's .venv — verify-vllm.py only"
echo "HTTP-calls the vLLM server, so it needs OpenTeddy's deps, not vLLM's):"
echo "    cd ${OT_HOME} && OPENTEDDY_LOCAL_ENGINE=vllm \\"
echo "      VLLM_BASE_URL=http://127.0.0.1:${PORT} QWEN_MODEL=${MODEL} \\"
echo "      .venv/bin/python scripts/verify-vllm.py"
echo ""
echo "Service control:"
echo "    sudo systemctl status  openteddy-vllm.service"
echo "    sudo systemctl restart openteddy-vllm.service"
echo "    sudo journalctl -u openteddy-vllm.service -f"
echo ""
echo "Uninstall:"
echo "    sudo bash scripts/setup-vllm.sh --uninstall"
