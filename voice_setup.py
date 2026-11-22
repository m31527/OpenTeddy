"""
Voice mode turned on from the page: the voice service (contrib/voice —
SenseVoice + Fun-CosyVoice3) installed and run on this machine as a Docker
container named openteddy-voice.

Nothing here runs unasked. POST /voice/setup — the person pressing
「下載並啟用」 on the 🎙 prompt or in Settings — builds the image from
contrib/voice when it is missing or older than the files in this checkout,
(re)starts the container with the model folder mounted, and waits until the
service answers. The service downloads its models (~10 GB) into that folder
the first time. Then voice mode is turned on and pointed at it.

Turning voice mode off stops the container, which gives its few GB of GPU
memory back; the models stay on disk for next time.

What runs is fixed in this file: nothing from a request reaches a command.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import logging
import os
import platform
import re
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

NAME = "openteddy-voice"            # the container — the same name contrib/voice/README.md's docker run uses
IMAGE = "openteddy-voice:latest"
PORT = 8095
URL = f"http://127.0.0.1:{PORT}"
SRC = Path(__file__).resolve().parent / "contrib" / "voice"
MODELS = Path(os.getenv("OPENTEDDY_VOICE_MODELS", str(Path.home() / "voice-models")))
VOICES = Path(os.getenv("OPENTEDDY_VOICE_VOICES", str(Path.home() / "voice-voices")))
REV_LABEL = "org.openteddy.voice.rev"
# Sizes measured on demo2 (11/21), for what the prompt tells the person.
MODEL_GB = 10.0          # SenseVoiceSmall 0.9 GB + Fun-CosyVoice3-0.5B-2512 9.1 GB
BASE_GB = 11.0           # the vLLM base image, compressed — only when the machine doesn't have it
BASE_DISK_GB = 38.0      # … and unpacked
IMAGE_GB = 2.5           # what the build adds on top of the base
SPARE_GB = 5.0
LOAD_TIMEOUT_S = 3 * 3600   # 10 GB on a slow line
_LOCAL_URLS = {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}


class SetupError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = (detail or "").strip()[-1500:]


# ── What this machine has ────────────────────────────────────────────────────

def base_image() -> str:
    """The vLLM image the voice image is built on: its torch is the GPU's."""
    arch = platform.machine().lower()
    return "vllm/vllm-openai:v0.33.0-aarch64" if arch in {"aarch64", "arm64"} else "vllm/vllm-openai:v0.33.0"


def source_rev() -> str:
    """Which contrib/voice the image was built from (a label on it)."""
    h = hashlib.sha256()
    for name in ("Dockerfile", "server.py"):
        try:
            h.update((SRC / name).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:12]


def managed(url: str) -> bool:
    """Is this address the voice service this file runs?"""
    return (url or "").strip().rstrip("/") in _LOCAL_URLS


def models_gb() -> float:
    total = 0
    for root, _dirs, files in os.walk(MODELS):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total / 1e9


def _mem_available_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def _label(text: str) -> str:
    v = (text or "").strip()
    return "" if v in {"", "<no value>"} else v


async def _docker(*args: str, timeout: float = 30.0) -> Tuple[int, str]:
    exe = shutil.which("docker")
    if not exe:
        return 127, "docker not found"
    proc = await asyncio.create_subprocess_exec(exe, *args, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, f"docker {args[0]} took longer than {int(timeout)} s"
    return proc.returncode or 0, out.decode(errors="replace")


async def check() -> Dict[str, Any]:
    """Can 下載並啟用 work here, and what is already in place."""
    out: Dict[str, Any] = {
        "installable": False, "reason": "", "detail": "", "arch": platform.machine(),
        "image": False, "outdated": False, "container": "", "base": False,
        "models_gb": 0.0, "model_gb": MODEL_GB, "download_gb": 0.0, "disk_needed_gb": 0.0,
        "disk_free_gb": None, "mem_gb": None, "models_dir": str(MODELS),
    }
    if platform.system() != "Linux":
        out["reason"] = "not_linux"
        return out
    if platform.machine().lower() not in {"aarch64", "arm64", "x86_64", "amd64"}:
        out["reason"] = "arch"
        return out
    if not (SRC / "Dockerfile").is_file():
        out["reason"] = "no_source"
        return out
    if not shutil.which("docker"):
        out["reason"] = "no_docker"
        return out
    if not shutil.which("nvidia-smi"):
        out["reason"] = "no_gpu"
        return out
    rc, txt = await _docker("version", "--format", "{{.Server.Version}}", timeout=10)
    if rc != 0:
        out["reason"] = "docker_denied" if "permission denied" in txt.lower() else "docker_down"
        out["detail"] = txt.strip()[-400:]
        return out
    rev = source_rev()
    rc, txt = await _docker("image", "inspect", IMAGE, "--format", "{{ index .Config.Labels \"%s\" }}" % REV_LABEL)
    image_rev = _label(txt) if rc == 0 else ""
    out["image"] = rc == 0
    rc, txt = await _docker("container", "inspect", NAME, "--format",
                            "{{.State.Status}}|{{ index .Config.Labels \"%s\" }}" % REV_LABEL)
    container_rev = ""
    if rc == 0 and "|" in txt:
        status, container_rev = txt.strip().split("|", 1)
        out["container"] = status
        container_rev = _label(container_rev)
    out["outdated"] = bool((out["image"] and image_rev != rev) or (out["container"] and container_rev != rev))
    out["image_rev"], out["container_rev"], out["rev"] = image_rev, container_rev, rev
    rc, _ = await _docker("image", "inspect", base_image(), "--format", "{{.Id}}")
    out["base"] = rc == 0 or out["image"]
    out["models_gb"] = round(await asyncio.to_thread(models_gb), 1) if MODELS.is_dir() else 0.0
    left = max(0.0, MODEL_GB - out["models_gb"])
    out["download_gb"] = round(left + (0 if out["base"] else BASE_GB), 1)
    out["disk_needed_gb"] = round(left + (0 if out["image"] else IMAGE_GB) + (0 if out["base"] else BASE_DISK_GB), 1)
    probe = MODELS if MODELS.exists() else Path.home()
    try:
        out["disk_free_gb"] = round(shutil.disk_usage(probe).free / 1e9, 1)
    except OSError:
        pass
    mem = _mem_available_gb()
    out["mem_gb"] = round(mem, 1) if mem is not None else None
    if out["disk_free_gb"] is not None and out["disk_free_gb"] < out["disk_needed_gb"] + SPARE_GB:
        out["reason"] = "low_disk"
        return out
    out["installable"] = True
    return out


# ── The install, in the background ───────────────────────────────────────────

class _Job:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = "idle"          # idle | running | ready | error
        self.phase = ""              # check | build | start | download | load | warm | done
        self.step = ""               # the latest line worth showing
        self.progress: Optional[float] = None
        self.error = ""
        self.detail = ""
        self.started = 0.0
        self.finished = 0.0
        self.log: Deque[str] = collections.deque(maxlen=40)

    def view(self) -> Dict[str, Any]:
        return {"state": self.state, "phase": self.phase, "step": self.step, "progress": self.progress,
                "error": self.error, "detail": self.detail, "started": self.started,
                "elapsed_s": int((self.finished or time.time()) - self.started) if self.started else 0,
                "log": list(self.log)[-12:]}


JOB = _Job()
_task: Optional["asyncio.Task[None]"] = None
_STEP = re.compile(r"\[(?:[\w.-]+ )?(\d+)/(\d+)\]")


def job() -> Dict[str, Any]:
    return JOB.view()


def running() -> bool:
    return _task is not None and not _task.done()


def begin(on_ready: Callable[[str], Awaitable[None]]) -> Dict[str, Any]:
    """Start (or keep) the install; on_ready(url) turns voice mode on."""
    global _task
    if running():
        return job()
    JOB.reset()
    JOB.state, JOB.phase, JOB.started = "running", "check", time.time()
    _task = asyncio.create_task(_install(on_ready))
    return job()


async def cancel() -> None:
    if running() and _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


def _on_build_line(line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    JOB.log.append(line[:300])
    m = _STEP.search(line)
    if m:
        k, n = int(m.group(1)), int(m.group(2))
        if n:
            JOB.progress = max(JOB.progress or 0.0, min(0.99, (k - 1) / n))
        JOB.step = line.split("]", 1)[-1].strip()[:160]


async def _build() -> Tuple[int, str]:
    exe = shutil.which("docker") or "docker"
    proc = await asyncio.create_subprocess_exec(
        exe, "build", "--progress=plain", "--build-arg", f"BASE={base_image()}",
        "--label", f"{REV_LABEL}={source_rev()}", "-t", IMAGE, str(SRC),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, limit=1 << 20,
        env={**os.environ, "DOCKER_BUILDKIT": "1"})
    tail: Deque[str] = collections.deque(maxlen=30)
    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace")
            tail.append(line.rstrip())
            _on_build_line(line)
        return await proc.wait(), "\n".join(tail)
    except asyncio.CancelledError:
        proc.kill()
        raise


async def _run_container() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    VOICES.mkdir(parents=True, exist_ok=True)
    rc, txt = await _docker(
        "run", "-d", "--name", NAME, "--restart", "unless-stopped", "--gpus", "all",
        "-p", f"127.0.0.1:{PORT}:{PORT}", "-e", "VOICE_HOST=0.0.0.0",
        "-v", f"{MODELS}:/models", "-v", f"{VOICES}:/voices", IMAGE, timeout=180)
    if rc != 0:
        raise SetupError("start_failed", txt)


async def _wait_ready(client: httpx.AsyncClient) -> None:
    deadline = time.time() + LOAD_TIMEOUT_S
    while time.time() < deadline:
        rc, txt = await _docker("container", "inspect", NAME, "--format", "{{.State.Status}}|{{.RestartCount}}")
        status, _, restarts = txt.strip().partition("|")
        if rc != 0 or status in {"exited", "dead"} or (restarts.isdigit() and int(restarts) >= 3):
            _, logs = await _docker("logs", "--tail", "30", NAME)
            raise SetupError("crashed", logs)
        try:
            h = (await client.get(f"{URL}/health")).json()
        except Exception:  # noqa: BLE001 — still starting
            h = {}
        if h.get("ok"):
            return
        stage = str(h.get("stage") or "starting")
        if stage == "error":
            raise SetupError("load_failed", str(h.get("detail") or ""))
        JOB.phase = {"downloading": "download", "loading": "load", "warming": "warm"}.get(stage, "start")
        JOB.step = str(h.get("detail") or "")
        JOB.progress = (min(0.99, await asyncio.to_thread(models_gb) / MODEL_GB)
                        if JOB.phase == "download" else None)
        await asyncio.sleep(2.0)
    raise SetupError("timeout")


async def _install(on_ready: Callable[[str], Awaitable[None]]) -> None:
    try:
        c = await check()
        if not c["installable"]:
            raise SetupError(c["reason"] or "not_installable", c.get("detail") or "")
        rev = c["rev"]
        built = False
        if c["image_rev"] != rev:
            JOB.phase, JOB.progress, JOB.step = "build", 0.0, ""
            logger.info("voice setup: building %s from %s (rev %s)", IMAGE, SRC, rev)
            rc, tail = await _build()
            if rc != 0:
                raise SetupError("build_failed", tail)
            built = True
        JOB.phase, JOB.progress, JOB.step = "start", None, ""
        status = c["container"]
        if status and (built or c["container_rev"] != rev):
            await _docker("rm", "-f", NAME, timeout=60)        # an older build of it: replaced, models kept
            status = ""
        if not status:
            await _run_container()
        elif status != "running":
            rc, txt = await _docker("start", NAME, timeout=60)
            if rc != 0:
                raise SetupError("start_failed", txt)
        async with httpx.AsyncClient(timeout=5.0) as client:
            if status == "running":
                try:
                    h = (await client.get(f"{URL}/health")).json()
                except Exception:  # noqa: BLE001
                    h = {}
                if h.get("stage") == "error":                   # a failed load: try again from scratch
                    await _docker("restart", NAME, timeout=60)
            await _wait_ready(client)
        JOB.phase, JOB.progress, JOB.step = "done", 1.0, ""
        await on_ready(URL)
        JOB.state = "ready"
        logger.info("voice setup: ready at %s after %ds", URL, int(time.time() - JOB.started))
    except asyncio.CancelledError:
        JOB.state, JOB.error = "error", "cancelled"
        raise
    except SetupError as exc:
        JOB.state, JOB.error, JOB.detail = "error", exc.code, exc.detail
        logger.warning("voice setup failed: %s %s", exc.code, exc.detail[-300:])
    except Exception as exc:  # noqa: BLE001
        JOB.state, JOB.error, JOB.detail = "error", "unexpected", f"{type(exc).__name__}: {exc}"
        logger.exception("voice setup failed")
    finally:
        JOB.finished = time.time()


async def stop() -> bool:
    """Stop the container (voice mode turned off): its GPU memory back, models kept."""
    await cancel()
    if not shutil.which("docker"):
        return False
    rc, txt = await _docker("container", "inspect", NAME, "--format", "{{.State.Status}}")
    if rc != 0 or txt.strip() != "running":
        return False
    rc, _ = await _docker("stop", NAME, timeout=60)
    return rc == 0
