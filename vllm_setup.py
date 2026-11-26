"""
vLLM for the executor, turned on from Settings: one container named
openteddy-vllm-executor on this machine serving one of the models in CHOICES, and
the executor pointed at it through the vllm_models setting. The planner,
vision and embeddings stay on Ollama.

Nothing here runs unasked. POST /vllm/setup (the person choosing vLLM in
Settings and pressing 下載並啟用) pulls the model's vLLM image when it is
missing, (re)starts the container with the Hugging Face cache mounted, and
waits until the model answers. The first time, the container downloads the
weights into that cache. Then main.py points the executor at it and keeps
the model it had.

POST /vllm/off (choosing Ollama again) stops the container, which gives its
GPU memory back, and the executor gets its old model back. The image and
weights stay on disk, so switching back later is a start (about 5 minutes
of loading), not a download.

What runs is fixed in this file: the images, the models and every vLLM
flag. Nothing from a request reaches a command except the id of a CHOICE.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import platform
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Not "openteddy-vllm": that is the systemd unit scripts/setup-vllm.sh makes.
NAME = "openteddy-vllm-executor"
# Not 8001: vLLM servers started by hand (scripts/setup-vllm.sh, demo, demo2)
# sit there, and this one must not collide with them.
PORT = 8021
URL = f"http://127.0.0.1:{PORT}"
CACHE = Path(os.getenv("OPENTEDDY_VLLM_CACHE", str(Path.home() / "hf-cache")))
MODEL_LABEL = "org.openteddy.vllm.model"
SPARE_DISK_GB = 5.0
SPARE_MEM_GB = 4.0
MAX_SHARE = 0.85            # the most of the GPU's memory it may be given
LOAD_TIMEOUT_S = 4 * 3600   # 20 GB on a slow line, then the warm-up


@dataclass(frozen=True)
class Choice:
    id: str                  # what the executor calls it (vLLM --served-model-name)
    title: str
    hf: str                  # Hugging Face repo of the weights
    images: Tuple[str, str]  # (aarch64, x86-64)
    model_gb: float          # the weights, measured on demo and demo2
    image_gb: float          # the image's download, compressed (arm64)
    image_disk_gb: float     # … and unpacked
    mem_gb: float            # GPU memory it is given: weights + KV cache
    args: Tuple[str, ...]
    recommended: bool = False

    def image(self) -> str:
        return self.images[0] if platform.machine().lower() in {"aarch64", "arm64"} else self.images[1]


def _j(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


# The flags are the ones measured on a DGX Spark (GB10), 11/18 and 11/26.
CHOICES: Dict[str, Choice] = {c.id: c for c in (
    Choice(
        id="inkling-2-mini", title="Inkling 2 Mini", hf="thinkingmachines/Inkling-2-Mini",
        images=("vllm/vllm-openai:inkling-2-mini", "vllm/vllm-openai:inkling-2-mini"),
        model_gb=20.2, image_gb=11.3, image_disk_gb=27.1, mem_gb=36.0, recommended=True,
        args=("--max-model-len", "32768",
              "--reasoning-parser", "inkling", "--tool-call-parser", "inkling", "--enable-auto-tool-choice",
              # DSpark wants exactly 7: the checkpoint's block size − 1.
              "--speculative-config", _j({"method": "dspark", "num_speculative_tokens": 7,
                                          "draft_sample_method": "probabilistic"}),
              # Inkling always thinks first; "low" keeps that short (its own default is "high").
              "--default-chat-template-kwargs", _j({"reasoning_effort": "low"})),
    ),
    Choice(
        id="qwen3.6:35b", title="Qwen3.6 35B", hf="nvidia/Qwen3.6-35B-A3B-NVFP4",
        images=("vllm/vllm-openai:v0.33.0-aarch64", "vllm/vllm-openai:v0.33.0"),
        model_gb=23.5, image_gb=10.9, image_disk_gb=37.7, mem_gb=48.0,
        args=("--trust-remote-code", "--quantization", "modelopt", "--kv-cache-dtype", "fp8",
              "--attention-backend", "flashinfer", "--moe-backend", "marlin",
              "--max-model-len", "65536", "--max-num-seqs", "4", "--max-num-batched-tokens", "8192",
              "--enable-chunked-prefill", "--async-scheduling", "--enable-prefix-caching",
              "--speculative-config", _j({"method": "mtp", "num_speculative_tokens": 3, "moe_backend": "triton"}),
              "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_xml", "--enable-auto-tool-choice",
              "--limit-mm-per-prompt", _j({"image": 0, "video": 0})),
    ),
)}


class SetupError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = (detail or "").strip()[-1500:]


# ── What this machine has ────────────────────────────────────────────────────

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


async def _nvidia_smi() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ""
    try:
        proc = await asyncio.create_subprocess_exec(
            exe, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 10)
        return out.decode(errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _group_pending(name: str) -> bool:
    """This user was added to group `name`, but this process started before
    and doesn't have it yet. A login shell would; OpenTeddy as a service
    keeps the groups its systemd user manager started with (until a reboot,
    or until that manager is restarted)."""
    try:
        import grp
        import pwd
        g = grp.getgrnam(name)
        user = pwd.getpwuid(os.getuid()).pw_name
        listed = user in g.gr_mem or pwd.getpwuid(os.getuid()).pw_gid == g.gr_gid
        return listed and g.gr_gid not in os.getgroups()
    except (KeyError, ImportError, OSError):
        return False


def _meminfo() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    out[key] = int(rest.split()[0]) / 1e6
    except OSError:
        pass
    return out


async def memory() -> Dict[str, Any]:
    """The GPU's memory in GB. A DGX Spark's GB10 has none of its own
    (nvidia-smi says [N/A]): it shares the system's, so /proc/meminfo counts."""
    first = (await _nvidia_smi()).strip().splitlines()[:1]
    parts = [p.strip() for p in first[0].split(",")] if first else []
    if len(parts) == 2 and all(p.replace(".", "", 1).isdigit() for p in parts):
        return {"total": round(float(parts[0]) / 1024, 1), "free": round(float(parts[1]) / 1024, 1), "shared": False}
    m = _meminfo()
    total, free = m.get("MemTotal"), m.get("MemAvailable")
    return {"total": round(total, 1) if total else None, "free": round(free, 1) if free else None, "shared": True}


def _repo_dir(c: Choice) -> Path:
    return CACHE / "hub" / ("models--" + c.hf.replace("/", "--"))


def weights_gb(c: Choice) -> float:
    """How much of the model is downloaded: its largest complete snapshot."""
    best = 0
    snaps = _repo_dir(c) / "snapshots"
    if not snaps.is_dir():
        return 0.0
    for snap in snaps.iterdir():
        total = 0
        for root, _dirs, files in os.walk(snap):
            for f in files:
                try:
                    total += os.stat(os.path.join(root, f)).st_size     # follows the link into blobs
                except OSError:
                    pass
        best = max(best, total)
    return best / 1e9


def has_weights(c: Choice) -> bool:
    return weights_gb(c) >= 0.95 * c.model_gb


def cache_bytes() -> int:
    """Everything in the cache, half-downloaded files included: how far a
    download has got, whatever layout huggingface_hub writes."""
    total = 0
    for root, _dirs, files in os.walk(CACHE):
        for f in files:
            try:
                st = os.lstat(os.path.join(root, f))
            except OSError:
                continue
            if not os.path.islink(os.path.join(root, f)):
                total += st.st_size
    return total


def share_for(c: Choice, total_gb: Optional[float]) -> float:
    """--gpu-memory-utilization: the model's memory as a share of the GPU's."""
    if not total_gb:
        return 0.3
    return round(min(MAX_SHARE, max(0.1, (c.mem_gb + 0.5) / total_gb)), 2)


async def container() -> Tuple[str, str]:
    """(status, model) of the container; ("", "") when there is none."""
    rc, txt = await _docker("container", "inspect", NAME, "--format",
                            "{{.State.Status}}|{{ index .Config.Labels \"%s\" }}" % MODEL_LABEL)
    if rc != 0 or "|" not in txt:
        return "", ""
    status, model = txt.strip().split("|", 1)
    return status, ("" if model in ("", "<no value>") else model)


async def check(reclaim_gb: float = 0.0) -> Dict[str, Any]:
    """Can 下載並啟用 work here, for each model, and what is already in place.

    reclaim_gb: memory that turning vLLM on gives back first (the executor
    model Ollama has loaded)."""
    out: Dict[str, Any] = {
        "installable": False, "reason": "", "detail": "", "arch": platform.machine(),
        "container": "", "container_model": "", "disk_free_gb": None,
        "mem_total_gb": None, "mem_free_gb": None, "shared_memory": False,
        "cache_dir": str(CACHE), "choices": [],
    }
    if platform.system() != "Linux":
        out["reason"] = "not_linux"
        return out
    if platform.machine().lower() not in {"aarch64", "arm64", "x86_64", "amd64"}:
        out["reason"] = "arch"
        return out
    if not shutil.which("docker"):
        out["reason"] = "no_docker"
        return out
    if not shutil.which("nvidia-smi"):
        out["reason"] = "no_gpu"
        return out
    rc, txt = await _docker("version", "--format", "{{.Server.Version}}", timeout=10)
    if rc != 0:
        denied = "permission denied" in txt.lower()
        out["reason"] = ("docker_restart" if denied and _group_pending("docker")
                         else "docker_denied" if denied else "docker_down")
        out["detail"] = txt.strip()[-400:]
        return out
    status, running_model = await container()
    out["container"], out["container_model"] = status, running_model
    mem = await memory()
    out["mem_total_gb"], out["shared_memory"] = mem["total"], mem["shared"]
    out["mem_free_gb"] = mem["free"]
    try:
        out["disk_free_gb"] = round(shutil.disk_usage(CACHE if CACHE.exists() else Path.home()).free / 1e9, 1)
    except OSError:
        pass
    for c in CHOICES.values():
        rc, _ = await _docker("image", "inspect", c.image(), "--format", "{{.Id}}")
        have_image = rc == 0
        have_gb = await asyncio.to_thread(weights_gb, c)
        have_model = have_gb >= 0.95 * c.model_gb
        left_gb = 0.0 if have_model else max(0.0, c.model_gb - have_gb)
        entry: Dict[str, Any] = {
            "id": c.id, "title": c.title, "recommended": c.recommended, "mem_gb": c.mem_gb,
            "model_gb": c.model_gb, "have_model": have_model, "have_image": have_image,
            "download_gb": round(left_gb + (0 if have_image else c.image_gb), 1),
            "disk_needed_gb": round(left_gb + (0 if have_image else c.image_disk_gb), 1),
            "ok": False, "reason": "",
        }
        # Memory our own container holds now comes back when it is replaced or restarted.
        free = mem["free"]
        if free is not None:
            free += reclaim_gb
            if status == "running" and running_model in CHOICES:
                free += CHOICES[running_model].mem_gb
        if out["disk_free_gb"] is not None and out["disk_free_gb"] < entry["disk_needed_gb"] + SPARE_DISK_GB:
            entry["reason"] = "low_disk"
        elif mem["total"] is not None and mem["total"] * MAX_SHARE < c.mem_gb:
            entry["reason"] = "too_small"
        elif free is not None and free < c.mem_gb + SPARE_MEM_GB:
            entry["reason"] = "low_mem"
        else:
            entry["ok"] = True
        out["choices"].append(entry)
    ok = [e for e in out["choices"] if e["ok"]]
    out["installable"] = bool(ok)
    if not ok:
        out["reason"] = out["choices"][0]["reason"] if out["choices"] else "no_choice"
    return out


async def answers(model: str, timeout: float = 3.0) -> bool:
    """Does the container answer, serving `model`?"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{URL}/v1/models")
            return r.status_code == 200 and any(m.get("id") == model for m in (r.json().get("data") or []))
    except Exception:  # noqa: BLE001
        return False


# ── The install, in the background ───────────────────────────────────────────

class _Job:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = "idle"          # idle | running | ready | error
        self.phase = ""              # check | pull | start | download | load | done
        self.model = ""
        self.step = ""
        self.progress: Optional[float] = None
        self.done_gb: Optional[float] = None
        self.error = ""
        self.detail = ""
        self.started = 0.0
        self.finished = 0.0
        self.log: Deque[str] = collections.deque(maxlen=40)

    def view(self) -> Dict[str, Any]:
        return {"state": self.state, "phase": self.phase, "model": self.model, "step": self.step,
                "progress": self.progress, "done_gb": self.done_gb, "error": self.error,
                "detail": self.detail, "started": self.started,
                "elapsed_s": int((self.finished or time.time()) - self.started) if self.started else 0,
                "log": list(self.log)[-12:]}


JOB = _Job()
_task: Optional["asyncio.Task[None]"] = None
_LAYER = re.compile(r"^([0-9a-f]{12}): (.+)$")
_PERCENT = re.compile(r"(\d{1,3})%\|")


def job() -> Dict[str, Any]:
    return JOB.view()


def running() -> bool:
    return _task is not None and not _task.done()


def begin(model: str, on_ready: Callable[[Choice], Awaitable[None]],
          before: Optional[Callable[[], Awaitable[float]]] = None) -> Dict[str, Any]:
    """Start (or keep) the install of CHOICES[model]. before() runs first and
    says how much memory it freed; on_ready(choice) points the executor at it."""
    global _task
    if running():
        return job()
    if model not in CHOICES:
        raise KeyError(model)
    JOB.reset()
    JOB.state, JOB.phase, JOB.model, JOB.started = "running", "check", model, time.time()
    _task = asyncio.create_task(_install(CHOICES[model], on_ready, before))
    return job()


async def cancel() -> None:
    if running() and _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


def _on_pull_line(line: str, seen: set, done: set) -> None:
    line = line.strip()
    if not line:
        return
    JOB.log.append(line[:300])
    m = _LAYER.match(line)
    if not m:
        return
    layer, what = m.groups()
    seen.add(layer)
    if what.startswith(("Pull complete", "Already exists")):
        done.add(layer)
    JOB.progress = min(0.99, len(done) / len(seen))
    JOB.step = f"{len(done)} / {len(seen)}"


async def _pull(image: str) -> Tuple[int, str]:
    exe = shutil.which("docker") or "docker"
    proc = await asyncio.create_subprocess_exec(exe, "pull", image, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT, limit=1 << 20)
    tail: Deque[str] = collections.deque(maxlen=30)
    seen: set = set()
    done: set = set()
    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace")
            tail.append(line.rstrip())
            _on_pull_line(line, seen, done)
        return await proc.wait(), "\n".join(tail)
    except asyncio.CancelledError:
        proc.kill()
        raise


async def _run_container(c: Choice, share: float, offline: bool) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cmd = ["run", "-d", "--name", NAME, "--restart", "unless-stopped", "--gpus", "all", "--ipc=host",
           "-p", f"127.0.0.1:{PORT}:8000", "-v", f"{CACHE}:/root/.cache/huggingface",
           "--label", f"{MODEL_LABEL}={c.id}"]
    if offline:                       # the weights are here: no need to ask the Hub on each start
        cmd += ["-e", "HF_HUB_OFFLINE=1"]
    cmd += [c.image(), c.hf, "--served-model-name", c.id, "--gpu-memory-utilization", f"{share}", *c.args]
    rc, txt = await _docker(*cmd, timeout=180)
    if rc != 0:
        raise SetupError("start_failed", txt)


def _last_log_line(txt: str) -> str:
    """The latest thing vLLM said (a progress bar redraws with \\r)."""
    lines = [seg.strip() for ln in txt.splitlines() for seg in ln.split("\r") if seg.strip()]
    return lines[-1] if lines else ""


async def _wait_ready(c: Choice) -> None:
    deadline = time.time() + LOAD_TIMEOUT_S
    base = await asyncio.to_thread(cache_bytes)
    need = max(1.0, (c.model_gb - await asyncio.to_thread(weights_gb, c)) * 1e9)
    while time.time() < deadline:
        rc, txt = await _docker("container", "inspect", NAME, "--format", "{{.State.Status}}|{{.RestartCount}}")
        status, _, restarts = txt.strip().partition("|")
        if rc != 0 or status in {"exited", "dead"} or (restarts.isdigit() and int(restarts) >= 2):
            _, logs = await _docker("logs", "--tail", "40", NAME)
            raise SetupError("crashed", logs)
        if await answers(c.id):
            return
        if not await asyncio.to_thread(has_weights, c):
            JOB.phase = "download"
            got = max(0, await asyncio.to_thread(cache_bytes) - base)
            JOB.progress = min(0.99, got / need)
            JOB.done_gb = round(min(c.model_gb, c.model_gb - need / 1e9 + got / 1e9), 1)
            JOB.step = ""
        else:
            JOB.phase, JOB.done_gb = "load", None
            _, logs = await _docker("logs", "--tail", "3", NAME)
            line = _last_log_line(logs)
            m = _PERCENT.search(line)
            JOB.progress = int(m.group(1)) / 100 if m and "graph" in line.lower() else None
            JOB.step = re.sub(r"^\(\w+ pid=\d+\)\s*", "", line)[:160]
        await asyncio.sleep(3.0)
    raise SetupError("timeout")


async def _install(c: Choice, on_ready: Callable[[Choice], Awaitable[None]],
                   before: Optional[Callable[[], Awaitable[float]]]) -> None:
    try:
        freed = 0.0
        if before is not None:
            try:
                freed = float(await before() or 0.0)
            except Exception:  # noqa: BLE001 — making room is a courtesy
                logger.debug("vllm setup: before() failed", exc_info=True)
        chk = await check(reclaim_gb=freed)
        entry = next((e for e in chk["choices"] if e["id"] == c.id), None)
        if entry is None or not entry["ok"]:
            raise SetupError((entry or {}).get("reason") or chk["reason"] or "not_installable", chk.get("detail") or "")
        if not entry["have_image"]:
            JOB.phase, JOB.progress, JOB.step = "pull", 0.0, ""
            logger.info("vllm setup: pulling %s", c.image())
            rc, tail = await _pull(c.image())
            if rc != 0:
                raise SetupError("pull_failed", tail)
        JOB.phase, JOB.progress, JOB.step = "start", None, ""
        status, model = chk["container"], chk["container_model"]
        if status and model != c.id:
            await _docker("stop", NAME, timeout=90)
            await _docker("rm", "-f", NAME, timeout=60)        # another model: replaced, its weights kept
            status = ""
        if not status:
            mem = await memory()
            await _run_container(c, share_for(c, mem["total"]), offline=entry["have_model"])
        elif status != "running":
            rc, txt = await _docker("start", NAME, timeout=60)
            if rc != 0:
                raise SetupError("start_failed", txt)
        await _wait_ready(c)
        JOB.phase, JOB.progress, JOB.step, JOB.done_gb = "done", 1.0, "", None
        await on_ready(c)
        JOB.state = "ready"
        logger.info("vllm setup: %s ready at %s after %ds", c.id, URL, int(time.time() - JOB.started))
    except asyncio.CancelledError:
        JOB.state, JOB.error = "error", "cancelled"
        raise
    except SetupError as exc:
        JOB.state, JOB.error, JOB.detail = "error", exc.code, exc.detail
        logger.warning("vllm setup failed: %s %s", exc.code, exc.detail[-300:])
    except Exception as exc:  # noqa: BLE001
        JOB.state, JOB.error, JOB.detail = "error", "unexpected", f"{type(exc).__name__}: {exc}"
        logger.exception("vllm setup failed")
    finally:
        JOB.finished = time.time()


async def stop() -> bool:
    """Stop the container (back to Ollama): its memory back, image and weights kept."""
    await cancel()
    if not shutil.which("docker"):
        return False
    status, _ = await container()
    if status != "running" and status != "restarting":
        return False
    rc, _ = await _docker("stop", NAME, timeout=90)
    return rc == 0


def choices() -> List[Dict[str, Any]]:
    return [{"id": c.id, "title": c.title, "recommended": c.recommended} for c in CHOICES.values()]
