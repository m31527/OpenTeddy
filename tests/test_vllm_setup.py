"""
Settings → vLLM (vllm_setup.py and /vllm/* in main.py), without Docker,
a GPU or a model:

  A. what this machine can do: not Linux / no Docker / no GPU / no Docker
     permission said; per model, not enough disk, a GPU too small, not
     enough free memory; a GB10's shared memory read from /proc/meminfo
  B. the install: nothing here → pulled, started (this machine only, the
     cache mounted, the model's own flags), download then load followed,
     executor pointed at it; already here → just started; another model's
     container → replaced; a failed pull and a crash said with the reason
  C. the endpoints: status; 下載並啟用 moves only the executor (Ollama lets
     go of it first); back to Ollama gives the old model back and stops the
     container; a route it replaced comes back; a server started by hand is
     used and left running; one set up by hand with no record of the model
     before falls back to the planner's model, never to one Ollama lacks

Run: python tests/test_vllm_setup.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENTEDDY_SKIP_WARMUP", "1")

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from config import config
import vllm_setup as V

INKLING, QWEN = V.CHOICES["inkling-2-mini"], V.CHOICES["qwen3.6:35b"]


async def serve(app: FastAPI) -> tuple:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    return server, task, port


def put_weights(c: V.Choice, gb: float = None) -> None:
    """A downloaded model in the cache: a sparse file of its size, linked
    from a snapshot the way huggingface_hub lays it out."""
    repo = V._repo_dir(c)
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True, exist_ok=True)
    blob = repo / "blobs" / "f00d"
    with open(blob, "wb") as fh:
        fh.truncate(int((gb if gb is not None else c.model_gb) * 1e9))
    link = snap / "model.safetensors"
    if not link.exists():
        os.symlink(blob, link)


class FakeDocker:
    """docker as vllm_setup sees it: which images are here, one container."""

    def __init__(self, images=(), container=None, crash=False, denied=False):
        self.images, self.container, self.crash, self.denied = set(images), container, crash, denied
        self.calls = []

    async def __call__(self, *args, timeout=30.0):
        self.calls.append(args)
        a = list(args)
        if a[:1] == ["version"]:
            return (1, "permission denied while trying to connect to the docker API") if self.denied else (0, "29.1.3")
        if a[:2] == ["image", "inspect"]:
            return (0, "sha256:x") if a[2] in self.images else (1, "No such image")
        if a[:2] == ["container", "inspect"]:
            if not self.container:
                return 1, "No such container"
            status, model = self.container
            if "{{.RestartCount}}" in a[-1]:
                return 0, f"{'exited' if self.crash else status}|0"
            return 0, f"{status}|{model or '<no value>'}"
        if a[0] == "run":
            label = next(x for x in a if x.startswith(V.MODEL_LABEL + "="))
            self.container = ("running", label.split("=", 1)[1])
            return 0, "abc123"
        if a[0] == "start":
            self.container = ("running", self.container[1])
            return 0, V.NAME
        if a[0] == "stop":
            if self.container:
                self.container = ("exited", self.container[1])
            return 0, V.NAME
        if a[:2] == ["rm", "-f"]:
            self.container = None
            return 0, V.NAME
        if a[0] == "logs":
            if self.crash:
                return 0, "torch.OutOfMemoryError: CUDA out of memory"
            return 0, "INFO loading\rCapturing CUDA graphs (PIECEWISE):  40%|████      | 20/51 [00:20<00:31]"
        return 0, ""


def fake_vllm(state: dict) -> FastAPI:
    """A vLLM server that takes a few polls to come up; it 'downloads' the
    weights on the second one."""
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        state["polls"] = state.get("polls", 0) + 1
        if state["polls"] == 2 and state.get("download"):
            put_weights(state["download"])
        if state["polls"] < state.get("ready_after", 3):
            return JSONResponse({"detail": "loading"}, status_code=503)
        return {"object": "list", "data": [{"id": state["serves"]}]}
    return app


def fake_ollama(state: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/api/ps")
    async def ps():
        return {"models": [{"name": m, "size": 23e9} for m in state["loaded"]]}

    @app.get("/api/tags")
    async def tags():
        return {"models": [{"name": m} for m in ("qwen3.6:35b", "gemma4:e4b", "qwen3-vl:8b-instruct")]}

    @app.post("/api/generate")
    async def generate(body: dict):
        if body.get("keep_alive") == 0:
            state["loaded"].discard(body.get("model"))
            state["unloaded"].append(body.get("model"))
        return {"done": True}
    return app


def nvidia(text: str):
    async def run():
        return text
    return run


REAL = None


def patch(tmp: str) -> None:
    global REAL
    REAL = (V.platform.system, V.platform.machine, V.shutil.which, V._docker, V._nvidia_smi, V._meminfo,
            V.CACHE, V.URL, V.shutil.disk_usage, V.asyncio.sleep, V._pull)
    V.platform.system, V.platform.machine = (lambda: "Linux"), (lambda: "aarch64")
    V.shutil.which = lambda name: f"/usr/bin/{name}"
    V._nvidia_smi = nvidia("[N/A], [N/A]\n")                 # a GB10: no memory of its own
    V._meminfo = lambda: {"MemTotal": 121.0, "MemAvailable": 100.0}
    V.CACHE = Path(tmp, "hf-cache")
    V.shutil.disk_usage = lambda p: types.SimpleNamespace(free=500e9)


def unpatch() -> None:
    (V.platform.system, V.platform.machine, V.shutil.which, V._docker, V._nvidia_smi, V._meminfo,
     V.CACHE, V.URL, V.shutil.disk_usage, V.asyncio.sleep, V._pull) = REAL


# ── A ────────────────────────────────────────────────────────────────────────

async def test_check() -> None:
    V.platform.system = lambda: "Darwin"
    c = await V.check()
    assert not c["installable"] and c["reason"] == "not_linux" and not c["choices"], c
    V.platform.system = lambda: "Linux"
    V.shutil.which = lambda name: None if name == "docker" else f"/usr/bin/{name}"
    assert (await V.check())["reason"] == "no_docker"
    V.shutil.which = lambda name: None if name == "nvidia-smi" else f"/usr/bin/{name}"
    assert (await V.check())["reason"] == "no_gpu"
    V.shutil.which = lambda name: f"/usr/bin/{name}"
    V._docker = FakeDocker(denied=True)
    real_pending = V._group_pending
    V._group_pending = lambda name: False
    c = await V.check()
    assert c["reason"] == "docker_denied" and "permission denied" in c["detail"], c
    # Added to the group after OpenTeddy's user manager started (demo, 11/26):
    # a restart is what's missing, not the usermod.
    V._group_pending = lambda name: name == "docker"
    assert (await V.check())["reason"] == "docker_restart"
    V._group_pending = real_pending
    assert real_pending("no-such-group-here") is False
    print("  ✓ not Linux / no Docker / no GPU / no Docker permission / group not active yet: said, nothing offered")

    V._docker = FakeDocker()
    c = await V.check()
    assert c["installable"] and c["shared_memory"] and c["mem_total_gb"] == 121.0 and c["mem_free_gb"] == 100.0, c
    ink = c["choices"][0]
    assert ink["id"] == "inkling-2-mini" and ink["recommended"] and ink["ok"], ink
    assert ink["download_gb"] == round(INKLING.model_gb + INKLING.image_gb, 1), ink
    put_weights(INKLING)
    V._docker = FakeDocker(images={INKLING.image()})
    ink = (await V.check())["choices"][0]
    assert ink["have_model"] and ink["have_image"] and ink["download_gb"] == 0 and ink["disk_needed_gb"] == 0, ink
    print("  ✓ a GB10's memory from /proc/meminfo; what's already here isn't downloaded again")

    V.shutil.disk_usage = lambda p: types.SimpleNamespace(free=40e9)
    by = {e["id"]: e for e in (await V.check())["choices"]}
    assert by["inkling-2-mini"]["ok"], "its weights and image are here: little disk needed"
    assert by["qwen3.6:35b"]["reason"] == "low_disk", by
    V.shutil.disk_usage = lambda p: types.SimpleNamespace(free=500e9)
    V._nvidia_smi = nvidia("40960, 38000\n")                 # a 40 GB card of its own
    c = await V.check()
    by = {e["id"]: e for e in c["choices"]}
    assert not c["shared_memory"] and c["mem_total_gb"] == 40.0, c
    assert by["qwen3.6:35b"]["reason"] == "too_small" and by["inkling-2-mini"]["reason"] == "too_small", by
    V._nvidia_smi = nvidia("[N/A], [N/A]\n")
    V._meminfo = lambda: {"MemTotal": 121.0, "MemAvailable": 30.0}
    by = {e["id"]: e for e in (await V.check())["choices"]}
    assert by["inkling-2-mini"]["reason"] == "low_mem", by
    by = {e["id"]: e for e in (await V.check(reclaim_gb=23.0))["choices"]}
    assert by["inkling-2-mini"]["ok"], "the executor Ollama lets go of first counts"
    V._docker = FakeDocker(images={INKLING.image()}, container=("running", "inkling-2-mini"))
    by = {e["id"]: e for e in (await V.check())["choices"]}
    assert by["inkling-2-mini"]["ok"], "our own running container gives its memory back"
    V._meminfo = lambda: {"MemTotal": 121.0, "MemAvailable": 100.0}
    assert V.share_for(INKLING, 121.0) == 0.3 and V.share_for(QWEN, 121.0) == 0.4 and V.share_for(QWEN, 50.0) == 0.85
    print("  ✓ per model: not enough disk, a GPU too small, not enough free memory (what comes back counted); "
          "its share of the GPU")


# ── B ────────────────────────────────────────────────────────────────────────

async def run_job(docker: FakeDocker, model: str, ready: list, phases: list) -> dict:
    V._docker = docker

    async def on_ready(c):
        ready.append(c.id)

    async def before():
        return 0.0
    V.begin(model, on_ready, before=before)
    while V.running():
        await REAL[9](0.005)
    return V.job()


async def test_install() -> None:
    shutil.rmtree(V.CACHE, ignore_errors=True)
    state = {"serves": "inkling-2-mini", "download": INKLING}
    srv, task, port = await serve(fake_vllm(state))
    V.URL = f"http://127.0.0.1:{port}"
    phases, ready, pulled = [], [], []

    async def quick(_s):
        phases.append(V.JOB.phase)
        await REAL[9](0)
    V.asyncio.sleep = quick

    async def pull(image):
        pulled.append(image)
        seen, done = set(), set()
        for line in ("inkling-2-mini: Pulling from vllm/vllm-openai", "aaaaaaaaaaaa: Pulling fs layer",
                     "bbbbbbbbbbbb: Pulling fs layer", "aaaaaaaaaaaa: Pull complete"):
            V._on_pull_line(line, seen, done)
        assert V.JOB.phase == "pull" and V.JOB.progress == 0.5 and V.JOB.step == "1 / 2", V.job()
        d.images.add(image)
        return 0, ""
    V._pull = pull
    try:
        d = FakeDocker()
        j = await run_job(d, "inkling-2-mini", ready, phases)
        assert j["state"] == "ready" and ready == ["inkling-2-mini"] and pulled == [INKLING.image()], j
        runs = [c for c in d.calls if c[0] == "run"]
        assert len(runs) == 1, d.calls
        r = list(runs[0])
        assert "127.0.0.1:%d:8000" % V.PORT in r and f"{V.CACHE}:/root/.cache/huggingface" in r, r
        assert "--gpus" in r and "--ipc=host" in r and "--restart" in r and "HF_HUB_OFFLINE=1" not in r, r
        assert r[r.index(INKLING.image()) + 1] == INKLING.hf and r[r.index("--served-model-name") + 1] == "inkling-2-mini", r
        assert r[r.index("--gpu-memory-utilization") + 1] == "0.3" and "inkling" in r and \
            '{"reasoning_effort":"low"}' in r, r
        assert "download" in phases and "load" in phases and phases.index("download") < phases.index("load"), phases
        print("  ✓ nothing here: pulled, started (this machine only, cache mounted, its own flags), "
              "download then load followed, the executor pointed at it")

        ready.clear(); phases.clear(); pulled.clear()
        state.update(polls=0, download=None)
        d = FakeDocker(images={INKLING.image()}, container=("exited", "inkling-2-mini"))
        j = await run_job(d, "inkling-2-mini", ready, phases)
        assert j["state"] == "ready" and not pulled and \
            [c[0] for c in d.calls if c[0] in ("run", "start", "rm", "stop")] == ["start"], d.calls
        assert "download" not in phases and "load" in phases, phases
        print("  ✓ already here: just started, nothing downloaded")

        ready.clear()
        state.update(polls=0, serves="qwen3.6:35b", download=QWEN)
        d = FakeDocker(images={INKLING.image()}, container=("running", "inkling-2-mini"))
        j = await run_job(d, "qwen3.6:35b", ready, phases)
        assert j["state"] == "ready" and ready == ["qwen3.6:35b"] and pulled == [QWEN.image()], j
        assert [c[0] for c in d.calls if c[0] in ("run", "start", "rm", "stop")] == ["stop", "rm", "run"], d.calls
        r = list(next(c for c in d.calls if c[0] == "run"))
        assert r[r.index("--gpu-memory-utilization") + 1] == "0.4" and "qwen3_xml" in r, r
        print("  ✓ another model's container: stopped and replaced (weights kept), the new one pulled")

        async def bad_pull(image):
            return 1, "Error response from daemon: no space left on device"
        V._pull = bad_pull
        ready.clear()
        shutil.rmtree(V.CACHE, ignore_errors=True)
        j = await run_job(FakeDocker(), "inkling-2-mini", ready, phases)
        assert j["state"] == "error" and j["error"] == "pull_failed" and "no space" in j["detail"] and not ready, j
        V._pull = pull
        state.update(polls=0, serves="inkling-2-mini", ready_after=99, download=None)
        put_weights(INKLING)
        j = await run_job(FakeDocker(images={INKLING.image()}, crash=True), "inkling-2-mini", ready, phases)
        assert j["state"] == "error" and j["error"] == "crashed" and "out of memory" in j["detail"] and not ready, j
        state["ready_after"] = 3
        print("  ✓ a failed pull and a crash: said, with the reason; the executor not moved")

        d = FakeDocker(container=("running", "inkling-2-mini"))
        V._docker = d
        assert await V.stop() and d.container[0] == "exited"
        assert not await V.stop(), "already stopped"
        print("  ✓ stop: the container stops (image and weights stay)")
    finally:
        srv.should_exit = True
        await task


# ── C ────────────────────────────────────────────────────────────────────────

async def test_endpoints(tmp: str) -> None:
    import main as M
    M.settings_store._db_path = os.path.join(tmp, "settings.db")
    await M.settings_store.init()
    ol = {"loaded": {"qwen3.6:35b"}, "unloaded": []}
    vl = {"serves": "inkling-2-mini", "ready_after": 1}
    manual = {"serves": "qwen3.6:35b", "ready_after": 1}
    o_srv, o_task, o_port = await serve(fake_ollama(ol))
    v_srv, v_task, v_port = await serve(fake_vllm(vl))
    m_srv, m_task, m_port = await serve(fake_vllm(manual))
    V.URL = f"http://127.0.0.1:{v_port}"
    manual_url = f"http://127.0.0.1:{m_port}"

    async def quick(_s):
        await REAL[9](0)
    V.asyncio.sleep = quick
    try:
        put_weights(INKLING)
        d = FakeDocker(images={INKLING.image(), QWEN.image()})
        V._docker = d
        await M._vllm_save({"executor_model": "qwen3.6:35b", "orchestrator_model": "gemma4:e4b",
                            "vision_model": "qwen3-vl:8b-instruct", "local_engine": "ollama", "vllm_models": "",
                            "vllm_managed": "", "qwen_base_url": f"http://127.0.0.1:{o_port}"})
        st = await M.vllm_status()
        assert not st["on"] and st["setup"]["installable"] and st["setup"]["job"]["state"] in ("idle", "ready", "error"), st

        r = await M.vllm_setup_start(M._VllmSetupBody(model="inkling-2-mini"))
        assert r["job"]["state"] == "running"
        while V.running():
            await REAL[9](0.01)
        assert V.job()["state"] == "ready", V.job()
        assert config.qwen_model == "inkling-2-mini" and config.gemma_model == "gemma4:e4b", config.qwen_model
        assert config.vllm_models == f"inkling-2-mini={V.URL}" and config.local_engine == "ollama", config.vllm_models
        assert ol["unloaded"] == ["qwen3.6:35b"], "Ollama lets go of the executor before vLLM loads"
        m = json.loads(config.vllm_managed)
        assert m["on"] and m["prev_executor"] == "qwen3.6:35b" and m["url"] == V.URL, m
        st = await M.vllm_status()
        assert st["on"] and st["managed"] and st["ready"] and st["prev_executor"] == "qwen3.6:35b", st
        print("  ✓ 下載並啟用: only the executor moves (planner and local_engine untouched); "
              "Ollama lets go of it first; status says it's on and answering")

        vl["serves"] = "qwen3.6:35b"
        r = await M.vllm_setup_start(M._VllmSetupBody(model="qwen3.6:35b"))
        while V.running():
            await REAL[9](0.01)
        assert config.qwen_model == "qwen3.6:35b" and config.vllm_models == f"qwen3.6:35b={V.URL}", config.vllm_models
        assert json.loads(config.vllm_managed)["prev_executor"] == "qwen3.6:35b", "what goes back stays the first one"

        r = await M.vllm_off()
        assert r["ok"] and r["stopped"] and d.container[0] == "exited", r
        assert config.qwen_model == "qwen3.6:35b" and config.vllm_models == "" and \
            not json.loads(config.vllm_managed)["on"], config.vllm_models
        st = await M.vllm_status()
        assert not st["on"] and st["last_model"] == "qwen3.6:35b", st
        print("  ✓ another model while on vLLM: what goes back stays the first; back to Ollama: "
              "the old model, routes cleaned, the container stopped")

        await M._vllm_save({"vllm_models": f"qwen3.6:35b={manual_url}", "vllm_managed": ""})
        r = await M.vllm_setup_start(M._VllmSetupBody(model="qwen3.6:35b"))
        while V.running():
            await REAL[9](0.01)
        assert config.vllm_models == f"qwen3.6:35b={V.URL}" and \
            json.loads(config.vllm_managed)["replaced_route"] == manual_url, config.vllm_managed
        await M.vllm_off()
        assert config.vllm_models == f"qwen3.6:35b={manual_url}", "the route it replaced comes back"
        print("  ✓ a route it replaced (demo2's hand-run qwen) comes back when it is turned off")

        await M._vllm_save({"vllm_models": "", "vllm_managed": "", "executor_model": "qwen3.6:35b"})
        d.container = ("exited", "qwen3.6:35b")
        manual["serves"] = "inkling-2-mini"
        r = await M.vllm_use(M._VllmUseBody(url=manual_url))
        assert r["ok"] and r["model"] == "inkling-2-mini" and config.qwen_model == "inkling-2-mini" and \
            config.vllm_models == f"inkling-2-mini={manual_url}", (r, config.vllm_models)
        st = await M.vllm_status()
        assert st["on"] and not st["managed"] and st["ready"], st
        calls = len(d.calls)
        r = await M.vllm_off()
        assert r["ok"] and not r["stopped"] and config.qwen_model == "qwen3.6:35b" and config.vllm_models == "", r
        assert not [c for c in d.calls[calls:] if c[0] == "stop"], "a server started by hand is left running"
        r = await M.vllm_use(M._VllmUseBody(url="http://127.0.0.1:1"))
        assert not r["ok"] and config.qwen_model == "qwen3.6:35b", r

        # demo, 11/26: local_engine=vllm and the executor named by hand, no record of the one before.
        await M._vllm_save({"local_engine": "vllm", "vllm_base_url": manual_url, "executor_model": "inkling-2-mini",
                            "vllm_models": "", "vllm_managed": ""})
        st = await M.vllm_status()
        assert st["on"] and not st["managed"] and st["url"] == manual_url and st["ready"], st
        r = await M.vllm_off()
        assert r["ok"] and r["fallback"] and config.local_engine == "ollama" and config.qwen_model == "gemma4:e4b", r
        assert not [c for c in d.calls[calls:] if c[0] == "stop"], "still not ours to stop"
        try:
            await M.vllm_setup_start(M._VllmSetupBody(model="llama; rm -rf /"))
            raise AssertionError("an unknown model must be refused")
        except M.HTTPException as exc:
            assert exc.status_code == 400
        print("  ✓ a server started by hand: used, then left running; nothing answering: not used; "
              "an unknown model refused")
    finally:
        for s, t in ((o_srv, o_task), (v_srv, v_task), (m_srv, m_task)):
            s.should_exit = True
            await t


async def main() -> None:
    tmp = tempfile.mkdtemp()
    patch(tmp)
    try:
        print("A. what this machine can do"); await test_check()
        print("B. the install"); await test_install()
        print("C. the endpoints"); await test_endpoints(tmp)
    finally:
        unpatch()
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL VLLM SETUP TESTS PASS")


asyncio.run(main())
