# Deploying vLLM as OpenTeddy's local engine

vLLM is an **optional** alternative to Ollama for the local executor
model, useful on a fleet node that serves **concurrent** load (multiple
operators + watcher loops hitting the same node). On single-stream /
single-user use it does **not** beat Ollama — both are memory-bandwidth-
bound on DGX-Spark-class hardware, and Ollama's Q4 GGUF actually reads
fewer bytes per token. So: vLLM only earns its keep under concurrency.

> macOS / non-CUDA hosts: skip this entirely. vLLM is Linux + NVIDIA
> CUDA only; OpenTeddy hard-gates Darwin to Ollama.

## TL;DR — the happy path

```bash
# on the DGX, on the vLLM branch
cd ~/OpenTeddy
git fetch origin && git reset --hard origin/feat/local-engine-vllm   # see note 7
df -h /                                  # need ~25 GB free for a 7B model + venv

# free GPU memory: stop Ollama for the test (see note 4)
sudo systemctl stop ollama

# install + start vLLM in its OWN venv (handles ninja / python3-dev for you)
sudo bash scripts/setup-vllm.sh --model Qwen/Qwen2.5-7B-Instruct --gpu-mem 0.5 --enforce-eager

# verify (uses OpenTeddy's .venv — it only HTTP-calls vLLM)
OPENTEDDY_LOCAL_ENGINE=vllm VLLM_BASE_URL=http://127.0.0.1:8001 \
  QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct .venv/bin/python scripts/verify-vllm.py
# → "ALL PASS" means OpenTeddy talks to vLLM correctly

# restore Ollama when done (or keep both with --gpu-mem 0.35, see note 4)
sudo systemctl start ollama
```

Then point OpenTeddy at vLLM via **Settings → Model Settings → Local
Inference Engine → vLLM**, or `.env`:

```bash
OPENTEDDY_LOCAL_ENGINE=vllm
VLLM_BASE_URL=http://127.0.0.1:8001
QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## The seven things that bit us (so they don't bite you)

### 1. NEVER `pip install vllm` into OpenTeddy's .venv

vLLM pins aggressive dependency versions — it downgrades `httpx`, which
breaks `chromadb`, and then OpenTeddy won't even boot:

```
AttributeError: module 'httpx' has no attribute 'Limits'
```

`setup-vllm.sh` installs vLLM into a **dedicated** venv
(`/var/lib/openteddy/vllm-venv`); the two processes only ever talk over
HTTP. If you already broke your .venv by hand:

```bash
.venv/bin/pip uninstall -y vllm
.venv/bin/pip install --upgrade 'httpx>=0.27'
.venv/bin/python -c "import main; print('OpenTeddy OK')"
```

### 2. Missing `ninja` → crash ~90 s in

vLLM's FlashInfer sampler JIT-compiles a CUDA kernel at first startup.
Without ninja the model loads fully (~90 s) then dies with
`FileNotFoundError: ... 'ninja'`. `setup-vllm.sh` installs `ninja-build`
automatically; manual fix: `sudo apt install -y ninja-build`.

### 3. Missing Python dev headers → crash ~90 s in

vLLM's Triton sampler JIT-compiles via gcc, which needs `Python.h`:

```
fatal error: Python.h: No such file or directory
```

`setup-vllm.sh` installs `python3-dev` + `build-essential`. Manual fix:
`sudo apt install -y python3-dev build-essential`.

> Notes 2 + 3 both surface only during the post-load *profile run*, ~90 s
> after launch, so they look like late mysterious crashes. The script now
> installs the whole JIT toolchain up front.

### 4. Out of memory — Ollama is hogging the GPU

```
ValueError: Free memory on device cuda:0 (46/119 GiB) is less than
desired GPU memory utilization (0.5, 60 GiB)
```

On a unified-memory box (DGX Spark) Ollama's resident models eat the
pool vLLM wants. Two options:

- **Test cleanly:** `sudo systemctl stop ollama`, run vLLM at `--gpu-mem
  0.5`, `sudo systemctl start ollama` when done. (But note: OpenTeddy's
  planner runs on Ollama, so a full task can't complete while Ollama is
  stopped — `verify-vllm.py` doesn't need Ollama, so it's fine for the
  test.)
- **Coexist:** keep Ollama up, run vLLM at `--gpu-mem 0.35` (≈42 GB) so
  both fit. This is the real fleet config (planner on Ollama, executor on
  vLLM).

### 5. systemd hides the real error (restart loop)

The `openteddy-vllm.service` unit has `Restart=always`, so a failing
engine loads → crashes → restarts → loads again, and the root cause
scrolls past. To see the actual error, run it in the **foreground**:

```bash
sudo systemctl stop openteddy-vllm.service
sudo pkill -9 -f vllm ; sleep 3
/var/lib/openteddy/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --port 8001 --max-model-len 16384 \
  --gpu-memory-utilization 0.5 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes
```

The error prints directly and nothing restarts under you. Once it starts
clean, `sudo systemctl start openteddy-vllm.service` runs it managed.

### 6. Startup hangs silently at "FlashAttention version 2"

vLLM's `torch.compile` step produces no log output for several minutes on
the first run — it looks frozen but is compiling. `--enforce-eager` skips
torch.compile + CUDA-graph capture entirely: startup in ~1-2 min instead
of 10+, at ~10-20 % runtime cost. Use it for testing; drop it for a
tuned production server once you've confirmed everything works.

### 7. `git pull` won't update the branch

The vLLM branch was rebased + force-pushed, so a plain `git pull` on an
older local copy diverges and refuses to fast-forward. Force-align to the
remote:

```bash
git fetch origin
git reset --hard origin/feat/local-engine-vllm
git log --oneline -1                 # confirm you're on the latest
grep -c vllm-venv scripts/setup-vllm.sh   # >0 means you have the dedicated-venv version
```

## Benchmarking: single-stream vs concurrent

Single-request tok/s tells you little — that's bandwidth-bound and Ollama
often matches or beats vLLM there. vLLM's win is **concurrent throughput**
(continuous batching). Test that:

```bash
# vLLM: 8 requests at once → total throughput
/var/lib/openteddy/vllm-venv/bin/python -c "
import asyncio, httpx, time
URL='http://127.0.0.1:8001/v1/chat/completions'
async def one(c,i):
    r=await c.post(URL,json={'model':'Qwen/Qwen2.5-7B-Instruct','messages':[{'role':'user','content':f'Write 100 words about topic {i}'}],'max_tokens':150})
    return r.json()['usage']['completion_tokens']
async def main():
    async with httpx.AsyncClient(timeout=180) as c:
        t=time.time(); toks=await asyncio.gather(*[one(c,i) for i in range(8)]); dt=time.time()-t
        print(f'vLLM 8-parallel: {sum(toks)} tok / {dt:.1f}s = {sum(toks)/dt:.0f} tok/s total')
asyncio.run(main())
"
```

Compare against Ollama running the same 8 requests (Ollama queues them
serially, vLLM batches them). The gap there — not the single-request
number — is what justifies vLLM on a fleet node.

## Uninstall / revert to Ollama

```bash
sudo bash scripts/setup-vllm.sh --uninstall     # removes the systemd service
sudo rm -rf /var/lib/openteddy/vllm-venv        # reclaim the venv
# then in OpenTeddy Settings, switch Local Inference Engine back to Ollama
```
