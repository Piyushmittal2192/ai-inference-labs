# Lab 1.1 — Async GPU Serving and the Anatomy of a Request: Prefill vs. Decode, TTFT and ITL

**Level 1 · Foundations** · Hardware: 1× RTX 2080 (8 GB, Turing sm_75) · Model: `Qwen/Qwen2.5-0.5B-Instruct` (fp16)

This lab merges two goals into one build:

1. **Serving foundations.** A production-style async API server in Docker: health semantics, bounded queues, graceful shutdown, a non-root image with no weights baked in.
2. **Engine mechanics.** A hand-written prefill + decode loop that times each phase on the GPU, a streaming benchmark that measures **TTFT**, **ITL** and **TPOT** from the client, and a log parser that extracts the same numbers from inside the server.

The server speaks the OpenAI `/v1/completions` protocol on purpose. In Level 2 you will point the exact same benchmark at vLLM and see, with your own numbers, what continuous batching and PagedAttention buy you over this baseline.

> **How to read this runbook.** Every step has **What**, **How** (copy-paste commands) and **Why**, four layers deep. Layer 1 is the practical reason; each layer below it asks "and why is *that* true?" until it reaches the hardware or a first principle. Read Layer 1 the first time through; come back for Layers 2–4 when you are preparing for interviews.

---

## Contents

- [The request you are about to measure](#the-request-you-are-about-to-measure)
- [Architecture of the code](#architecture-of-the-code)
- [File map](#file-map)
- [Part A — Prepare the machine](#part-a--prepare-the-machine) (Steps 0–1)
- [Part B — Build and run the server](#part-b--build-and-run-the-server) (Steps 2–8)
- [Part C — Talk to it](#part-c--talk-to-it) (Steps 9–10)
- [Part D — Measure prefill vs. decode](#part-d--measure-prefill-vs-decode) (Steps 11–15)
- [Part E — Resilience checks](#part-e--resilience-checks) (Steps 16–17)
- [Gatekeeper: Verification test](#gatekeeper-verification-test)
- [Gatekeeper: Chaos / Tuning Challenge](#gatekeeper-chaos--tuning-challenge)
- [Checkpoint questions](#checkpoint-questions)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## The request you are about to measure

```
client          server (event loop)                      GPU worker thread
  │  POST /v1/completions
  │ ───────────────▶ tokenize ─▶ admit/queue ─▶ submit ─▶ PREFILL: 1 forward pass over all N prompt tokens
  │                                                          ├─ writes the KV cache for N tokens
  │                                                          └─ samples token #1 ─────────┐
  │ ◀──────────── SSE chunk #1 ◀──────────── call_soon_threadsafe ◀───────────────────────┘
  │                                                       DECODE step: 1 forward pass over 1 token (+KV cache)
  │ ◀──────────── SSE chunk #2 ◀─────────────────────────────┘ samples token #2
  │ ◀──────────── SSE chunk #3 ◀───────────────────────────── ... repeated until EOS / max_tokens
  │ ◀──────────── final chunk (server_timings) + usage + [DONE]

  ├────────────────────────── TTFT ─────────────────────────┤├─ ITL ─┤├─ ITL ─┤ ...
  ├───────────────────────────────────────────── E2E ──────────────────────────────────┤
```

| Metric | Definition | Dominated by |
|---|---|---|
| **TTFT** — time to first token | request sent → first text chunk received | queueing + **prefill** |
| **ITL** — inter-token latency | gap between consecutive text chunks | one **decode** step |
| **TPOT** — time per output token | `(E2E − TTFT) / (output_tokens − 1)` | mean decode step |
| **E2E** | request sent → stream finished | `TTFT + (output_tokens − 1) × TPOT` |
| **Throughput** | output tokens / second across all requests | batching (or here, the lack of it) |

---

## Architecture of the code

```
                    ┌──────────────── one Python process (uvicorn, 1 worker) ─────────────────┐
HTTP ──▶ uvloop ──▶ │ FastAPI (event loop thread)                                               │
                    │   /livez /readyz /stats /v1/models      ← never touch the GPU             │
                    │   /v1/completions:                                                       │
                    │     tokenize → validate MAX_MODEL_LEN → scheduler.submit()               │
                    │                                           │  bounded queue + semaphore   │
                    │                                           ▼                              │
                    │                                  ThreadPoolExecutor ("gpu-worker")       │
                    │                                    engine.generate(): prefill, decode…   │
                    │     asyncio.Queue  ◀── loop.call_soon_threadsafe(token) ──┘              │
                    │     StreamingResponse (SSE) ◀── reads the queue                          │
                    │                                                                          │
                    │ model-loader daemon thread: downloads + loads weights + warmup at boot   │
                    └──────────────────────────────────────────────────────────────────────────┘
```

Key design decisions (each is explained in depth in the steps):

- **The event loop never runs model code.** A blocked event loop cannot stream tokens or answer health checks. The chaos challenge proves it.
- **The GPU slot is tied to the worker thread, not the HTTP request.** If a client disconnects mid-generation, the slot is released only when the thread actually stops, so a new request can never pile onto a GPU that is still busy.
- **Admission control before work.** When the queue is full, the server returns `503 + Retry-After` immediately instead of building an unbounded backlog.
- **Weights are data, not image layers.** They live in a Docker volume, and the image stays reproducible and smaller.
- **Phases are run by hand.** `model.generate()` hides prefill and decode inside one call, so the engine runs them explicitly and puts a GPU sync point (`.item()`) at the end of each timed section.

## File map

```
lab-1.1-gpu-serving/
├── Dockerfile              multi-stage, CUDA 12.6 torch wheels, non-root, tini, HEALTHCHECK
├── .dockerignore
├── requirements.txt        pinned server deps (torch is installed separately in the Dockerfile)
├── server/
│   ├── config.py           every env var, validated at boot
│   ├── protocol.py         engine ↔ API contract: SamplingParams, GenerationStats, errors
│   ├── detokenizer.py      incremental detokenizer (never emits half a UTF-8 character)
│   ├── engine.py           model load, explicit PREFILL + DECODE loop, per-phase timing, OOM recovery
│   ├── scheduler.py        admission control, bounded queue, worker thread, slot accounting
│   ├── schemas.py          strict OpenAI /v1/completions request schema
│   └── main.py             FastAPI app: health, stats, models, completions (JSON + SSE)
└── bench/
    ├── requirements.txt    httpx (client side only)
    ├── bench.py            async streaming benchmark: prompt-sweep, output-sweep, load
    └── parse_logs.py       extracts engine-side TTFT / prefill / ITL from the server log
```

---

## Part A — Prepare the machine

Run every command in this runbook from a **Linux shell**: native Linux, or the Ubuntu distro inside WSL2 if your RTX 2080 lives in a Windows PC. PowerShell's `curl` is a different program.

### Step 0 — Confirm the host driver sees the GPU

**What:** Check that the NVIDIA driver is installed, sees the RTX 2080, and supports CUDA 12.6 or newer. Also check how much VRAM is already in use.

**How:**
```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,compute_cap --format=csv
```
You want: name `NVIDIA GeForce RTX 2080`, `compute_cap` `7.5`, and a header line showing `CUDA Version: 12.6` or higher. Note `memory.used`. If a monitor is plugged into this GPU, the desktop already uses some of your 8 GB.

**Why:**
1. **Practical.** The image ships CUDA 12.6 user-space libraries inside the PyTorch wheels. They only work if the host driver supports CUDA ≥ 12.6. Checking now turns a confusing runtime error later into an obvious fix now.
2. **Mechanism.** CUDA comes in two halves.
   - The *user-mode* libraries (`libcudart`, cuBLAS, cuDNN) ship inside your container.
   - The *kernel-mode* driver (`nvidia.ko`) and its user-space companion `libcuda.so` live only on the host.

   The runtime libraries call into `libcuda`, and the driver version caps which runtime versions it can serve. The "CUDA Version" that `nvidia-smi` prints is that cap, not an installed toolkit.
3. **System.** A container shares the host kernel, so it can never bring its own kernel driver. GPU containers are therefore always "driver from the host, libraries from the image". Version skew between those two halves is the most common cause of `CUDA driver version is insufficient` and `no kernel image is available` errors in production.
4. **First principles.** The GPU is a PCIe device owned by exactly one kernel module. That module maps the GPU's memory into processes, schedules its work and switches contexts. User space can only reach it through `ioctl` calls on `/dev/nvidia*`, so every process, containerized or not, must speak that one driver's ABI. VRAM used by your desktop compositor is VRAM your model cannot use, and on an 8 GB card that is a real budget line.

### Step 1 — Confirm containers can see the GPU

**What:** Prove that Docker can hand the GPU to a container *before* you spend minutes building an image.

**How:**
```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```
You should see the same table as Step 0. If you get `could not select device driver "" with capabilities: [[gpu]]`, the NVIDIA Container Toolkit is missing (see [Troubleshooting](#troubleshooting)).

**Why:**
1. **Practical.** It is a 5-second test of the whole GPU-in-container chain, using a stock image that has no CUDA in it at all.
2. **Mechanism.** `--gpus all` makes Docker run the NVIDIA container runtime hook before the container starts. The hook bind-mounts the device nodes into the container (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`), along with the host's driver libraries and the `nvidia-smi` binary. That is why `nvidia-smi` exists inside a plain Ubuntu image. On WSL2, Docker Desktop does the equivalent through the paravirtualized `/dev/dxg` device.
3. **System.** Containers are Linux namespaces plus cgroups. Devices are hidden by default, and the hook adds the GPU device nodes to what the container may open. In Level 3, the Kubernetes NVIDIA device plugin does exactly this for each pod. Time-slicing is simply that plugin advertising one physical GPU as N schedulable units.
4. **First principles.** GPU isolation is much weaker than CPU or memory isolation. There is no cgroup controller for VRAM or for streaming multiprocessors (SMs). Any process that can open the device can allocate all of its memory. That is why this lab caps memory *inside* the process (`GPU_MEMORY_FRACTION`), and why time-slicing in Level 3 shares compute but gives no memory isolation at all.

---

## Part B — Build and run the server

### Step 2 — Get the code

**What:** Clone the repo and move into the lab folder. Every later command assumes this folder.

**How:**
```bash
git clone https://github.com/Piyushmittal2192/ai-inference-labs.git
cd ai-inference-labs/labs/lab-1.1-gpu-serving
ls -R
```

**Why:**
1. **Practical.** The Dockerfile's `COPY` paths and the benchmark's defaults are relative to this folder.
2. **Mechanism.** `docker build .` sends this folder, the *build context*, to the Docker daemon. `.dockerignore` strips `bench/`, results and docs from that context.
3. **System.** A small, deterministic build context means faster builds and no secrets accidentally copied into images. It also means a code change in `bench/` never invalidates the server image's cache.
4. **First principles.** Reproducibility is a property of *every* input: code (git SHA), dependencies (pinned versions), weights (pinned model revision), and the build context itself. Remove any one and two machines can produce different artifacts from the "same" source.

### Step 3 — Create the model-cache volume

**What:** Create a named Docker volume that will hold the downloaded weights.

**How:**
```bash
docker volume create hf-cache
docker volume inspect hf-cache
```

**Why:**
1. **Practical.** The weights (~1 GB) download once and survive every rebuild, restart and `docker rm` of the container.
2. **Mechanism.** A named volume is a directory Docker manages outside the container's layered filesystem. It is mounted at `/models`, and `HF_HOME=/models` makes Hugging Face cache files there. The first time an empty named volume is mounted over a path that exists in the image, Docker copies that path's contents *and ownership* into the volume. So `/models` arrives owned by the non-root `app` user, which can write to it.
3. **System.** This separates artifacts with different lifecycles. Code changes daily, weights change per model version, and runtime state is disposable. Weights baked into an image add a gigabyte or more per model per version, and every node must pull them again. In Level 3 the same idea becomes a PersistentVolumeClaim or a node-local cache.
4. **First principles.** Cold-start time is roughly image pull + weight load + warmup, and an autoscaler cannot react faster than one cold start. Every gigabyte on that critical path costs seconds. Large files written into a copy-on-write image layer are also slower to read than files on a plain volume.

### Step 4 — Build the image

**What:** Build `lab11-gpu:1.1` with pinned dependencies and CUDA 12.6 PyTorch wheels.

**How:**
```bash
docker build -t lab11-gpu:1.1 .
```
The first build downloads several GB of CUDA libraries, so expect it to take a while. Rebuild after editing only `server/*.py` and it should finish in seconds, because every layer above `COPY server/` is reused.

**Why:**
1. **Practical.** You get one reproducible artifact that runs the same on any machine with an NVIDIA driver.
2. **Mechanism.**
   - *Multi-stage build:* the `builder` stage creates a virtualenv, and the `runtime` stage copies in only that virtualenv.
   - *Layer order:* torch is installed in its own layer *before* `requirements.txt` is copied, so editing the requirements does not re-download torch.
   - *Cache mounts:* `--mount=type=cache` keeps pip's download cache on the build machine instead of baking it into a layer.
3. **System.** Docker caches each layer by a key built from the instruction and the checksums of the files it reads. Put slow, stable layers first and fast-changing ones last. Keeping build tooling out of the runtime image shrinks both its size and its attack surface. The pins (FastAPI, pydantic, uvicorn, transformers 4.57.6, torch 2.9.0, the model's commit SHA) mean a rebuild next month produces the same software.
4. **First principles.** NVIDIA's "devel" images carry the compiler (`nvcc`), headers and static libraries. You only need those to *compile* CUDA kernels, and inference only *runs* precompiled ones, which the PyTorch wheel already contains. The `cu126` wheel includes machine code (SASS) built for `sm_75`, your Turing GPU, so nothing has to be JIT-compiled from PTX on first use. Because a GPU only runs code compiled for its own architecture, a wheel without your `sm_xx` produces `no kernel image is available for execution on the device`.

### Step 5 — Inspect what you built

**What:** Check the image's size and runtime user, and confirm PyTorch sees the GPU and has kernels for `sm_75`.

**How:**
```bash
docker image ls lab11-gpu:1.1
docker history --format 'table {{.Size}}\t{{.CreatedBy}}' lab11-gpu:1.1 | head -n 15
docker run --rm --entrypoint id lab11-gpu:1.1
docker run --rm --gpus all --entrypoint python lab11-gpu:1.1 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0)); print('compiled for', torch.cuda.get_arch_list())"
```
Expect `uid=10001(app)` and `capability (7, 5)`, with `sm_75` in the compiled-for list. Write down the image size; you will compare it with the vLLM image in Level 2.

**Why:**
1. **Practical.** Each check fails fast on a different class of problem: a bloated image, root execution, a CPU-only torch, or a missing GPU architecture.
2. **Mechanism.** `docker history` shows which instruction produced each byte. The torch layer dominates, and that is expected. `get_arch_list()` lists the GPU architectures the wheel was compiled for, which is the definitive answer to "will this run on my card?"
3. **System.** On a cluster, image size becomes pull time on every new node, which becomes cold start, which becomes how fast you can scale up. Running as non-root means a compromised process cannot write to system paths.
4. **First principles.** GPU binaries are not portable the way x86 binaries are. Each architecture generation has its own instruction set, and forward compatibility exists only through PTX, which the driver compiles at load time and then caches. Knowing your `sm` number is to GPU serving what knowing your CPU architecture is to cross-compiling.

### Step 6 — Run the server

**What:** Start the container in the background with the GPU, the port and the weight cache attached.

**How:**
```bash
docker run -d --name lab11 --gpus all -p 8000:8000 -v hf-cache:/models lab11-gpu:1.1
docker ps --filter name=lab11
```

**Why:**
1. **Practical.** The flags mean:
   - `--gpus all` attaches the GPU.
   - `-p 8000:8000` publishes the API on your machine.
   - `-v hf-cache:/models` reuses the cached weights.
   - `-d` runs detached, so your terminal is free.
2. **Mechanism.** The image's entrypoint is `tini`, which runs `uvicorn server.main:app` with one worker and a 30 s graceful-shutdown timeout. Uvicorn starts the FastAPI lifespan, which logs the config and starts the model-loader thread. Uvicorn begins accepting connections immediately, *while* the weights are still loading.
3. **System.** One worker per GPU is deliberate. Each uvicorn worker is a separate process with its own copy of the model, so four workers would mean four copies of the weights competing for 8 GB. You scale serving with *replicas*, not with web-framework workers.
4. **First principles.** PID 1 in a Linux PID namespace is special. The kernel does not apply the default action (terminate) to signals sent to PID 1 unless that process installed a handler, and PID 1 must also reap orphaned child processes. `tini` is a tiny init that forwards `SIGTERM` to uvicorn and reaps zombies, so `docker stop` (and Kubernetes pod termination) leads to a clean, graceful shutdown instead of a `SIGKILL` 10 seconds later.

### Step 7 — Watch liveness and readiness diverge during startup

**What:** While the model loads, `/livez` says the process is fine and `/readyz` says "not yet". Watch both until ready.

**How:**
```bash
curl -s -w '  HTTP %{http_code}\n' localhost:8000/livez
curl -s -w '  HTTP %{http_code}\n' localhost:8000/readyz
until curl -sf localhost:8000/readyz > /dev/null; do echo "$(date +%T) not ready yet"; sleep 3; done
curl -s localhost:8000/readyz; echo
docker inspect --format '{{.State.Health.Status}}' lab11
```
Expected during loading: `/livez` → `200 {"status":"alive","phase":"loading"}`, and `/readyz` → `503 {"status":"loading"}`. After loading, `/readyz` → `200` and Docker health shows `healthy`. On the very first run the weight download makes the loading window longer.

**Why:**
1. **Practical.** The two endpoints answer different questions:
   - Liveness: *should this process be restarted?*
   - Readiness: *should traffic be sent here right now?*

   A model that is still loading is alive but not ready.
2. **Mechanism.** The weights load on a *daemon* thread, so the event loop answers health checks the whole time. Daemon status matters: if you stop the container mid-download, a non-daemon thread would hold the interpreter open until the download finished. A failed load flips `/livez` to `503` too, because nothing inside the process can recover from it.
3. **System.** In Kubernetes (Level 3), a failing *liveness* probe kills and restarts the container, while a failing *readiness* probe only removes it from the Service's endpoints. If liveness depended on the model being loaded, every slow cold start would be killed before it finished, and restarted, and killed again: a restart loop that looks like a model bug.
4. **First principles.** Health checks are a control loop. The orchestrator observes a signal and takes an action. If the signal conflates *slow* with *dead*, the controller's action (restart) makes the problem worse (another slow start). Signals must map one-to-one onto the actions they trigger.

### Step 8 — Read the startup logs

**What:** Confirm what the server actually did on your GPU: device, dtype, attention backend, memory cap, weight footprint and warmup.

**How:**
```bash
docker logs lab11 2>&1 | grep -E "config:|gpu:|pre-Ampere|weights ready|warmup|model_ready"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
curl -s localhost:8000/stats | python3 -m json.tool
```
Look for `capability=7.5`, `dtype=torch.float16`, `attention=sdpa`, `allocated_mb≈950`, a warmup time and `event=model_ready load_s=…`. Stop and start the container once (`docker restart lab11`, then wait for ready) and compare `load_s`. The second start reads weights from the volume instead of the network.

**Why:**
1. **Practical.** Logs are the ground truth. Defaults like dtype and attention backend vary by GPU and library version, so verify them; don't assume.
2. **Mechanism.**
   - *fp16:* Qwen2.5 ships bf16 weights, but Turing tensor cores support fp16 and not bf16, so the engine loads in fp16.
   - *SDPA:* PyTorch's `scaled_dot_product_attention` picks a kernel at runtime. FlashAttention-2 needs Ampere or newer, so on `sm_75` SDPA uses its memory-efficient kernel instead.
   - *Memory cap:* `GPU_MEMORY_FRACTION=0.60` caps PyTorch's caching allocator at 60 % of VRAM.
   - *Warmup:* runs two throwaway requests so the first real request does not pay one-time CUDA costs.
3. **System.** Those one-time costs include:
   - CUDA context creation;
   - loading kernels into the GPU (recent CUDA versions load kernels lazily, on first use);
   - cuBLAS picking GEMM algorithms per matrix shape;
   - the caching allocator growing its pool.

   Without warmup they all land on the first user's TTFT, and in a fleet they land on every freshly scaled replica. `nvidia-smi` shows more memory used than `allocated_mb`. The difference is the CUDA context plus the allocator's cached-but-free blocks.
4. **First principles.** fp16 has 5 exponent bits, so its largest value is 65504. bf16 has 8 exponent bits, matching fp32's range with less precision. Models trained in bf16 can produce activations that overflow fp16. Small models like this one are generally fine in fp16, but if you ever see `inf`/`nan` logits on a pre-Ampere GPU, numeric range is the first suspect. The memory cap also illustrates a concept you will meet in vLLM: `--gpu-memory-utilization`. Here, the fraction is a *ceiling*; in vLLM it is a *reservation* that the KV cache is pre-allocated to fill.

---

## Part C — Talk to it

### Step 9 — First requests: models, non-streaming, streaming

**What:** List the model, run one non-streaming completion and one streaming completion, and read the `server_timings` the server attaches.

**How:**
```bash
curl -s localhost:8000/v1/models | python3 -m json.tool

curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The difference between prefill and decode in LLM inference is", "max_tokens": 48, "temperature": 0}' \
  | python3 -m json.tool

curl -sN localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "Write one sentence about GPUs:", "max_tokens": 24, "temperature": 0, "stream": true, "stream_options": {"include_usage": true}}'
```
In the JSON response, find `usage` and `server_timings` (`prefill_ms`, `itl_mean_ms`, `decode_tokens_per_s`, `peak_memory_mb`). In the stream, every `data:` line is one chunk. The second-to-last chunk carries `finish_reason` and `server_timings`, the last carries `usage`, and then comes `data: [DONE]`.

**Why:**
1. **Practical.** Streaming is how users experience LLMs, and TTFT only exists in streaming mode. In non-streaming mode the client sees nothing until E2E.
2. **Mechanism.** The response is Server-Sent Events: `Content-Type: text/event-stream`, each event a `data: …` line followed by a blank line, sent over one HTTP/1.1 response using chunked transfer encoding. `curl -N` turns off curl's output buffering so you see chunks arrive. Under the hood, each token travels from the GPU worker thread to the event loop through `loop.call_soon_threadsafe`, into an `asyncio.Queue`, and out of the `StreamingResponse` generator.
3. **System.** Matching the OpenAI wire format lets one client (`bench.py`) benchmark this server, vLLM, SGLang or any OpenAI-compatible endpoint. That makes the comparison in Level 2 apples-to-apples. The `server_timings` field is a non-standard extension that standard clients simply ignore.
4. **First principles.** Perceived latency is set by when the *first* useful output appears, not by when the last one does, and people read at a few tokens per second. A server whose TTFT is 200 ms and ITL is 30 ms feels faster than one that returns everything at once in 2 s, even though the second has better E2E. That is why serving SLOs are written per phase: TTFT p99, and ITL or TPOT p99.

### Step 10 — Contract tests: the server rejects bad requests loudly

**What:** Send four bad requests: an unsupported field, too many tokens, the wrong model, and a prompt too long for `MAX_MODEL_LEN`. Each should fail with a precise error code.

**How:**
```bash
curl -s -w '  HTTP %{http_code}\n' localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"prompt": "hi", "n": 2}'
curl -s -w '  HTTP %{http_code}\n' localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"prompt": "hi", "max_tokens": 5000}'
curl -s -w '  HTTP %{http_code}\n' localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"prompt": "hi", "model": "gpt-4"}'
python3 -c 'import json; print(json.dumps({"prompt": "hello " * 4000, "max_tokens": 200}))' \
  | curl -s -w '  HTTP %{http_code}\n' localhost:8000/v1/completions -H 'Content-Type: application/json' -d @-
```
Expected: `422` (extra field `n`), `422` (`max_tokens` > 1024), `404` (wrong model), `400` (prompt + `max_tokens` > 4096).

**Why:**
1. **Practical.** A server that silently ignores `n: 2` returns a 200 that lies. The client believes it got two samples. Rejecting unknown fields turns silent wrongness into a visible error.
2. **Mechanism.** pydantic validates the body with `extra="forbid"` and numeric bounds before any GPU work. The context-length check runs right after tokenization, because only the tokenizer knows how many tokens a string is.
3. **System.** `MAX_MODEL_LEN` bounds the worst-case memory a single request can demand. Prompt and output both grow the KV cache, so admitting a request you cannot finish wastes GPU time and then fails anyway. In vLLM, `--max-model-len` also determines how many sequences fit in memory at once, which makes it one of your most important knobs on an 8 GB card.
4. **First principles.** Every token in a sequence stores a key and a value vector at every layer, for every KV head. Memory per request therefore grows *linearly* with sequence length, while attention compute during prefill grows with its *square*. Admission control is the only point where those costs can be refused cheaply: before the GPU spends anything.

---

## Part D — Measure prefill vs. decode

### Step 11 — Set up the benchmark client on the host

**What:** Create a Python virtualenv on your machine (outside the container) with `httpx`.

**How:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r bench/requirements.txt
python bench/bench.py --help
```

**Why:**
1. **Practical.** The load generator must not share the GPU server's CPU and event loop. Otherwise you measure the client's contention as the server's latency.
2. **Mechanism.** `bench.py` uses one `httpx.AsyncClient` with keep-alive connections and reads the SSE stream line by line, timestamping every text chunk with `time.perf_counter()`. It sets `trust_env=False`, so a corporate `HTTP(S)_PROXY` never silently sits between the benchmark and `localhost`.
3. **System.** Before measuring, the client **calibrates** the prompt length. It sends two prompts of known word counts, reads the server-reported `prompt_tokens`, and fits `tokens = a + b × words`. That lets it hit target token lengths for *any* tokenizer without shipping the tokenizer to the client, so the same script works against vLLM unchanged. Every prompt is random common words behind a unique `Request <nonce>:` prefix, so a server with prefix caching (vLLM, Level 2) cannot reuse work between requests and quietly flatter the numbers.
4. **First principles.** A benchmark is an experiment, so it needs controlled variables. Here those are:
   - temperature 0;
   - `ignore_eos` so every request generates *exactly* `max_tokens`;
   - unique prompts;
   - warmup requests excluded from the results.

   Change one factor at a time (prompt length *or* output length *or* concurrency), otherwise you cannot attribute the effect.

### Step 12 — Benchmark A: the prompt sweep isolates PREFILL

**What:** Vary the prompt length (64 → 3,584 tokens) with a small fixed output (32 tokens) at concurrency 1. TTFT is then almost pure prefill.

**How:**
```bash
python bench/bench.py prompt-sweep
```
The defaults are `--lengths 64,256,512,1024,2048,3584 --output-tokens 32 --repeats 5`. The table shows, per length:
- client TTFT p50/p90;
- server `prefill_ms`;
- prefill throughput (tokens/s);
- ITL p50/p99;
- peak GPU memory.

Under the table is a linear fit, `TTFT_ms = a + b × prompt_tokens`, plus the implied marginal prefill throughput. Results are saved to `results/prompt-sweep-*.csv` and `.json`.

**Why:**
1. **Practical.** TTFT is the user-facing number most sensitive to prompt length: RAG contexts, long system prompts, chat history. You need its slope on *your* hardware to budget context length against a latency SLO.
2. **Mechanism.** Prefill is one forward pass in which all N prompt tokens go through every weight matrix *together*, as large matrix–matrix multiplies (GEMMs). The fit's intercept `a` is the fixed cost every request pays: HTTP, tokenization, kernel launches and sampling. The slope `b` is the marginal cost per prompt token.
3. **System.** Watch the ITL column as the prompt grows. Decode steps read the KV cache for every previous token, so ITL *can* grow with context, but by how much depends on how big the KV cache is relative to the weights. For this model, the [checkpoint questions](#checkpoint-questions) ask you to compute that ratio from `config.json` and explain what you observe.
4. **First principles — the roofline.** A GPU has a compute ceiling (FLOP/s) and a memory-bandwidth ceiling (bytes/s). The RTX 2080's GDDR6 moves about 448 GB/s.
   - A forward pass costs about `2 × parameters` FLOPs per token, and it must read every weight (`2 bytes × parameters` in fp16) at least once per pass.
   - With N tokens in the pass, arithmetic intensity is about N FLOPs per byte of weights.
   - Prefill, with N in the hundreds or thousands, is therefore high-intensity and **compute-bound**.
   - Prefill throughput in tokens/s is high, and roughly constant once N is large enough to saturate the tensor cores, so TTFT grows roughly linearly in N.
   - At much longer contexts the attention term, which grows with N², starts to bend the curve upward.

### Step 13 — Benchmark B: the output sweep isolates DECODE

**What:** Hold the prompt at 256 tokens and vary output length (16 → 512). E2E grows linearly with output tokens, and the slope *is* the per-token decode cost.

**How:**
```bash
python bench/bench.py output-sweep
```
Read the table (TTFT, TPOT, ITL p50/p99, E2E, server decode tokens/s) and the fit `E2E_ms = a + b × output_tokens`. The slope `b` should match TPOT and ITL, and R² should be very close to 1.

**Why:**
1. **Practical.** For long answers, decode dominates E2E. The slope tells you exactly what each extra output token costs, which turns `max_tokens` into a latency budget.
2. **Mechanism.** Each decode step is one forward pass over **one** token that reuses the KV cache. The engine times each step individually and ends each timed section with `.item()`, a GPU-to-CPU copy that forces all queued kernels to finish, so the timing is real GPU time and not just kernel-launch time. The regression intercept recovers the fixed per-request cost (≈ TTFT). That gives you two independent measurements of the same quantity, and if they disagree, something in your setup is wrong.
3. **System.** At batch size 1, every decode step must read the full ~1 GB of weights from VRAM to produce a single token. That makes the step **memory-bandwidth-bound**: about 2 FLOPs per 2 bytes read, roughly 1 FLOP/byte, far below what the tensor cores could do. Batching is the only way to reuse those weight reads. One pass reads the weights once and produces a token for *each* sequence in the batch. That idea drives everything vLLM does with continuous batching (Level 2).
4. **First principles.** Bandwidth sets a hard floor: `time per step ≥ bytes read ÷ bandwidth ≈ 1 GB ÷ 448 GB/s ≈ 2.2 ms`. Compare your measured ITL with that floor. The gap is *overhead*:
   - Python executing the model code;
   - launching hundreds of small kernels per step (each launch costs microseconds of CPU time);
   - sampling;
   - copying the token to the CPU.

   For small models, overhead, not bandwidth, is often the real decode bottleneck, and that is exactly what CUDA graphs (recording all of a step's kernel launches once and replaying them with one launch) remove. You will see vLLM use them in Level 2.

### Step 14 — Benchmark C: concurrent load exposes queueing

**What:** Send many streams at once to a server that runs one request at a time, and watch TTFT turn into a queueing metric.

**How:**
```bash
python bench/bench.py load --concurrency 1 --requests 16
python bench/bench.py load --concurrency 4 --requests 32 --probe
curl -s localhost:8000/stats | python3 -m json.tool
```
Compare the two runs: TTFT p50/p99, TPOT, throughput (output tok/s), the `srv_queue` row, and the `/livez` probe line.

**Why:**
1. **Practical.** A single-user benchmark says nothing about multi-user latency. You need both numbers before quoting either one.
2. **Mechanism.** `MAX_CONCURRENCY=1` means one generation at a time. With 4 concurrent clients, each new request waits for the ones ahead of it to *fully* finish, both prefill and all decode steps. TTFT therefore grows with queue position, while TPOT stays flat (once running, each request has the GPU to itself) and throughput barely changes. The `/livez` probe keeps answering quickly because the event loop never touches the GPU.
3. **System.** **Little's law** holds for any stable system: `requests in system = arrival rate × time in system`. With no batching, throughput is capped at one request's worth of decode. Adding load raises latency without raising throughput, which is the definition of a saturated system. The server's bounded queue (`MAX_QUEUE_DEPTH`) and `QUEUE_TIMEOUT_S` turn that saturation into fast `503`s instead of unbounded waits.
4. **First principles.** Because decode at batch 1 is memory-bound (Step 13), the GPU's arithmetic units sit mostly idle during it. Running 4 sequences together would cost only a little more time per step than running 1, since the weights are read once either way, but this engine cannot batch. The gap between the throughput you measure here and what the hardware could deliver is what Level 2 closes.

### Step 15 — Extract TTFT from the engine logs

**What:** Parse the server's per-request log lines to see TTFT from *inside* the server and split it into queue, tokenize and prefill. Then compare with the client's view.

**How:**
```bash
docker logs lab11 2>&1 | grep -m 3 "event=request_done"
docker logs lab11 2>&1 | python bench/parse_logs.py
docker logs lab11 2>&1 | python bench/parse_logs.py --group-by completion_tokens
docker logs lab11 2>&1 | python bench/parse_logs.py --csv results/engine-log.csv
```

**Why:**
1. **Practical.** Client-side TTFT tells you *that* something is slow. The engine log tells you *which phase* is slow: queueing, tokenization or prefill.
2. **Mechanism.** Every finished request writes one `key=value` line with `event=request_done`, its token counts and its phase timings. `parse_logs.py` groups requests whose prompt lengths are within 10 % of each other and prints percentiles per group.
3. **System.** `client TTFT − server TTFT` is your *serving overhead*: HTTP parsing, SSE framing, the loopback network and the client's own event loop. On localhost it should be small and roughly constant. In production it includes load balancers and real networks, and that is the first place to look when the engine says "fast" and users say "slow".
4. **First principles.** Measure at more than one vantage point, because each one hides different things. A client can't see queueing inside the server, and a server can't see time spent in the network or in the client. Structured, machine-parseable logs (one event per line, `key=value`) are the bridge to real observability: in Level 4 these same fields become Prometheus histograms.

---

## Part E — Resilience checks

### Step 16 — Disconnects, graceful shutdown, warm restart

**What:** Prove three behaviors:
- a client that disconnects mid-stream stops the GPU work;
- `docker stop` lets an in-flight stream finish;
- a restart reloads weights from the volume.

**How:**
```bash
# 1) disconnect: curl gives up after 1 second on an 800-token request
curl -sN -m 1 localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "Count from one to one thousand:", "max_tokens": 800, "ignore_eos": true, "stream": true}' > /dev/null
sleep 1
docker logs lab11 2>&1 | grep request_cancelled | tail -n 1
curl -s localhost:8000/stats | python3 -c "import sys, json; s = json.load(sys.stdin)['scheduler']; print({k: s[k] for k in ('in_flight', 'completed', 'failed')})"

# 2) graceful shutdown: stop the container while a 300-token stream is running
curl -sN localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "Tell a long story:", "max_tokens": 300, "ignore_eos": true, "stream": true}' | tail -n 2 &
sleep 1; time docker stop -t 40 lab11; wait
docker logs lab11 2>&1 | grep -E "Shutting down|Waiting for connections|event=shutdown" | tail -n 3

# 3) warm restart: weights come from the volume this time
docker start lab11
until curl -sf localhost:8000/readyz > /dev/null; do sleep 2; done
docker logs lab11 2>&1 | grep model_ready | tail -n 2
```
Expected:
1. A `request_cancelled` line, and `in_flight` back to 0.
2. The in-flight stream still ends in `data: [DONE]` before the container exits.
3. A second `load_s` that is shorter than the first-ever one.

**Why:**
1. **Practical.** Users close tabs, load balancers drain nodes, and deploys restart pods. Each of those is a request interrupted on purpose, and each must neither waste the GPU nor drop a response.
2. **Mechanism.**
   - *Disconnect:* when the client goes away, Starlette stops the streaming generator, whose `finally` block sets a `threading.Event`. The decode loop checks that event before every step and stops within one step.
   - *Shutdown:* `docker stop` sends `SIGTERM` (through `tini`), and uvicorn stops accepting new connections and waits up to 30 s for open ones to finish before running lifespan shutdown. `-t 40` gives Docker more patience than uvicorn's own 30 s limit, so Docker does not `SIGKILL` it first.
3. **System.** The scheduler releases a GPU slot when the *worker thread* finishes, not when the HTTP request ends. A disconnected request's thread may still be inside a forward pass. Freeing its slot early would start a new request on a GPU that is still busy, quietly breaking the `MAX_CONCURRENCY` contract the unit tests rely on. In Kubernetes, the same timeouts must nest: `terminationGracePeriodSeconds` (default 30 s) must be at least uvicorn's graceful timeout plus any `preStop` delay.
4. **First principles.** A thread cannot be killed safely from outside. It might hold a lock, or be in the middle of a CUDA call. The only safe cancellation is *cooperative*: set a flag and let the worker check it at a safe point, which here is between decode steps. That is also why cancellation latency is bounded by one ITL, and why a cancelled *prefill* still runs to completion.

### Step 17 — Save your baseline

**What:** Commit your results as the baseline that every later lab is compared against.

**How:**
```bash
ls results/
git add results/
git commit -m "Lab 1.1 baseline: RTX 2080, Qwen2.5-0.5B fp16, HF eager engine"
git push
```

**Why:**
1. **Practical.** Level 2 asks "how much faster is vLLM?", and you can only answer that against numbers measured the same way on the same machine.
2. **Mechanism.** The JSON files store the exact benchmark configuration next to the summary statistics. The CSV files keep every individual request, so you can recompute any percentile later.
3. **System.** Keeping raw results in version control makes performance claims auditable. Anyone can see the configuration that produced a number, which is exactly what a strong interview answer about "how did you measure it?" sounds like.
4. **First principles.** Performance numbers without their conditions (hardware, software versions, request shape, concurrency) are not data. A regression can only be detected relative to a baseline measured under identical conditions.

---

## Gatekeeper: Verification test

Paste these into the chat to pass Lab 1.1:

1. `nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,compute_cap --format=csv`
2. The output of Step 5's PyTorch check, plus `docker image ls lab11-gpu:1.1`.
3. `docker logs lab11 2>&1 | grep -E "gpu:|weights ready|warmup|model_ready"`
4. The full printed output of **prompt-sweep**, **output-sweep**, and both **load** runs.
5. `docker logs lab11 2>&1 | python bench/parse_logs.py`
6. The results of Step 16: the cancellation line, the `in_flight` value, the shutdown log lines, and both `load_s` values.

Pass criteria:
- TTFT rises with prompt length, and the prompt-sweep fit has R² ≥ 0.95.
- The output-sweep fit has R² ≥ 0.99, and its slope is within 15 % of median TPOT.
- `/livez` probes show 0 failures during `load --concurrency 4 --probe`.
- No `request_failed` lines appear in the baseline runs.

---

## Gatekeeper: Chaos / Tuning Challenge

Before you run each part, **write down your prediction** (a sentence and a rough number), then paste both the prediction and the output. The analysis happens together in chat after you report; it is deliberately not in this file.

### Part A (engine) — the prefill memory cliff

**What:** Make the engine compute logits for **every** prompt position (as a naive `model(...)` call does) instead of only the last one, under a tighter memory cap. Then sweep prompt lengths and find where it breaks.

**How:**
```bash
docker rm -f lab11
docker run -d --name lab11 --gpus all -p 8000:8000 -v hf-cache:/models \
  -e PREFILL_LOGITS=all -e GPU_MEMORY_FRACTION=0.25 lab11-gpu:1.1
until curl -sf localhost:8000/readyz > /dev/null; do sleep 2; done
python bench/bench.py prompt-sweep --lengths 512,1024,1536,2048,2560,3072,3584 --repeats 3
docker logs lab11 2>&1 | grep request_failed | tail -n 3
curl -s -w '  HTTP %{http_code}\n' localhost:8000/readyz

# control run: same cap, only the last position's logits
docker rm -f lab11
docker run -d --name lab11 --gpus all -p 8000:8000 -v hf-cache:/models \
  -e PREFILL_LOGITS=last -e GPU_MEMORY_FRACTION=0.25 lab11-gpu:1.1
until curl -sf localhost:8000/readyz > /dev/null; do sleep 2; done
python bench/bench.py prompt-sweep --lengths 512,1024,1536,2048,2560,3072,3584 --repeats 3
```

**Predict before running:**
1. At which prompt length does `PREFILL_LOGITS=all` start failing? Derive it from `vocab_size` in the model's `config.json`, fp16's 2 bytes per value, the weight footprint from Step 8, and the cap: 25 % of what `nvidia-smi` reports as total memory, roughly 2 GB.
2. What happens to `peak_mb` as the prompt grows, in each mode?
3. Does the server stay *ready* after an OOM?

**Why this experiment matters (no spoilers):** Memory during prefill is not just the weights plus the KV cache. Some per-request buffers scale with prompt length × something much larger than the hidden size. Real engines are built around this fact: vLLM's startup profiling run, and the way it sizes the KV cache, exist because of it. You will find the buffer, measure it, and size it from first principles.

### Part B (server) — streaming that doesn't stream

**What:** Run the model **on the event loop** instead of on a worker thread, then load-test with the health probe on.

**How:**
```bash
docker rm -f lab11
docker run -d --name lab11 --gpus all -p 8000:8000 -v hf-cache:/models -e INFERENCE_MODE=blocking lab11-gpu:1.1
until curl -sf localhost:8000/readyz > /dev/null; do sleep 2; done
python bench/bench.py load --concurrency 4 --requests 24 --output-tokens 256 --probe
```
In a **second terminal**, while the load runs:
```bash
for i in $(seq 1 12); do docker inspect --format '{{.State.Health.Status}} failing_streak={{.State.Health.FailingStreak}}' lab11; sleep 3; done
```
Then also capture `docker logs lab11 2>&1 | python bench/parse_logs.py`.

**Predict before running:**
1. What happens to TTFT relative to E2E, and to TPOT?
2. Will throughput (output tok/s) go up, down or stay the same compared with your Step 14 baseline?
3. What will the `/livez` probe and Docker's health status report?
4. What will the *server's own* `queue_ms` say, and is it telling the truth?

**Why this experiment matters (no spoilers):** "async" is a property of every line of code on the event loop, not of the framework you picked. One blocking call breaks several guarantees at once, and some of them break in ways your own metrics will not reveal. You will see which ones, and why an orchestrator would react the way it does.

**Restore the baseline afterwards:**
```bash
docker rm -f lab11
docker run -d --name lab11 --gpus all -p 8000:8000 -v hf-cache:/models lab11-gpu:1.1
```

---

## Checkpoint questions

Answer these in your lab report, using **your** numbers:

1. From `config.json` (`num_hidden_layers`, `num_key_value_heads`, `hidden_size`, `num_attention_heads`), compute this model's KV-cache bytes **per token** in fp16. How many MB of KV does a 3,584-token prompt create? Compare that with the ~1 GB of weights every decode step reads, and use the ratio to explain your ITL column in the prompt sweep.
2. Using your prompt-sweep slope, compute the effective TFLOP/s of prefill (≈ `2 × 494M × tokens/s`). Using your output-sweep TPOT, compute the effective GB/s of decode (≈ weight bytes ÷ TPOT). Place both on the roofline against the RTX 2080's ~448 GB/s. Which phase is far from its ceiling, and what is eating the difference?
3. Account for every millisecond of `client TTFT − server prefill_ms` at 64 tokens and at 3,584 tokens. What is constant, and what grows with the prompt?
4. At concurrency 4, TTFT rose but TPOT did not. Explain it with Little's law, then predict TTFT p50 at concurrency 8 *before* running it. Then run it and compare.

---

## Configuration reference

All settings are environment variables (`docker run -e NAME=value`). Invalid values crash the container at boot with a clear message.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model id |
| `MODEL_REVISION` | `7ae5576…` (pinned SHA) | weights commit; becomes `main` if you change `MODEL_ID` |
| `SERVED_MODEL_NAME` | = `MODEL_ID` | name reported in `/v1/models` and required in `model` |
| `DEVICE` | `cuda` | `cuda` or `cpu` (cpu is for debugging only) |
| `GPU_MEMORY_FRACTION` | `0.60` | hard cap on PyTorch's allocator, as a fraction of VRAM |
| `MAX_MODEL_LEN` | `4096` | prompt tokens + `max_tokens` must fit |
| `MAX_TOKENS_LIMIT` | `1024` | largest `max_tokens` a request may ask for |
| `MAX_CONCURRENCY` | `1` | generations running at once (1–8) |
| `MAX_QUEUE_DEPTH` | `32` | requests allowed to wait; beyond that → 503 |
| `QUEUE_TIMEOUT_S` | `60` | maximum wait for a slot → 503 |
| `PREFILL_LOGITS` | `last` | `last` (production) or `all` (Chaos A) |
| `INFERENCE_MODE` | `threadpool` | `threadpool` (production) or `blocking` (Chaos B) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

To try Llama-3.2-1B (a gated model: accept its license on Hugging Face and create a read token first):
```bash
docker run -d --name lab11-llama --gpus all -p 8001:8000 -v hf-cache:/models \
  -e MODEL_ID=meta-llama/Llama-3.2-1B-Instruct -e HF_TOKEN="$HF_TOKEN" lab11-gpu:1.1
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing (Linux) | Install `nvidia-container-toolkit`, run `sudo nvidia-ctk runtime configure --runtime=docker`, then `sudo systemctl restart docker` |
| Same error on Windows | Docker Desktop not using the WSL2 backend, or an old Windows NVIDIA driver | Enable the WSL2 backend; update the Windows driver; do **not** install a Linux driver inside WSL |
| `/livez` → 503 with `CUDA is not available` | Container started without `--gpus all` | Re-run with `--gpus all` |
| `CUDA driver version is insufficient` | Host driver older than CUDA 12.6 | Update the NVIDIA driver until `nvidia-smi` shows CUDA ≥ 12.6 |
| `PermissionError` under `/models` | Volume created by a different image or user | `docker volume rm hf-cache && docker volume create hf-cache` |
| `/readyz` stuck at `loading` for minutes | First weight download, or a slow network | `docker logs -f lab11`; this happens once per volume |
| `request_failed … out_of_memory` in baseline runs | Something else is using VRAM, or the cap is too low | Check `nvidia-smi`; raise `GPU_MEMORY_FRACTION` |
| `bind: address already in use` | Port 8000 is taken | Use `-p 8001:8000` and `--url http://localhost:8001` |
| Benchmark: `these shapes would exceed max_model_len` | Requested lengths too long for `MAX_MODEL_LEN` | Shorten `--lengths` or raise `MAX_MODEL_LEN` |

---

## How this lab was verified

The server, scheduler, detokenizer, benchmark and log parser were exercised end-to-end in a CPU-only sandbox. That run used the real `server/main.py`, `scheduler.py`, `protocol.py`, `detokenizer.py`, `bench/bench.py` and `bench/parse_logs.py` against a simulated engine with the same interface as `engine.py`. It covered:
- readiness vs. liveness during load, and a failed load;
- every endpoint, including the 400/404/422 error paths;
- streaming and non-streaming responses;
- queue shedding (503);
- slot accounting through cancellation;
- client-disconnect cancellation;
- graceful drain on `SIGTERM`;
- both chaos modes.

The GPU code path in `engine.py` (PyTorch + transformers on CUDA) and the Docker build are validated by **your** first run on the RTX 2080 in Steps 4–8.
