# AI Inference Engineering Labs

A hands-on, local-first curriculum in **LLM serving mechanics, benchmarking and AI infrastructure**, built and measured on one consumer GPU.

- **Hardware:** 1× NVIDIA RTX 2080 (8 GB GDDR6, Turing `sm_75`)
- **Models:** small open models that fit the budget: `Qwen2.5-0.5B/1.5B-Instruct`, `Llama-3.2-1B-Instruct`
- **Stack:** Docker, FastAPI, PyTorch, Hugging Face transformers, then vLLM, Kind (Kubernetes in Docker), NVIDIA time-slicing, Prometheus
- **Method:** every lab ships production-style code, a benchmark you run yourself, and a chaos/tuning challenge that breaks something on purpose. Results are committed next to the code that produced them.

Each lab's README is a runbook: every step explains **what** to do, **how** to do it (copy-paste commands), and **why**, four layers deep, from the practical reason down to the hardware or first principle behind it.

## Curriculum

| Level | Lab | Topic | Status |
|---|---|---|---|
| **1 · Foundations** | [1.1](labs/lab-1.1-gpu-serving/README.md) | Async GPU serving in Docker · prefill vs. decode · TTFT / ITL / TPOT benchmarking · log-based metrics | **Built, first GPU run pending** |
| | 1.2 | Queueing under load: open-loop arrivals, head-of-line blocking, backpressure, SLO-based capacity | Planned |
| **2 · Engine mechanics (vLLM on 8 GB)** | 2.1 | vLLM under a strict memory budget: `--gpu-memory-utilization`, `--max-model-len`, the KV block budget; Lab 1.1 benchmarks re-run as an A/B | Planned |
| | 2.2 | KV cache and PagedAttention: block math, preemption and KV thrashing under concurrency, prefix caching | Planned |
| | 2.3 | Scheduling: continuous batching, chunked prefill, `max-num-seqs` / `max-num-batched-tokens` trade-offs | Planned |
| | 2.4 | Quantization on Turing: int4 (AWQ/GPTQ) vs. fp16 — memory, TTFT, ITL, quality | Planned |
| | 2.5 | Speculative decoding: draft/target pairs and n-gram drafting; acceptance rate vs. ITL | Planned |
| **3 · Kubernetes for GPUs (Kind)** | 3.1 | Kind with GPU access: device plugin, scheduling, liveness/readiness/startup probes | Planned |
| | 3.2 | NVIDIA time-slicing: one GPU as N, no memory isolation, multi-model concurrency profiles | Planned |
| | 3.3 | Model-cache volumes, headless services, rolling updates without dropped streams | Planned |
| **4 · Scale and observability** | 4.1 | Prometheus metrics for TTFT, ITL and KV-cache usage; Grafana dashboards | Planned |
| | 4.2 | High-concurrency load profiles and autoscaling on queue depth | Planned |
| | 4.3 | Capstone: SLO-driven capacity plan and design doc | Planned |

The plan is deliberately gated: each lab unlocks only after the previous one's verification test and chaos challenge have been run and analyzed.

## Progress log

### 2026-10-24 — Lab 1.1 built

**What was built**
- `labs/lab-1.1-gpu-serving/server/` — an OpenAI-compatible `/v1/completions` server (JSON + SSE streaming) that:
  - runs prefill and decode by hand, so each phase is timed on the GPU;
  - keeps every GPU operation off the asyncio event loop;
  - bounds its queue and sheds load with `503 + Retry-After`;
  - cancels generation when a client disconnects;
  - drains in-flight streams on `SIGTERM`;
  - recovers from CUDA out-of-memory without restarting;
  - separates liveness from readiness.
- `labs/lab-1.1-gpu-serving/Dockerfile` — multi-stage image with CUDA 12.6 PyTorch wheels for Turing, non-root, `tini` as PID 1, weights in a volume (not in the image), built-in health check.
- `labs/lab-1.1-gpu-serving/bench/bench.py` — async streaming benchmark:
  - three modes: prompt sweep (isolates prefill), output sweep (isolates decode), concurrent load (queueing);
  - linear fits for per-token costs;
  - an optional health-probe sidecar;
  - CSV + JSON results.
- `labs/lab-1.1-gpu-serving/bench/parse_logs.py` — extracts engine-side TTFT, prefill and ITL from server logs.

**How it was verified so far**
- Unit-tested: config validation, incremental detokenization of multi-byte UTF-8, timing statistics, and the scheduler. The scheduler tests covered bounded concurrency, queue shedding, queue timeouts, exception accounting, blocking-mode loop stalls, and the guarantee that a GPU slot is held until the worker thread actually finishes, even when its request is cancelled.
- End-to-end HTTP tests ran against the real server code with a simulated engine in a CPU-only sandbox. They covered:
  - every endpoint and error path (400 / 404 / 422 / 503);
  - streaming and non-streaming;
  - benchmark and log-parser output;
  - client-disconnect cancellation;
  - graceful shutdown;
  - both chaos modes.
- **Still to validate on the RTX 2080:** the Docker build, and the CUDA path in `engine.py`. That is the first run of Lab 1.1, Steps 4–8.

**Next**
- Run the Lab 1.1 runbook on the RTX 2080 and commit the baseline in `labs/lab-1.1-gpu-serving/results/`.
- Run the Chaos / Tuning Challenge (prefill memory cliff; blocking event loop) and record the analysis here.

## Repository layout

```
labs/
  lab-1.1-gpu-serving/   runbook (README.md), server/, bench/, Dockerfile, results/
```
