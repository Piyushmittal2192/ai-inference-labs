"""Lab 1.1 server: an async, OpenAI-compatible completions API over one GPU.

  GET  /livez            the process is alive and its event loop is responsive
  GET  /readyz           weights are on the GPU and warmed up: send traffic here
  GET  /stats            scheduler counters and GPU memory
  GET  /v1/models        OpenAI-style model list (includes max_model_len)
  POST /v1/completions   OpenAI-style text completion, streaming (SSE) or not

The event loop only does cheap work: parse, validate, admit, move bytes. Every
GPU operation runs on a worker thread owned by server/scheduler.py. Tokens
cross from that thread to the event loop through loop.call_soon_threadsafe().
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from server.config import get_settings
from server.engine import InferenceEngine
from server.protocol import EngineError, GenerationStats, SamplingParams
from server.scheduler import InferenceScheduler, SchedulerRejection
from server.schemas import CompletionRequest

SETTINGS = get_settings()

logging.basicConfig(
    level=SETTINGS.log_level,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("lab11.api")


# ---------------------------------------------------------------- lifecycle


class Phase(str, Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ModelStatus:
    """Model lifecycle state. Only ever mutated on the event-loop thread."""

    def __init__(self) -> None:
        self.phase = Phase.LOADING
        self.error: str | None = None
        self.load_seconds: float | None = None

    def mark_ready(self, load_seconds: float) -> None:
        self.phase = Phase.READY
        self.load_seconds = load_seconds
        log.info("event=model_ready load_s=%.2f", load_seconds)

    def mark_failed(self, error: str) -> None:
        self.phase = Phase.FAILED
        self.error = error
        log.error("event=model_failed error=%s", json.dumps(error))


def start_model_loader(engine: InferenceEngine, status: ModelStatus, loop: asyncio.AbstractEventLoop) -> None:
    """Load weights on a daemon thread so /livez answers while loading.

    Daemon matters: if the container is stopped mid-download, a non-daemon
    thread would hold the interpreter open until the download finished.
    """

    def work() -> None:
        try:
            load_seconds = engine.load()
        except Exception as exc:
            log.exception("model load failed")
            callback, argument = status.mark_failed, f"{type(exc).__name__}: {exc}"
        else:
            callback, argument = status.mark_ready, load_seconds
        try:
            loop.call_soon_threadsafe(callback, argument)
        except RuntimeError:
            pass  # the event loop closed while we were loading

    threading.Thread(target=work, name="model-loader", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info(
        "config: model=%s revision=%s device=%s gpu_memory_fraction=%.2f max_model_len=%d "
        "max_tokens_limit=%d max_concurrency=%d max_queue_depth=%d queue_timeout_s=%.1f "
        "prefill_logits=%s inference_mode=%s",
        SETTINGS.model_id,
        SETTINGS.model_revision[:12],
        SETTINGS.device,
        SETTINGS.gpu_memory_fraction,
        SETTINGS.max_model_len,
        SETTINGS.max_tokens_limit,
        SETTINGS.max_concurrency,
        SETTINGS.max_queue_depth,
        SETTINGS.queue_timeout_s,
        SETTINGS.prefill_logits,
        SETTINGS.inference_mode,
    )
    if SETTINGS.blocking:
        log.warning("INFERENCE_MODE=blocking: GPU work runs ON the event loop (chaos mode)")
    engine = InferenceEngine(
        model_id=SETTINGS.model_id,
        revision=SETTINGS.model_revision,
        device=SETTINGS.device,
        gpu_memory_fraction=SETTINGS.gpu_memory_fraction,
        max_model_len=SETTINGS.max_model_len,
        prefill_logits=SETTINGS.prefill_logits,
    )
    scheduler = InferenceScheduler(
        max_concurrency=SETTINGS.max_concurrency,
        max_queue_depth=SETTINGS.max_queue_depth,
        queue_timeout_s=SETTINGS.queue_timeout_s,
        blocking=SETTINGS.blocking,
    )
    status = ModelStatus()
    app.state.engine = engine
    app.state.scheduler = scheduler
    app.state.model_status = status
    app.state.started_monotonic = time.monotonic()
    app.state.started_unix = int(time.time())
    start_model_loader(engine, status, asyncio.get_running_loop())
    try:
        yield
    finally:
        log.info("event=shutdown scheduler=%s", json.dumps(scheduler.snapshot()))
        scheduler.shutdown()


app = FastAPI(title="Lab 1.1 GPU Completion Server", version="1.1.0", lifespan=lifespan)


# ---------------------------------------------------------------- health


@app.get("/livez")
async def livez(request: Request):
    """Liveness: should this process be kept? Never touches the GPU.

    Slow loading is NOT a liveness failure. A failed load is: nothing in this
    process will fix it, so report 503 and let the orchestrator restart us.
    """
    status: ModelStatus = request.app.state.model_status
    if status.phase is Phase.FAILED:
        return JSONResponse(status_code=503, content={"status": "failed", "error": status.error})
    uptime = time.monotonic() - request.app.state.started_monotonic
    return {"status": "alive", "phase": status.phase.value, "uptime_s": round(uptime, 1)}


@app.get("/readyz")
async def readyz(request: Request):
    """Readiness: should traffic be routed here right now?"""
    status: ModelStatus = request.app.state.model_status
    body: dict[str, Any] = {"status": status.phase.value, "model": SETTINGS.served_model_name}
    if status.phase is Phase.READY:
        body["load_seconds"] = round(status.load_seconds or 0.0, 2)
        return body
    if status.error:
        body["error"] = status.error
    return JSONResponse(status_code=503, content=body)


@app.get("/stats")
async def stats(request: Request):
    status: ModelStatus = request.app.state.model_status
    engine: InferenceEngine = request.app.state.engine
    scheduler: InferenceScheduler = request.app.state.scheduler
    return {
        "model": {
            "id": SETTINGS.model_id,
            "revision": SETTINGS.model_revision,
            "phase": status.phase.value,
            "load_seconds": status.load_seconds,
        },
        "scheduler": scheduler.snapshot(),
        "memory": engine.memory_snapshot(),
        "limits": {
            "max_model_len": SETTINGS.max_model_len,
            "max_tokens_limit": SETTINGS.max_tokens_limit,
            "gpu_memory_fraction": SETTINGS.gpu_memory_fraction,
            "prefill_logits": SETTINGS.prefill_logits,
        },
    }


@app.get("/v1/models")
async def list_models(request: Request):
    return {
        "object": "list",
        "data": [
            {
                "id": SETTINGS.served_model_name,
                "object": "model",
                "created": request.app.state.started_unix,
                "owned_by": "lab11",
                "max_model_len": SETTINGS.max_model_len,
            }
        ],
    }


# ---------------------------------------------------------------- completions


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    created: int
    prompt_tokens: int
    tokenize_ms: float
    queue_ms: float
    stream: bool


def timings_payload(ctx: RequestContext, stats: GenerationStats) -> dict[str, Any]:
    """Server-side phase timings. Not part of the OpenAI spec; clients that
    don't know the field ignore it."""

    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, 2)

    return {
        "queue_ms": rounded(ctx.queue_ms),
        "tokenize_ms": rounded(ctx.tokenize_ms),
        "prefill_ms": rounded(stats.prefill_ms),
        "ttft_ms": rounded(ctx.tokenize_ms + ctx.queue_ms + stats.prefill_ms),
        "decode_ms": rounded(stats.decode_ms),
        "decode_steps": len(stats.decode_step_ms),
        "itl_mean_ms": rounded(stats.itl_mean_ms),
        "itl_p50_ms": rounded(stats.itl_percentile_ms(50)),
        "itl_p99_ms": rounded(stats.itl_percentile_ms(99)),
        "prefill_tokens_per_s": rounded(stats.prefill_tokens_per_s),
        "decode_tokens_per_s": rounded(stats.decode_tokens_per_s),
        "peak_memory_mb": rounded(stats.peak_memory_mb),
    }


def usage_payload(stats: GenerationStats) -> dict[str, int]:
    return {
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.prompt_tokens + stats.completion_tokens,
    }


def log_request_done(ctx: RequestContext, stats: GenerationStats) -> None:
    """One machine-parseable line per finished request (bench/parse_logs.py reads it)."""
    fields = {
        "event": "request_done",
        "request_id": ctx.request_id,
        "stream": str(ctx.stream).lower(),
        "finish_reason": stats.finish_reason,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
    }
    for key, value in timings_payload(ctx, stats).items():
        fields[key] = "na" if value is None else value
    log.info(" ".join(f"{key}={value}" for key, value in fields.items()))


def log_request_failed(ctx: RequestContext, exc: BaseException) -> None:
    error_type = exc.error_type if isinstance(exc, EngineError) else "internal_error"
    log.error(
        "event=request_failed request_id=%s error_type=%s prompt_tokens=%d message=%s",
        ctx.request_id,
        error_type,
        ctx.prompt_tokens,
        json.dumps(str(exc)),
    )


def error_payload(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, EngineError):
        return {"message": str(exc), "type": exc.error_type}
    return {"message": "internal server error", "type": "internal_error"}


def sse(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


def chunk(ctx: RequestContext, text: str, finish_reason: str | None) -> dict[str, Any]:
    return {
        "id": ctx.request_id,
        "object": "text_completion",
        "created": ctx.created,
        "model": SETTINGS.served_model_name,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
    }


async def stream_events(
    ctx: RequestContext,
    events: asyncio.Queue[tuple[str, Any]],
    cancel: threading.Event,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    finished = False
    try:
        while True:
            kind, payload = await events.get()
            if kind == "token":
                yield sse(chunk(ctx, payload, None))
            elif kind == "done":
                finished = True
                log_request_done(ctx, payload)
                final = chunk(ctx, "", payload.finish_reason)
                final["server_timings"] = timings_payload(ctx, payload)
                yield sse(final)
                if include_usage:
                    usage_chunk = chunk(ctx, "", None)
                    usage_chunk["choices"] = []
                    usage_chunk["usage"] = usage_payload(payload)
                    yield sse(usage_chunk)
                yield sse("[DONE]")
                return
            else:
                finished = True
                log_request_failed(ctx, payload)
                yield sse({"error": error_payload(payload)})
                yield sse("[DONE]")
                return
    finally:
        if not finished:
            # The client went away mid-stream. Stop the engine at its next
            # decode step instead of burning GPU time on tokens nobody will read.
            cancel.set()
            log.info("event=request_cancelled request_id=%s", ctx.request_id)


async def collect_events(
    ctx: RequestContext,
    events: asyncio.Queue[tuple[str, Any]],
    cancel: threading.Event,
) -> dict[str, Any]:
    parts: list[str] = []
    try:
        while True:
            kind, payload = await events.get()
            if kind == "token":
                parts.append(payload)
            elif kind == "done":
                log_request_done(ctx, payload)
                body = chunk(ctx, "".join(parts), payload.finish_reason)
                body["usage"] = usage_payload(payload)
                body["server_timings"] = timings_payload(ctx, payload)
                return body
            else:
                log_request_failed(ctx, payload)
                raise HTTPException(status_code=500, detail=error_payload(payload))
    except asyncio.CancelledError:
        cancel.set()
        raise


@app.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    status: ModelStatus = request.app.state.model_status
    if status.phase is not Phase.READY:
        raise HTTPException(
            status_code=503,
            detail=f"model not ready (phase={status.phase.value})",
            headers={"Retry-After": "5"},
        )
    if body.model is not None and body.model != SETTINGS.served_model_name:
        raise HTTPException(
            status_code=404,
            detail=f"model {body.model!r} not found; this server serves {SETTINGS.served_model_name!r}",
        )
    engine: InferenceEngine = request.app.state.engine
    scheduler: InferenceScheduler = request.app.state.scheduler

    tokenize_started = time.perf_counter()
    prompt_ids = engine.tokenize(body.prompt)
    tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0
    if len(prompt_ids) + body.max_tokens > SETTINGS.max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"prompt has {len(prompt_ids)} tokens and max_tokens is {body.max_tokens}; "
                f"together they exceed MAX_MODEL_LEN={SETTINGS.max_model_len}"
            ),
        )

    params = SamplingParams(
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        ignore_eos=body.ignore_eos,
        seed=body.seed,
    )
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    cancel = threading.Event()

    def push(kind: str, payload: Any) -> None:
        # Runs on the worker thread. asyncio objects are not thread-safe, so
        # the put is scheduled onto the event loop instead of done here.
        try:
            loop.call_soon_threadsafe(events.put_nowait, (kind, payload))
        except RuntimeError:
            pass  # event loop closed during shutdown

    def work() -> GenerationStats:
        try:
            result = engine.generate(prompt_ids, params, lambda text: push("token", text), cancel)
        except BaseException as exc:
            push("error", exc)
            raise
        push("done", result)
        return result

    request_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    try:
        _future, queue_s = await scheduler.submit(work)
    except SchedulerRejection as exc:
        log.warning("event=request_rejected request_id=%s reason=%s", request_id, json.dumps(str(exc)))
        raise HTTPException(
            status_code=503, detail=str(exc), headers={"Retry-After": str(exc.retry_after_s)}
        ) from None

    ctx = RequestContext(
        request_id=request_id,
        created=int(time.time()),
        prompt_tokens=len(prompt_ids),
        tokenize_ms=tokenize_ms,
        queue_ms=queue_s * 1000.0,
        stream=body.stream,
    )
    if body.stream:
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        return StreamingResponse(
            stream_events(ctx, events, cancel, include_usage),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await collect_events(ctx, events, cancel)
