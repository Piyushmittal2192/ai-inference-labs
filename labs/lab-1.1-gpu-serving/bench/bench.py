#!/usr/bin/env python3
"""Lab 1.1 streaming benchmark: TTFT, ITL, TPOT and throughput.

Speaks the OpenAI /v1/completions streaming protocol, so the same script
benchmarks this lab's server today and vLLM in Level 2 without changes.

Subcommands
  prompt-sweep  vary prompt length, fixed output length  -> isolates PREFILL
  output-sweep  vary output length, fixed prompt length  -> isolates DECODE
  load          one request shape, N concurrent streams  -> queueing, throughput

Definitions (client-side wall clock)
  TTFT  request sent -> first non-empty text chunk received
  ITL   gap between two consecutive text chunks
  TPOT  (E2E - TTFT) / (output_tokens - 1): mean time per token after the first
  E2E   request sent -> stream finished

Prompts are random common words with a unique prefix per request, so a server
with prefix caching (vLLM, Level 2) cannot reuse work between requests and
quietly flatter the numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

WORDS = (
    "time person year way day thing man world life hand part child eye woman place work week case point "
    "government company number group problem fact water room mother area money story month lot right study "
    "book job word business issue side kind head house service friend father power hour game line end member "
    "law car city community name president team minute idea kid body information back parent face others "
    "level office door health art war history party result change morning reason research girl guy moment "
    "air teacher force education river garden market window table music paper light energy system program "
    "question during country school state family student night home picture model price value"
).split()


# ---------------------------------------------------------------- data


@dataclass
class RequestRecord:
    label: str
    target_prompt_tokens: int
    target_output_tokens: int
    ok: bool
    error: str | None = None
    http_status: int | None = None
    started_s: float = 0.0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    chunks: int = 0
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    server: dict[str, Any] = field(default_factory=dict)


def pct(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile; None when there is no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1]


def fmt(value: float | None, width: int = 9, digits: int = 1) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float] | None:
    """Least-squares fit y = a + b*x. Returns (a, b, r_squared)."""
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return intercept, slope, r2


# ---------------------------------------------------------------- prompts


class PromptFactory:
    """Builds prompts of a requested token length for whatever tokenizer the
    server runs, without needing that tokenizer on the client.

    It calibrates once: sends two prompts of known word counts, reads the
    server-reported prompt_tokens, and fits tokens = a + b * words.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.intercept = 0.0
        self.tokens_per_word = 1.0

    def _text(self, n_words: int) -> str:
        nonce = uuid.UUID(int=self._rng.getrandbits(128)).hex[:12]
        body = " ".join(self._rng.choice(WORDS) for _ in range(max(1, n_words)))
        return f"Request {nonce}: {body}"

    def build(self, target_tokens: int) -> str:
        n_words = round((target_tokens - self.intercept) / self.tokens_per_word)
        return self._text(max(1, n_words))

    async def calibrate(self, client: httpx.AsyncClient, model: str) -> None:
        samples: list[tuple[int, int]] = []
        for n_words in (64, 512):
            payload = {"model": model, "prompt": self._text(n_words), "max_tokens": 1, "temperature": 0.0}
            response = await client.post("/v1/completions", json=payload)
            response.raise_for_status()
            samples.append((n_words, int(response.json()["usage"]["prompt_tokens"])))
        fit = ols([float(w) for w, _ in samples], [float(t) for _, t in samples])
        if fit is None or fit[1] <= 0:
            raise RuntimeError(f"tokenizer calibration failed: {samples}")
        self.intercept, self.tokens_per_word = fit[0], fit[1]


# ---------------------------------------------------------------- one request


async def stream_completion(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
    label: str,
    target_prompt_tokens: int,
    t0: float,
) -> RequestRecord:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    record = RequestRecord(
        label=label,
        target_prompt_tokens=target_prompt_tokens,
        target_output_tokens=max_tokens,
        ok=False,
    )
    sent = time.perf_counter()
    record.started_s = sent - t0
    arrivals: list[float] = []
    usage: dict[str, Any] | None = None
    try:
        async with client.stream("POST", "/v1/completions", json=payload) as response:
            record.http_status = response.status_code
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                record.error = f"HTTP {response.status_code}: {body[:300]}"
                return record
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    record.error = f"stream error: {event['error']}"
                    return record
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("server_timings"):
                    record.server = event["server_timings"]
                for choice in event.get("choices") or []:
                    if choice.get("text"):
                        arrivals.append(time.perf_counter())
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    finished = time.perf_counter()
    if not arrivals:
        record.error = "stream ended without any text"
        return record
    record.ok = True
    record.chunks = len(arrivals)
    record.ttft_ms = (arrivals[0] - sent) * 1000.0
    record.e2e_ms = (finished - sent) * 1000.0
    record.itl_ms = [(b - a) * 1000.0 for a, b in zip(arrivals, arrivals[1:])]
    if usage:
        record.prompt_tokens = int(usage["prompt_tokens"])
        record.output_tokens = int(usage["completion_tokens"])
    else:
        record.output_tokens = len(arrivals)
    if record.output_tokens and record.output_tokens > 1:
        record.tpot_ms = (record.e2e_ms - record.ttft_ms) / (record.output_tokens - 1)
    return record


# ---------------------------------------------------------------- health probe


async def probe_loop(url: str, interval_s: float, timeout_s: float, stop: asyncio.Event, out: list[tuple[bool, float]]) -> None:
    """Hit /livez on a fixed cadence, like a Docker or Kubernetes health check."""
    async with httpx.AsyncClient(base_url=url, timeout=timeout_s, trust_env=False) as client:
        while not stop.is_set():
            started = time.perf_counter()
            try:
                response = await client.get("/livez")
                ok = response.status_code == 200
            except httpx.HTTPError:
                ok = False
            elapsed = time.perf_counter() - started
            out.append((ok, elapsed * 1000.0))
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(0.0, interval_s - elapsed))
            except TimeoutError:
                pass


# ---------------------------------------------------------------- reporting


def server_values(records: Sequence[RequestRecord], key: str) -> list[float]:
    return [float(r.server[key]) for r in records if r.ok and r.server.get(key) is not None]


def summarize(records: Sequence[RequestRecord]) -> dict[str, Any]:
    ok = [r for r in records if r.ok]
    itl = [gap for r in ok for gap in r.itl_ms]
    summary: dict[str, Any] = {
        "requests": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "prompt_tokens_p50": pct([float(r.prompt_tokens) for r in ok if r.prompt_tokens], 50),
        "output_tokens_p50": pct([float(r.output_tokens) for r in ok if r.output_tokens], 50),
    }
    for name, values in (
        ("ttft_ms", [r.ttft_ms for r in ok if r.ttft_ms is not None]),
        ("tpot_ms", [r.tpot_ms for r in ok if r.tpot_ms is not None]),
        ("e2e_ms", [r.e2e_ms for r in ok if r.e2e_ms is not None]),
        ("itl_ms", itl),
    ):
        for p in (50, 90, 99):
            summary[f"{name}_p{p}"] = pct(values, p)
    for key in ("queue_ms", "prefill_ms", "ttft_ms", "itl_p50_ms", "prefill_tokens_per_s", "decode_tokens_per_s", "peak_memory_mb"):
        values = server_values(ok, key)
        summary[f"server_{key}_p50"] = pct(values, 50)
        summary[f"server_{key}_max"] = max(values) if values else None
    errors: dict[str, int] = {}
    for r in records:
        if r.error:
            errors[r.error[:160]] = errors.get(r.error[:160], 0) + 1
    summary["errors"] = errors
    return summary


def write_results(out_dir: Path, mode: str, config: dict[str, Any], records: list[RequestRecord], summaries: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"{mode}-{stamp}"
    columns = [
        "label", "target_prompt_tokens", "target_output_tokens", "ok", "http_status", "error", "started_s",
        "prompt_tokens", "output_tokens", "chunks", "ttft_ms", "tpot_ms", "e2e_ms", "itl_p50_ms", "itl_p99_ms",
        "server_queue_ms", "server_prefill_ms", "server_ttft_ms", "server_itl_p50_ms", "server_peak_memory_mb",
    ]
    with base.with_suffix(".csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "label": r.label,
                "target_prompt_tokens": r.target_prompt_tokens,
                "target_output_tokens": r.target_output_tokens,
                "ok": r.ok,
                "http_status": r.http_status,
                "error": r.error,
                "started_s": round(r.started_s, 4),
                "prompt_tokens": r.prompt_tokens,
                "output_tokens": r.output_tokens,
                "chunks": r.chunks,
                "ttft_ms": r.ttft_ms,
                "tpot_ms": r.tpot_ms,
                "e2e_ms": r.e2e_ms,
                "itl_p50_ms": pct(r.itl_ms, 50),
                "itl_p99_ms": pct(r.itl_ms, 99),
                "server_queue_ms": r.server.get("queue_ms"),
                "server_prefill_ms": r.server.get("prefill_ms"),
                "server_ttft_ms": r.server.get("ttft_ms"),
                "server_itl_p50_ms": r.server.get("itl_p50_ms"),
                "server_peak_memory_mb": r.server.get("peak_memory_mb"),
            })
    base.with_suffix(".json").write_text(json.dumps({"config": config, "summaries": summaries}, indent=2, default=str))
    return base


def print_errors(summary: dict[str, Any]) -> None:
    for message, count in summary["errors"].items():
        print(f"    x{count}  {message}")


# ---------------------------------------------------------------- modes


async def run_sweep(
    client: httpx.AsyncClient,
    model: str,
    factory: PromptFactory,
    shapes: list[tuple[int, int]],
    repeats: int,
    label_key: str,
) -> list[RequestRecord]:
    t0 = time.perf_counter()
    records: list[RequestRecord] = []
    for prompt_tokens, output_tokens in shapes:
        label = f"{label_key}={prompt_tokens if label_key == 'prompt' else output_tokens}"
        for i in range(repeats):
            record = await stream_completion(
                client, model, factory.build(prompt_tokens), output_tokens, label, prompt_tokens, t0
            )
            records.append(record)
            status = "ok" if record.ok else "FAIL"
            print(f"  {label:<14} run {i + 1}/{repeats}  {status}  ttft={fmt(record.ttft_ms, 7)} ms", file=sys.stderr)
    return records


def report_prompt_sweep(records: list[RequestRecord], lengths: list[int]) -> dict[str, Any]:
    print("\n=== PROMPT SWEEP: prefill cost (concurrency 1) ===")
    print(f"{'target':>7} {'actual':>7} {'ok':>4} {'ttft_p50':>9} {'ttft_p90':>9} {'srv_prefill':>11} {'prefill_tok/s':>13} "
          f"{'itl_p50':>8} {'itl_p99':>8} {'peak_mb':>8}")
    summaries: dict[str, Any] = {}
    xs: list[float] = []
    ys: list[float] = []
    for length in lengths:
        group = [r for r in records if r.target_prompt_tokens == length]
        s = summarize(group)
        summaries[str(length)] = s
        actual = s["prompt_tokens_p50"]
        print(f"{length:>7} {fmt(actual, 7, 0)} {s['ok']:>4} {fmt(s['ttft_ms_p50'])} {fmt(s['ttft_ms_p90'])} "
              f"{fmt(s['server_prefill_ms_p50'], 11)} {fmt(s['server_prefill_tokens_per_s_p50'], 13, 0)} "
              f"{fmt(s['itl_ms_p50'], 8)} {fmt(s['itl_ms_p99'], 8)} {fmt(s['server_peak_memory_mb_max'], 8, 0)}")
        for r in group:
            if r.ok and r.prompt_tokens and r.ttft_ms is not None:
                xs.append(float(r.prompt_tokens))
                ys.append(r.ttft_ms)
    for length in lengths:
        if summaries[str(length)]["errors"]:
            print(f"  errors at prompt={length}:")
            print_errors(summaries[str(length)])
    fit = ols(xs, ys)
    if fit:
        a, b, r2 = fit
        print(f"\nfit: TTFT_ms = {a:.2f} + {b:.4f} * prompt_tokens   (R^2 = {r2:.3f})")
        if b > 0:
            print(f"     marginal prefill throughput ~ {1000.0 / b:,.0f} prompt tokens/s")
        summaries["fit"] = {"intercept_ms": a, "ms_per_prompt_token": b, "r2": r2}
    return summaries


def report_output_sweep(records: list[RequestRecord], lengths: list[int]) -> dict[str, Any]:
    print("\n=== OUTPUT SWEEP: decode cost (concurrency 1) ===")
    print(f"{'out_tok':>8} {'ok':>4} {'ttft_p50':>9} {'tpot_p50':>9} {'itl_p50':>8} {'itl_p99':>8} {'e2e_p50':>10} "
          f"{'srv_decode_tok/s':>16}")
    summaries: dict[str, Any] = {}
    xs: list[float] = []
    ys: list[float] = []
    for length in lengths:
        group = [r for r in records if r.target_output_tokens == length]
        s = summarize(group)
        summaries[str(length)] = s
        print(f"{length:>8} {s['ok']:>4} {fmt(s['ttft_ms_p50'])} {fmt(s['tpot_ms_p50'])} {fmt(s['itl_ms_p50'], 8)} "
              f"{fmt(s['itl_ms_p99'], 8)} {fmt(s['e2e_ms_p50'], 10)} {fmt(s['server_decode_tokens_per_s_p50'], 16)}")
        for r in group:
            if r.ok and r.output_tokens and r.e2e_ms is not None:
                xs.append(float(r.output_tokens))
                ys.append(r.e2e_ms)
    for length in lengths:
        if summaries[str(length)]["errors"]:
            print(f"  errors at output={length}:")
            print_errors(summaries[str(length)])
    fit = ols(xs, ys)
    if fit:
        a, b, r2 = fit
        print(f"\nfit: E2E_ms = {a:.1f} + {b:.3f} * output_tokens   (R^2 = {r2:.4f})")
        print("     intercept ~ fixed cost per request (TTFT-like); slope ~ cost per decoded token (TPOT)")
        summaries["fit"] = {"intercept_ms": a, "ms_per_output_token": b, "r2": r2}
    return summaries


def report_load(
    records: list[RequestRecord], wall_s: float, concurrency: int, probes: list[tuple[bool, float]] | None
) -> dict[str, Any]:
    s = summarize(records)
    ok = [r for r in records if r.ok]
    out_tokens = sum(r.output_tokens or 0 for r in ok)
    s["wall_s"] = wall_s
    s["requests_per_s"] = len(ok) / wall_s if wall_s > 0 else None
    s["output_tokens_per_s"] = out_tokens / wall_s if wall_s > 0 else None
    print(f"\n=== LOAD: {s['requests']} requests, concurrency {concurrency}, wall {wall_s:.1f}s ===")
    print(f"ok={s['ok']} failed={s['failed']}   throughput: {fmt(s['requests_per_s'], 0, 2)} req/s, "
          f"{fmt(s['output_tokens_per_s'], 0, 1)} output tok/s")
    print(f"{'':12}{'p50':>9}{'p90':>9}{'p99':>9}")
    for name in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
        print(f"{name:<12}{fmt(s[name + '_p50'])}{fmt(s[name + '_p90'])}{fmt(s[name + '_p99'])}")
    queue = server_values(ok, "queue_ms")
    if queue:
        print(f"{'srv_queue':<12}{fmt(pct(queue, 50))}{fmt(pct(queue, 90))}{fmt(pct(queue, 99))}")
    print_errors(s)
    if probes is not None:
        failed = sum(1 for good, _ in probes if not good)
        latencies = [ms for _, ms in probes]
        longest = run = 0
        for good, _ in probes:
            run = 0 if good else run + 1
            longest = max(longest, run)
        s["probe"] = {
            "count": len(probes),
            "failed": failed,
            "latency_ms_p50": pct(latencies, 50),
            "latency_ms_p99": pct(latencies, 99),
            "latency_ms_max": max(latencies) if latencies else None,
            "longest_failure_streak": longest,
        }
        print(f"\n/livez probe: {len(probes)} probes, {failed} failed, longest failure streak {longest}")
        print(f"  latency ms p50={fmt(pct(latencies, 50), 0)}  p99={fmt(pct(latencies, 99), 0)}  "
              f"max={fmt(max(latencies) if latencies else None, 0)}")
    return s


async def run_load(
    client: httpx.AsyncClient, model: str, factory: PromptFactory, args: argparse.Namespace
) -> tuple[list[RequestRecord], float, list[tuple[bool, float]] | None]:
    gate = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()
    done = 0

    async def one(_index: int) -> RequestRecord:
        nonlocal done
        async with gate:
            record = await stream_completion(
                client, model, factory.build(args.prompt_tokens), args.output_tokens, "load", args.prompt_tokens, t0
            )
        done += 1
        print(f"\r  {done}/{args.requests} done", end="", file=sys.stderr, flush=True)
        return record

    probes: list[tuple[bool, float]] | None = [] if args.probe else None
    stop = asyncio.Event()
    probe_task = (
        asyncio.create_task(probe_loop(args.url, args.probe_interval, args.probe_timeout, stop, probes))
        if probes is not None
        else None
    )
    records = await asyncio.gather(*(one(i) for i in range(args.requests)))
    wall = time.perf_counter() - t0
    print(file=sys.stderr)
    stop.set()
    if probe_task is not None:
        await probe_task
    return list(records), wall, probes


# ---------------------------------------------------------------- main


def int_list(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default="http://localhost:8000", help="server base URL")
    common.add_argument("--model", default=None, help="model name (default: first entry of /v1/models)")
    common.add_argument("--timeout", type=float, default=300.0, help="per-request read timeout, seconds")
    common.add_argument("--warmup", type=int, default=2, help="untimed requests sent before measuring")
    common.add_argument("--seed", type=int, default=1234, help="prompt RNG seed")
    common.add_argument("--out-dir", type=Path, default=Path("results"), help="where CSV/JSON results go")

    parser = argparse.ArgumentParser(description="Streaming LLM benchmark (OpenAI /v1/completions).")
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("prompt-sweep", parents=[common], help="vary prompt length to isolate prefill")
    p.add_argument("--lengths", type=int_list, default=int_list("64,256,512,1024,2048,3584"))
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)

    o = sub.add_parser("output-sweep", parents=[common], help="vary output length to isolate decode")
    o.add_argument("--lengths", type=int_list, default=int_list("16,64,128,256,512"))
    o.add_argument("--prompt-tokens", type=int, default=256)
    o.add_argument("--repeats", type=int, default=3)

    load = sub.add_parser("load", parents=[common], help="concurrent streams of one request shape")
    load.add_argument("--prompt-tokens", type=int, default=256)
    load.add_argument("--output-tokens", type=int, default=128)
    load.add_argument("--concurrency", type=int, default=4)
    load.add_argument("--requests", type=int, default=32)
    load.add_argument("--probe", action="store_true", help="also probe /livez during the run")
    load.add_argument("--probe-interval", type=float, default=0.25)
    load.add_argument("--probe-timeout", type=float, default=1.0)
    return parser


async def main_async(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/")
    args.url = url
    concurrency = getattr(args, "concurrency", 1)
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    # trust_env=False: never send localhost benchmark traffic through a
    # corporate HTTP(S)_PROXY, which would add latency you'd then misattribute.
    async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits, trust_env=False) as client:
        try:
            ready = await client.get("/readyz")
            models = await client.get("/v1/models")
        except httpx.HTTPError as exc:
            print(f"cannot reach {url}: {exc}", file=sys.stderr)
            return 2
        if ready.status_code != 200:
            print(f"server is not ready (HTTP {ready.status_code}): {ready.text}", file=sys.stderr)
            return 2
        entry = models.json()["data"][0]
        model = args.model or entry["id"]
        max_model_len = int(entry.get("max_model_len") or 0)

        factory = PromptFactory(args.seed)
        await factory.calibrate(client, model)
        print(f"model={model} max_model_len={max_model_len or 'unknown'} "
              f"calibration: tokens = {factory.intercept:.1f} + {factory.tokens_per_word:.3f} * words", file=sys.stderr)

        if args.mode == "prompt-sweep":
            shapes = [(n, args.output_tokens) for n in args.lengths]
        elif args.mode == "output-sweep":
            shapes = [(args.prompt_tokens, n) for n in args.lengths]
        else:
            shapes = [(args.prompt_tokens, args.output_tokens)]
        if max_model_len:
            # Leave room for calibration error in the prompt length estimate.
            too_long = [(p, o) for p, o in shapes if p * 1.05 + o > max_model_len]
            if too_long:
                print(f"these shapes would exceed max_model_len={max_model_len}: {too_long}", file=sys.stderr)
                return 2

        for _ in range(args.warmup):
            await stream_completion(client, model, factory.build(shapes[0][0]), min(16, shapes[0][1]), "warmup", 0, time.perf_counter())

        config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
        config.update(model=model, max_model_len=max_model_len)
        if args.mode == "prompt-sweep":
            records = await run_sweep(client, model, factory, shapes, args.repeats, "prompt")
            summaries = report_prompt_sweep(records, args.lengths)
        elif args.mode == "output-sweep":
            records = await run_sweep(client, model, factory, shapes, args.repeats, "output")
            summaries = report_output_sweep(records, args.lengths)
        else:
            records, wall, probes = await run_load(client, model, factory, args)
            summaries = {"load": report_load(records, wall, args.concurrency, probes)}

    base = write_results(args.out_dir, args.mode, config, records, summaries)
    print(f"\nresults: {base}.csv and {base}.json")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
