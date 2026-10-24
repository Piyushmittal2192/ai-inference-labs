"""GPU inference engine: an explicit prefill + decode loop with per-phase timing.

This deliberately does NOT call model.generate(). Running the two phases by
hand makes each one visible and measurable:

  prefill  ONE forward pass over the entire prompt. It writes the KV cache for
           every prompt token and produces the first output token. All prompt
           tokens move through each weight matrix together, so the GPU does a
           lot of math per byte of weights it reads: compute-bound.
  decode   one forward pass PER new token, reusing the KV cache. Each step
           reads every weight from GPU memory to produce a single token, so it
           is bound by memory bandwidth and per-step overhead, not by FLOPs.

Every timed section ends with a .item() call that copies a token id to the
CPU. That forces CUDA to finish all queued kernels, so wall-clock time equals
GPU time. Without that sync you would only be timing kernel launches.

Everything here blocks and handles one request at a time. Call it only from a
worker thread (server/scheduler.py), never from the asyncio event loop.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from server.detokenizer import IncrementalDetokenizer
from server.protocol import EngineError, EngineOOMError, GenerationStats, SamplingParams

log = logging.getLogger("lab11.engine")

MIB = 1024 * 1024


class InferenceEngine:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str,
        gpu_memory_fraction: float,
        max_model_len: int,
        prefill_logits: str,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.gpu_memory_fraction = gpu_memory_fraction
        self.max_model_len = max_model_len
        self.prefill_logits = prefill_logits
        # "last": the LM head runs only on the final prompt position (all that is
        # needed to sample token 1). "all": logits for EVERY prompt position,
        # which is what a naive forward() does by default.
        self._logits_to_keep = 1 if prefill_logits == "last" else 0
        self._device = torch.device(device)
        self._tokenizer = None
        self._model = None
        self._eos_ids: frozenset[int] = frozenset()
        # Fast (Rust) tokenizers are not guaranteed to be thread-safe, and the
        # event loop tokenizes while a worker thread detokenizes. Each call takes
        # microseconds, so the lock costs nothing measurable.
        self._tokenizer_lock = threading.Lock()
        self.gpu_name: str | None = None
        self.gpu_total_mb: float | None = None
        self.weights_mb: float | None = None

    # ------------------------------------------------------------------ load

    def load(self) -> float:
        """Load tokenizer and weights onto the device, then warm up. Returns seconds."""
        start = time.perf_counter()
        self._prepare_device()
        log.info("loading %s@%s (first start downloads weights into HF_HOME)", self.model_id, self.revision[:12])
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        # Turing (sm_75) has fp16 tensor cores but no bf16 ones, so we load fp16
        # even though Qwen2.5 ships bf16 checkpoints.
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        model.to(self._device)
        model.eval()

        model_max = getattr(model.config, "max_position_embeddings", None)
        if model_max is not None and self.max_model_len > model_max:
            raise EngineError(
                f"MAX_MODEL_LEN={self.max_model_len} exceeds the model's max_position_embeddings={model_max}"
            )

        self._eos_ids = self._collect_eos_ids(model, tokenizer)
        self._tokenizer = tokenizer
        self._model = model

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            self.weights_mb = torch.cuda.memory_allocated(self._device) / MIB
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            "weights ready: params=%.0fM dtype=%s device=%s allocated_mb=%s attention=%s prefill_logits=%s eos_ids=%s",
            n_params / 1e6,
            next(model.parameters()).dtype,
            self._device,
            f"{self.weights_mb:.0f}" if self.weights_mb is not None else "n/a",
            getattr(model.config, "_attn_implementation", "unknown"),
            self.prefill_logits,
            sorted(self._eos_ids),
        )
        self._warmup()
        return time.perf_counter() - start

    def _prepare_device(self) -> None:
        if self._device.type != "cuda":
            log.warning("DEVICE=%s: no GPU in use, so timings will not reflect GPU behavior", self._device)
            return
        if not torch.cuda.is_available():
            raise EngineError(
                "CUDA is not available inside this container. Start it with --gpus all, and confirm "
                "`docker run --rm --gpus all ubuntu:24.04 nvidia-smi` shows your GPU."
            )
        index = self._device.index if self._device.index is not None else 0
        self._device = torch.device("cuda", index)
        torch.cuda.set_device(index)
        props = torch.cuda.get_device_properties(index)
        self.gpu_name = props.name
        self.gpu_total_mb = props.total_memory / MIB
        # A hard ceiling on what PyTorch's caching allocator may hold. Allocations
        # beyond it raise OutOfMemoryError even if the GPU has free memory left.
        torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction, index)
        log.info(
            "gpu: name=%r capability=%d.%d total_mb=%.0f memory_cap=%.0f%% (%.0f MiB) torch=%s cuda=%s",
            props.name,
            props.major,
            props.minor,
            self.gpu_total_mb,
            self.gpu_memory_fraction * 100,
            self.gpu_total_mb * self.gpu_memory_fraction,
            torch.__version__,
            torch.version.cuda,
        )
        if (props.major, props.minor) < (8, 0):
            log.info(
                "pre-Ampere GPU: fp16 only (no bf16 tensor cores) and no FlashAttention-2; "
                "SDPA uses its memory-efficient kernel instead"
            )

    @staticmethod
    def _collect_eos_ids(model, tokenizer) -> frozenset[int]:
        ids: set[int] = set()
        for source in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
            if source is None:
                continue
            if isinstance(source, int):
                ids.add(source)
            else:
                ids.update(int(token_id) for token_id in source)
        return frozenset(ids)

    def _warmup(self) -> None:
        # The first CUDA calls pay one-off costs: kernel selection, cuBLAS
        # handle creation, allocator growth. Pay them now, not on a user's request.
        start = time.perf_counter()
        params = SamplingParams(max_tokens=4, temperature=0.0, top_p=1.0, ignore_eos=True, seed=0)
        long_len = max(8, min(512, self.max_model_len - params.max_tokens - 1))
        short_ids = self.tokenize("Warm up the GPU.")
        long_ids = (self.tokenize("The quick brown fox jumps over the lazy dog. ") * 64)[:long_len]
        for ids in (short_ids, long_ids):
            self.generate(ids, params, lambda _text: None, threading.Event())
        log.info("warmup done in %.2fs", time.perf_counter() - start)

    # ------------------------------------------------------------- serving

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def tokenize(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise EngineError("tokenizer is not loaded")
        with self._tokenizer_lock:
            return list(self._tokenizer(text, add_special_tokens=True)["input_ids"])

    def _decode(self, token_ids: list[int]) -> str:
        with self._tokenizer_lock:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        on_token: Callable[[str], None],
        cancel: threading.Event,
    ) -> GenerationStats:
        """Run prefill, then decode until EOS, max_tokens, or cancel. Blocking."""
        if self._model is None:
            raise EngineError("model is not loaded")
        if not prompt_ids:
            raise EngineError("prompt is empty after tokenization")
        on_cuda = self._device.type == "cuda"
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)

        oom_message: str | None = None
        try:
            prefill_ms, step_ms, completion_tokens, finish_reason = self._run_phases(
                prompt_ids, params, on_token, cancel
            )
        except torch.cuda.OutOfMemoryError as exc:
            oom_message = str(exc).splitlines()[0]
        if oom_message is not None:
            # _run_phases has returned, so its tensors (activations, KV cache,
            # logits) are unreferenced. Hand the cached blocks back so the next
            # request starts from a clean allocator instead of a fragmented one.
            torch.cuda.empty_cache()
            raise EngineOOMError(f"CUDA out of memory on a {len(prompt_ids)}-token prompt: {oom_message}")

        peak_mb = torch.cuda.max_memory_allocated(self._device) / MIB if on_cuda else None
        return GenerationStats(
            prompt_tokens=len(prompt_ids),
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            prefill_ms=prefill_ms,
            decode_step_ms=tuple(step_ms),
            peak_memory_mb=peak_mb,
        )

    def _run_phases(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        on_token: Callable[[str], None],
        cancel: threading.Event,
    ) -> tuple[float, list[float], int, str]:
        model = self._model
        device = self._device
        generator = torch.Generator(device=device)
        if params.seed is not None:
            generator.manual_seed(params.seed)
        else:
            generator.seed()
        detok = IncrementalDetokenizer(self._decode)
        step_ms: list[float] = []
        completion_tokens = 0
        finish_reason = "length"

        with torch.inference_mode():
            # ---- PREFILL: every prompt token in one forward pass -------------
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            started = time.perf_counter()
            out = model(input_ids=input_ids, use_cache=True, logits_to_keep=self._logits_to_keep)
            next_ids = self._sample(out.logits[:, -1, :], params, generator)
            token = int(next_ids.item())  # sync point: prefill has truly finished
            prefill_ms = (time.perf_counter() - started) * 1000.0
            cache = out.past_key_values
            del out, input_ids

            # ---- DECODE: one token per forward pass, reusing the KV cache ----
            while True:
                completion_tokens += 1
                if not params.ignore_eos and token in self._eos_ids:
                    finish_reason = "stop"
                    break
                delta = detok.add(token)
                if delta:
                    on_token(delta)
                if completion_tokens >= params.max_tokens:
                    finish_reason = "length"
                    break
                if cancel.is_set():
                    finish_reason = "cancelled"
                    break
                started = time.perf_counter()
                out = model(input_ids=next_ids.view(1, 1), past_key_values=cache, use_cache=True)
                next_ids = self._sample(out.logits[:, -1, :], params, generator)
                token = int(next_ids.item())  # sync point: this decode step has finished
                step_ms.append((time.perf_counter() - started) * 1000.0)
                cache = out.past_key_values
                del out

        tail = detok.flush()
        if tail:
            on_token(tail)
        return prefill_ms, step_ms, completion_tokens, finish_reason

    @staticmethod
    def _sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> torch.Tensor:
        """Pick the next token id from [1, vocab] logits. Returns a [1] tensor."""
        if params.temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits.float() / params.temperature, dim=-1)
        if params.top_p < 1.0:
            sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[(cumulative - sorted_probs) > params.top_p] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
            return sorted_ids.gather(-1, choice).squeeze(-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    # --------------------------------------------------------------- stats

    def memory_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "device": str(self._device),
            "gpu": self.gpu_name,
            "total_mb": round(self.gpu_total_mb, 1) if self.gpu_total_mb else None,
            "cap_mb": round(self.gpu_total_mb * self.gpu_memory_fraction, 1) if self.gpu_total_mb else None,
            "weights_mb": round(self.weights_mb, 1) if self.weights_mb is not None else None,
        }
        if self._device.type == "cuda" and self._model is not None:
            snapshot.update(
                allocated_mb=round(torch.cuda.memory_allocated(self._device) / MIB, 1),
                reserved_mb=round(torch.cuda.memory_reserved(self._device) / MIB, 1),
                peak_allocated_mb_last_request=round(torch.cuda.max_memory_allocated(self._device) / MIB, 1),
            )
        return snapshot
