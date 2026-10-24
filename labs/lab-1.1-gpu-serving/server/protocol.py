"""The contract between the HTTP layer and the engine. Standard library only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class EngineError(RuntimeError):
    """The engine could not finish a request. The server itself is still healthy."""

    error_type = "engine_error"


class EngineOOMError(EngineError):
    """The GPU ran out of memory while serving one request."""

    error_type = "out_of_memory"


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float
    top_p: float
    ignore_eos: bool
    seed: int | None


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile. Returns None for an empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class GenerationStats:
    """What one request cost the engine, phase by phase.

    prefill_ms     the forward pass over the whole prompt, plus sampling and
                   copying the first token to the CPU
    decode_step_ms one entry per later token: a forward pass over a single
                   token that reuses the KV cache, plus sampling
    """

    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    prefill_ms: float
    decode_step_ms: tuple[float, ...]
    peak_memory_mb: float | None

    @property
    def decode_ms(self) -> float:
        return sum(self.decode_step_ms)

    @property
    def itl_mean_ms(self) -> float | None:
        if not self.decode_step_ms:
            return None
        return self.decode_ms / len(self.decode_step_ms)

    def itl_percentile_ms(self, pct: float) -> float | None:
        return percentile(self.decode_step_ms, pct)

    @property
    def prefill_tokens_per_s(self) -> float | None:
        if self.prefill_ms <= 0:
            return None
        return self.prompt_tokens / (self.prefill_ms / 1000.0)

    @property
    def decode_tokens_per_s(self) -> float | None:
        if self.decode_ms <= 0:
            return None
        return len(self.decode_step_ms) / (self.decode_ms / 1000.0)
