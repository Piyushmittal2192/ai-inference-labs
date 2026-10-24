"""Request schemas: a strict subset of the OpenAI /v1/completions API.

Unknown fields are rejected (HTTP 422) instead of silently ignored. A client
that sends `stop` or `n` and gets back a 200 that ignored them is worse off
than one that gets a clear error.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from server.config import get_settings

_SETTINGS = get_settings()


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt: str = Field(min_length=1, max_length=_SETTINGS.max_prompt_chars)
    max_tokens: int = Field(default=min(16, _SETTINGS.max_tokens_limit), ge=1, le=_SETTINGS.max_tokens_limit)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stream_options: StreamOptions | None = None
    # Not part of the OpenAI spec; vLLM accepts both. Benchmarks need ignore_eos
    # so every request generates exactly max_tokens and runs are comparable.
    ignore_eos: bool = False
    seed: int | None = Field(default=None, ge=0)
