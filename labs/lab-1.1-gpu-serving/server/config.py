"""Runtime configuration for the Lab 1.1 server.

Every knob is an environment variable, read once at startup and validated
before the server accepts traffic. A bad value crashes the container at boot
instead of surfacing as a confusing 500 an hour later.

Defaults are sized for a single 8 GB Turing GPU (RTX 2080).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Pin weights to an exact commit, the same way requirements.txt pins packages.
# "main" is a moving target: if the repo owner pushes new weights, every
# replica silently changes behavior on its next restart.
DEFAULT_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

INFERENCE_MODES = ("threadpool", "blocking")
PREFILL_LOGITS_MODES = ("last", "all")
DEVICES = ("cuda", "cpu")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ConfigError(ValueError):
    """An environment variable holds an invalid value."""


def _str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _choice(name: str, default: str, choices: tuple[str, ...], *, upper: bool = False) -> str:
    value = _str(name, default)
    value = value.upper() if upper else value.lower()
    if value not in choices:
        raise ConfigError(f"{name} must be one of {choices}, got {value!r}")
    return value


def _int(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ConfigError(f"{name} must be {bound}, got {value}")
    return value


def _float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    model_id: str
    model_revision: str
    served_model_name: str
    device: str
    gpu_memory_fraction: float
    max_model_len: int
    max_tokens_limit: int
    max_concurrency: int
    max_queue_depth: int
    queue_timeout_s: float
    prefill_logits: str
    inference_mode: str
    log_level: str

    @property
    def blocking(self) -> bool:
        return self.inference_mode == "blocking"

    @property
    def max_prompt_chars(self) -> int:
        # Cheap guard on raw payload size before we spend time tokenizing.
        # Real English averages ~4 characters per token; 16 leaves headroom.
        return self.max_model_len * 16

    @classmethod
    def from_env(cls) -> Settings:
        model_id = _str("MODEL_ID", DEFAULT_MODEL_ID)
        default_revision = DEFAULT_MODEL_REVISION if model_id == DEFAULT_MODEL_ID else "main"
        max_model_len = _int("MAX_MODEL_LEN", 4096, minimum=64)
        max_tokens_limit = _int("MAX_TOKENS_LIMIT", 1024, minimum=1)
        if max_tokens_limit >= max_model_len:
            raise ConfigError(
                f"MAX_TOKENS_LIMIT ({max_tokens_limit}) must be smaller than MAX_MODEL_LEN ({max_model_len})"
            )
        return cls(
            model_id=model_id,
            model_revision=_str("MODEL_REVISION", default_revision),
            served_model_name=_str("SERVED_MODEL_NAME", model_id),
            device=_choice("DEVICE", "cuda", DEVICES),
            gpu_memory_fraction=_float("GPU_MEMORY_FRACTION", 0.60, minimum=0.05, maximum=1.0),
            max_model_len=max_model_len,
            max_tokens_limit=max_tokens_limit,
            max_concurrency=_int("MAX_CONCURRENCY", 1, minimum=1, maximum=8),
            max_queue_depth=_int("MAX_QUEUE_DEPTH", 32, minimum=0),
            queue_timeout_s=_float("QUEUE_TIMEOUT_S", 60.0, minimum=0.1, maximum=3600.0),
            prefill_logits=_choice("PREFILL_LOGITS", "last", PREFILL_LOGITS_MODES),
            inference_mode=_choice("INFERENCE_MODE", "threadpool", INFERENCE_MODES),
            log_level=_choice("LOG_LEVEL", "INFO", LOG_LEVELS, upper=True),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
