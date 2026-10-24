# syntax=docker/dockerfile:1
#
# Lab 1.1: production-style image for a single-GPU model server.
#
#   * Multi-stage: pip caches and build state never reach the runtime image.
#   * CUDA comes from PyTorch's pip wheels (cuBLAS, cuDNN and the CUDA runtime
#     ship as pip packages). Only the DRIVER comes from the host, injected by
#     the NVIDIA Container Toolkit when you pass --gpus. No 10 GB CUDA devel
#     base image is needed to run inference.
#   * cu126 wheels support Turing (sm_75, RTX 20-series) and run on any driver
#     whose nvidia-smi reports CUDA 12.6 or newer.
#   * Weights are NOT baked in. They live in a volume mounted at /models, so
#     the image stays small enough to pull fast and the weights download once.
#   * Non-root user, tini as PID 1, exec-form CMD, built-in HEALTHCHECK.

ARG PYTHON_VERSION=3.12

############################ builder ############################
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_VERSION=2.9.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Layer 1: torch alone. It is the biggest layer and changes least often, so it
# stays cached when you edit requirements.txt or the server code.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"

# Layer 2: everything else. Re-runs only when requirements.txt changes.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

############################ runtime ############################
FROM python:${PYTHON_VERSION}-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app \
 && mkdir -p /models \
 && chown app:app /models

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /srv
COPY --chown=app:app server/ ./server/

USER app
EXPOSE 8000

# Docker has a single health state, so it tracks readiness. In Kubernetes
# (Level 3) /livez and /readyz become two separate probes.
HEALTHCHECK --interval=10s --timeout=3s --start-period=300s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2); sys.exit(0 if r.status == 200 else 1)"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30", "--timeout-keep-alive", "5", "--no-server-header"]
