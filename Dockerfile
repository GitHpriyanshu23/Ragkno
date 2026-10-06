FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    WEB_CONCURRENCY=1 \
    RAGKNO_DATA_DIR=/data \
    HF_HOME=/data/huggingface

RUN pip install --no-cache-dir uv==0.11.14 \
    && useradd --create-home --uid 1000 ragkno \
    && mkdir -p /app /data \
    && chown -R ragkno:ragkno /app /data

WORKDIR /app

COPY --chown=ragkno:ragkno pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache \
    && .venv/bin/python -c "import torch; assert torch.version.cuda is None, 'CPU-only PyTorch is required'"

COPY --chown=ragkno:ragkno backend ./backend
COPY --chown=ragkno:ragkno src ./src

USER ragkno

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health/live', timeout=4)" || exit 1

CMD ["uv", "run", "--no-sync", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "30", "--proxy-headers", "--forwarded-allow-ips=*"]
