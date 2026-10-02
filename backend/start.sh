#!/usr/bin/env sh
set -eu

: "${PORT:=8000}"
: "${WEB_CONCURRENCY:=1}"

exec uv run uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers "$WEB_CONCURRENCY" \
  --timeout-keep-alive 30 \
  --proxy-headers \
  --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-127.0.0.1}"
