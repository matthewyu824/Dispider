#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${ONLINE_VIDEO_LLM_HOST:-0.0.0.0}"
PORT="${ONLINE_VIDEO_LLM_PORT:-7860}"

cd "${PROJECT_ROOT}"
exec .venv/bin/python web_demo/server.py --host "${HOST}" --port "${PORT}" "$@"
