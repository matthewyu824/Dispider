#!/usr/bin/env bash

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ONLINE_VIDEO_LLM_MODEL_PATH:-$ROOT/checkpoints/model}"
MODEL_REPOSITORY="${ONLINE_VIDEO_LLM_MODEL_REPOSITORY:-}"
CACHE_DIR="$MODEL_DIR/.cache/huggingface/download"
LOG_DIR="$ROOT/checkpoints"
LOG_FILE="$LOG_DIR/download_watchdog.log"
PID_FILE="$LOG_DIR/download_watchdog.pid"
DOWNLOAD_CMD=(
  .venv/bin/huggingface-cli
  download
  "$MODEL_REPOSITORY"
  --local-dir
  "$MODEL_DIR"
  --max-workers
  3
)

mkdir -p "$LOG_DIR"
cd "$ROOT"

if [ -z "$MODEL_REPOSITORY" ]; then
  echo "Set ONLINE_VIDEO_LLM_MODEL_REPOSITORY to a Hugging Face model ID." >&2
  exit 2
fi

echo "$$" > "$PID_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG_FILE"
}

download_complete() {
  [ -f "$MODEL_DIR/model-00001-of-00004.safetensors" ] &&
  [ -f "$MODEL_DIR/model-00002-of-00004.safetensors" ] &&
  [ -f "$MODEL_DIR/model-00003-of-00004.safetensors" ] &&
  [ -f "$MODEL_DIR/model-00004-of-00004.safetensors" ] &&
  [ -f "$MODEL_DIR/stream_compressor/model.safetensors" ]
}

active_download_pid() {
  pgrep -af "huggingface-cli download $MODEL_REPOSITORY" | awk '/\.venv\/bin\/huggingface-cli/ {print $1; exit}'
}

cleanup_broken_resume_state() {
  if tail -n 200 "$LOG_DIR/download.log" 2>/dev/null | grep -q "416 Client Error: Range Not Satisfiable"; then
    log "detected broken resume state, removing incomplete files"
    find "$CACHE_DIR" -name '*.incomplete' -type f -delete 2>/dev/null || true
    find "$CACHE_DIR" -name '*.lock' -type f -delete 2>/dev/null || true
  fi
}

log "watchdog started"

while true; do
  if download_complete; then
    log "all model files present, watchdog exiting"
    exit 0
  fi

  running_pid="$(active_download_pid || true)"
  if [ -n "${running_pid:-}" ]; then
    log "download already running with pid=$running_pid"
    sleep 60
    continue
  fi

  cleanup_broken_resume_state
  find "$CACHE_DIR" -name '*.lock' -type f -delete 2>/dev/null || true
  log "starting download"
  HF_HUB_DISABLE_XET=1 "${DOWNLOAD_CMD[@]}" >> "$LOG_DIR/download.log" 2>&1 || log "download exited with error, will retry"
  sleep 15
done
