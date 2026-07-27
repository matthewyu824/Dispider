#!/usr/bin/env bash

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ONLINE_VIDEO_LLM_MODEL_PATH:-$ROOT/checkpoints/model}"
CACHE_DIR="$MODEL_DIR/.cache/huggingface/download"
LOG_FILE="$ROOT/checkpoints/download_progress.log"
PID_FILE="$ROOT/checkpoints/download_progress.pid"
FILE2="$CACHE_DIR/t9msAuTjAZjuQnmzGOwTjiptvIU=.3242c9a0b6bf3dd33f134682ca8d39e726b4329c67fe2b823f213de5ba691cb7.incomplete"
FILE3="$CACHE_DIR/DaGOU-KRMVrY0aYktrsE34tL0Bs=.811c1c54adbb60728b96e24da2c16da0e6d80db3b8fbb26b30a6f0ec103358f7.incomplete"

size_bytes() {
  [ -f "$1" ] && stat -c%s "$1" || echo 0
}

fmt_mb() {
  awk -v b="$1" 'BEGIN { printf "%.1f MB", b/1024/1024 }'
}

mkdir -p "$ROOT/checkpoints"
cd "$ROOT"
echo "$$" > "$PID_FILE"

prev_total=-1

while true; do
  cur2="$(size_bytes "$FILE2")"
  cur3="$(size_bytes "$FILE3")"
  cur_total=$((cur2 + cur3))

  if [ "$prev_total" -lt 0 ]; then
    speed="n/a"
  else
    delta=$((cur_total - prev_total))
    speed="$(awk -v d="$delta" 'BEGIN { printf "%.2f MB/s", d/1024/1024/5 }')"
  fi

  printf '[%s] model-00002=%s | model-00003=%s | partial_total=%s | speed=%s\n' \
    "$(date '+%F %T')" \
    "$(fmt_mb "$cur2")" \
    "$(fmt_mb "$cur3")" \
    "$(fmt_mb "$cur_total")" \
    "$speed" >> "$LOG_FILE"

  prev_total=$cur_total
  sleep 5
done
