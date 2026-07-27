#!/usr/bin/env python3
"""Persistent JSONL worker for repeated video LLM inference on one GPU."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback

import torch

from inference import videoStream


READY_PREFIX = "__OVLLM_READY__"
RESPONSE_PREFIX = "__OVLLM_RESPONSE__"


def emit(payload: dict) -> None:
    print(RESPONSE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    streamer = videoStream(args.model_path)
    print(READY_PREFIX + json.dumps({"ok": True}), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        started = time.monotonic()
        try:
            answer = streamer.Run(
                request["video_path"],
                request["prompt"],
                int(request.get("max_new_tokens", 128)),
                int(request.get("max_clips", 1)),
                int(request.get("min_new_tokens", 0)),
            )
            emit(
                {
                    "id": request.get("id"),
                    "answer": answer,
                    "model_seconds": round(time.monotonic() - started, 3),
                }
            )
        except Exception:
            emit(
                {
                    "id": request.get("id"),
                    "error": traceback.format_exc(limit=12),
                    "model_seconds": round(time.monotonic() - started, 3),
                }
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
