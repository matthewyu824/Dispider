#!/usr/bin/env python3
"""Offline video demo and WebRTC online-video LLM testbed server."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from aiohttp import web
from aiortc import RTCSessionDescription, RTCPeerConnection
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
RUNTIME = ROOT / "tmp" / "web_demo"
UPLOADS = RUNTIME / "uploads"
ONLINE_RUNTIME = RUNTIME / "online"
PYTHON = ROOT / ".venv" / "bin" / "python"
WORKER_SCRIPT = Path(__file__).resolve().parent / "model_worker.py"
QUICK_RESULTS = ROOT / "tmp" / "videomme_quick_results"


def discover_model_path() -> Path:
    """Resolve a compatible checkpoint without coupling the app to a brand."""
    configured = os.environ.get("ONLINE_VIDEO_LLM_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    checkpoint_root = ROOT / "checkpoints"
    preferred = checkpoint_root / "model"
    if preferred.exists():
        return preferred

    for config_path in sorted(checkpoint_root.glob("*/config.json")):
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "LongQwen2ForCausalLM" in config.get("architectures", []):
            return config_path.parent
    return preferred


MODEL = discover_model_path()
SAMPLES = {
    "test": ROOT / "tmp" / "test_video.mp4",
    "animation": ROOT / "tmp" / "videomme_quick" / "drbi6HK1gSc_h264.mp4",
    "car": ROOT / "tmp" / "videomme_quick" / "m4qhFFdHTCc_h264.mp4",
    "football": ROOT / "tmp" / "web_samples" / "uJFbOrNrL3w_h264.mp4",
    "life-tip": ROOT / "tmp" / "web_samples" / "KOwR0Ln46Ks_h264.mp4",
    "fashion": ROOT / "tmp" / "web_samples" / "2iHmlBmLKUE_h264.mp4",
    "stage": ROOT / "tmp" / "web_samples" / "PSC_HUeqaUk_h264.mp4",
    "law": ROOT / "tmp" / "web_samples" / "jTzKgI68VLc_h264.mp4",
    "geography": ROOT / "tmp" / "web_samples" / "D52rTzibFRc_h264.mp4",
    "football-7": ROOT / "tmp" / "web_samples" / "PYqmI0Hiaho_h264.mp4",
}
SAMPLE_NAMES = {
    "test": "彩色电视",
    "animation": "动画女孩",
    "car": "缺少车轮",
    "football": "足球比赛",
    "life-tip": "生活技巧",
    "fashion": "时尚穿搭",
    "stage": "舞台表演",
    "law": "法律知识",
    "geography": "地理现象",
    "football-7": "7号球员",
}

JOBS: dict[str, dict[str, Any]] = {}
SESSIONS: dict[str, "OnlineSession"] = {}
WORKERS: dict[int, "GPUWorker"] = {}
GPU_LOCKS = {0: asyncio.Lock(), 1: asyncio.Lock()}
MOCK_MODEL = False


def now_ms() -> int:
    return int(time.time() * 1000)


def clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        return min(max(int(value), low), high)
    except (TypeError, ValueError):
        return default


def extract_answer(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    ignored = (
        "Special tokens",
        "You are using",
        "Loading checkpoint",
        "We detected",
        "UserWarning",
        "warnings.warn",
        "FutureWarning",
    )
    candidates = [
        line
        for line in lines
        if not line.startswith(ignored) and not re.match(r"^\d+%", line)
    ]
    if not candidates:
        return ""
    answer = re.sub(
        r"^[\s.,;:!?…，。；：！？]+(?=[A-Za-z0-9\u4e00-\u9fff])",
        "",
        candidates[-1],
    ).strip()
    return answer if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", answer) else ""


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key != "source_path"}


def evaluation_summary() -> dict[str, Any]:
    videos, rows = 0, []
    for path in sorted(QUICK_RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        videos += len(data)
        for video in data:
            for question in video.get("questions", []):
                response = question.get("response", "")
                match = re.search(r"[ABCD]", response)
                prediction = match.group(0) if match else ""
                options = question.get("options", [])
                option_map = {
                    option[:1]: option[3:]
                    if len(option) > 3 and option[1:3] == ". "
                    else option
                    for option in options
                }
                rows.append(
                    {
                        "video_id": video.get("video_id"),
                        "question_id": question.get("question_id"),
                        "task_type": question.get("task_type"),
                        "question": question.get("question"),
                        "options": options,
                        "prediction": prediction,
                        "answer": question.get("answer"),
                        "response": response,
                        "prediction_text": option_map.get(
                            prediction, "未识别到有效选项"
                        ),
                        "answer_text": option_map.get(question.get("answer"), ""),
                        "correct": prediction == question.get("answer"),
                    }
                )
    correct = sum(row["correct"] for row in rows)
    task_stats: dict[str, dict[str, int]] = {}
    for row in rows:
        stat = task_stats.setdefault(
            row["task_type"], {"correct": 0, "total": 0}
        )
        stat["correct"] += int(row["correct"])
        stat["total"] += 1
    tasks = [
        {
            "name": name,
            **stat,
            "accuracy": round(stat["correct"] / stat["total"] * 100, 1),
        }
        for name, stat in sorted(task_stats.items())
    ]
    return {
        "videos": videos,
        "questions": len(rows),
        "answered": sum(bool(row["prediction"]) for row in rows),
        "correct": correct,
        "accuracy": round(correct / len(rows) * 100, 1) if rows else 0,
        "paper_accuracy": 57.2,
        "comparable": False,
        "configuration": {
            "dataset": "VideoMME quick subset",
            "sampling": "1 clip / 16 frames",
            "subtitles": False,
            "gpus": 2,
        },
        "tasks": tasks,
        "rows": rows,
    }


async def run_command(*command: str, timeout: int = 900) -> subprocess.CompletedProcess:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"命令超时：{command[0]}") from None
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def transcode_if_needed(source: Path, job: dict[str, Any]) -> Path:
    probe = await run_command(
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=nw=1:nk=1",
        str(source),
        timeout=30,
    )
    codec = probe.stdout.strip().lower()
    job["original_codec"] = codec or "unknown"
    if codec in {"h264", "mpeg4"}:
        return source
    target = UPLOADS / f"{job['id']}_h264.mp4"
    job.update(stage="transcoding", message=f"正在将 {codec or '未知编码'} 转换为 H.264")
    result = await run_command(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        str(target),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "视频转码失败")
    return target


class GPUWorker:
    """A persistent model process bound to one physical GPU."""

    response_prefix = "__OVLLM_RESPONSE__"
    ready_prefix = "__OVLLM_READY__"

    def __init__(self, gpu: int):
        self.gpu = gpu
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.logs: deque[str] = deque(maxlen=60)
        self.stderr_task: asyncio.Task | None = None

    async def _consume_stderr(self) -> None:
        assert self.process and self.process.stderr
        while line := await self.process.stderr.readline():
            self.logs.append(line.decode(errors="replace").rstrip())

    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(self.gpu),
                "PYTHONPATH": str(ROOT),
                "PYTHONWARNINGS": "ignore",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": str(ROOT / "tmp" / "hf_runtime_cache"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        self.process = await asyncio.create_subprocess_exec(
            str(PYTHON),
            "-u",
            str(WORKER_SCRIPT),
            "--model-path",
            str(MODEL),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ROOT),
            env=env,
        )
        self.stderr_task = asyncio.create_task(self._consume_stderr())
        assert self.process.stdout
        deadline = asyncio.get_running_loop().time() + 600
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                await self.stop()
                raise RuntimeError(f"GPU {self.gpu} 模型加载超时")
            line = await asyncio.wait_for(self.process.stdout.readline(), remaining)
            if not line:
                detail = "\n".join(self.logs)
                raise RuntimeError(f"GPU {self.gpu} 模型进程提前退出\n{detail}")
            text = line.decode(errors="replace").rstrip()
            if text.startswith(self.ready_prefix):
                return
            self.logs.append(text)

    async def infer(
        self,
        video_path: Path,
        prompt: str,
        max_new_tokens: int,
        max_clips: int,
        min_new_tokens: int = 0,
    ) -> dict[str, Any]:
        async with self.lock:
            await self.start()
            assert self.process and self.process.stdin and self.process.stdout
            request_id = uuid.uuid4().hex[:12]
            payload = {
                "id": request_id,
                "video_path": str(video_path),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "max_clips": max_clips,
                "min_new_tokens": min_new_tokens,
            }
            self.process.stdin.write((json.dumps(payload) + "\n").encode())
            await self.process.stdin.drain()
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    detail = "\n".join(self.logs)
                    self.process = None
                    raise RuntimeError(f"GPU {self.gpu} 模型进程中断\n{detail}")
                text = line.decode(errors="replace").rstrip()
                if not text.startswith(self.response_prefix):
                    self.logs.append(text)
                    continue
                response = json.loads(text[len(self.response_prefix) :])
                if response.get("id") != request_id:
                    continue
                if response.get("error"):
                    raise RuntimeError(response["error"])
                return response

    async def stop(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 8)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.stderr_task:
            self.stderr_task.cancel()
        self.process = None


def get_worker(gpu: int) -> GPUWorker:
    return WORKERS.setdefault(gpu, GPUWorker(gpu))


@dataclass
class FrameSample:
    media_time: float
    received_at: int
    image: np.ndarray


@dataclass
class OnlineSession:
    id: str
    prompt: str
    gpu: int
    window_seconds: int
    interval_seconds: int
    max_new_tokens: int
    max_clips: int
    auto_query: bool
    created_at: int = field(default_factory=now_ms)
    status: str = "created"
    transport_state: str = "new"
    media_time: float = 0.0
    source_duration: float = 0.0
    received_frames: int = 0
    sampled_frames: int = 0
    last_sample_time: float = -1.0
    last_trigger_time: float = 0.0
    inference_running: bool = False
    frames: deque[FrameSample] = field(default_factory=deque)
    events: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    responses: list[dict[str, Any]] = field(default_factory=list)
    client_stats: dict[str, Any] = field(default_factory=dict)
    pc: RTCPeerConnection | None = None
    consumer_task: asyncio.Task | None = None
    inference_tasks: set[asyncio.Task] = field(default_factory=set)
    first_frame_time: float | None = None

    def event(self, kind: str, message: str, **extra: Any) -> None:
        self.events.append(
            {
                "at": now_ms(),
                "media_time": round(self.media_time, 2),
                "kind": kind,
                "message": message,
                **extra,
            }
        )

    def trim_frames(self) -> None:
        cutoff = self.media_time - self.window_seconds
        while self.frames and self.frames[0].media_time < cutoff:
            self.frames.popleft()

    def public(self) -> dict[str, Any]:
        buffered_seconds = (
            max(0.0, self.frames[-1].media_time - self.frames[0].media_time)
            if len(self.frames) > 1
            else 0.0
        )
        latest = self.responses[-1] if self.responses else None
        latest_frame = self.frames[-1] if self.frames else None
        return {
            "id": self.id,
            "prompt": self.prompt,
            "gpu": self.gpu,
            "window_seconds": self.window_seconds,
            "interval_seconds": self.interval_seconds,
            "auto_query": self.auto_query,
            "status": self.status,
            "transport_state": self.transport_state,
            "created_at": self.created_at,
            "media_time": round(self.media_time, 2),
            "source_duration": round(self.source_duration, 2),
            "inference_running": self.inference_running,
            "latest_response": latest,
            "responses": self.responses[-20:],
            "events": list(self.events)[-40:],
            "preview": {
                "available": latest_frame is not None,
                "url": f"/api/online/sessions/{self.id}/preview.jpg",
                "media_time": round(latest_frame.media_time, 2)
                if latest_frame
                else 0,
                "received_at": latest_frame.received_at if latest_frame else 0,
                "sampled_frame": self.sampled_frames,
            },
            "metrics": {
                "received_frames": self.received_frames,
                "sampled_frames": self.sampled_frames,
                "buffered_frames": len(self.frames),
                "buffered_seconds": round(buffered_seconds, 2),
                "input_fps": self.client_stats.get("fps", 0),
                "bitrate_kbps": self.client_stats.get("bitrate_kbps", 0),
                "packets_sent": self.client_stats.get("packets_sent", 0),
                "transport_lag_ms": self.client_stats.get("transport_lag_ms", 0),
                "last_inference_ms": latest.get("total_latency_ms", 0)
                if latest
                else 0,
                "model_latency_ms": latest.get("model_latency_ms", 0)
                if latest
                else 0,
            },
        }

    async def consume_video(self, track: Any) -> None:
        self.status = "streaming"
        self.transport_state = "connected"
        self.event("transport", "服务端已接收 WebRTC 视频轨道")
        sample_period = 0.25
        try:
            while True:
                frame = await track.recv()
                self.received_frames += 1
                raw_time = float(frame.time or 0)
                if self.first_frame_time is None:
                    self.first_frame_time = raw_time
                self.media_time = max(0.0, raw_time - self.first_frame_time)
                if (
                    self.last_sample_time >= 0
                    and self.media_time - self.last_sample_time < sample_period
                ):
                    continue
                self.last_sample_time = self.media_time
                image = frame.to_ndarray(format="rgb24")
                image = resize_frame(image)
                self.frames.append(FrameSample(self.media_time, now_ms(), image))
                self.sampled_frames += 1
                self.trim_frames()
                if (
                    self.auto_query
                    and not self.inference_running
                    and self.media_time >= max(2.0, self.interval_seconds)
                    and self.media_time - self.last_trigger_time
                    >= self.interval_seconds
                ):
                    self.last_trigger_time = self.media_time
                    self.schedule_inference("interval")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.status != "stopped":
                self.status = "ended"
                self.transport_state = "ended"
                self.event("transport", f"视频轨道结束：{exc}")

    def schedule_inference(self, reason: str) -> bool:
        if self.inference_running or len(self.frames) < 4:
            return False
        task = asyncio.create_task(run_online_inference(self, reason))
        self.inference_tasks.add(task)
        task.add_done_callback(self.inference_tasks.discard)
        return True

    async def stop(self) -> None:
        self.status = "stopped"
        self.transport_state = "closed"
        self.event("transport", "在线会话已停止")
        if self.consumer_task:
            self.consumer_task.cancel()
        for task in list(self.inference_tasks):
            task.cancel()
        if self.pc:
            await self.pc.close()


def resize_frame(image: np.ndarray, max_width: int = 640) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    ratio = max_width / image.shape[1]
    size = (max_width, max(2, int(image.shape[0] * ratio)))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))


def write_window_clip(path: Path, frames: list[FrameSample], fps: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].image.shape[:2]
    width -= width % 2
    height -= height % 2
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23"}
    try:
        for index, sample in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(
                sample.image[:height, :width], format="rgb24"
            )
            video_frame.pts = index
            video_frame.time_base = Fraction(1, fps)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


async def run_online_inference(session: OnlineSession, reason: str) -> None:
    session.inference_running = True
    trigger_wall = now_ms()
    frames = list(session.frames)
    media_start = frames[0].media_time
    media_end = frames[-1].media_time
    sequence = len(session.responses) + 1
    clip_path = ONLINE_RUNTIME / session.id / f"window_{sequence:03d}.mp4"
    session.event(
        "inference",
        f"触发第 {sequence} 次推理",
        reason=reason,
        window=[round(media_start, 2), round(media_end, 2)],
    )
    try:
        await asyncio.to_thread(write_window_clip, clip_path, frames)
        prompt = (
            f"{session.prompt}\n"
            f"The video window contains only observations from {media_start:.1f} "
            f"to {media_end:.1f} seconds. Answer using only observed content. "
            "Give one concrete complete sentence. Begin directly with the "
            "observed event or object; do not return an empty answer."
        )
        if MOCK_MODEL:
            await asyncio.sleep(0.15)
            result = {
                "answer": (
                    f"[Mock] 已接收 {media_start:.1f}–{media_end:.1f}s "
                    f"窗口，共 {len(frames)} 帧。"
                ),
                "model_seconds": 0.15,
            }
        else:
            async with GPU_LOCKS[session.gpu]:
                result = await get_worker(session.gpu).infer(
                    clip_path,
                    prompt,
                    session.max_new_tokens,
                    session.max_clips,
                    min_new_tokens=8,
                )
        completed_at = now_ms()
        answer = extract_answer(result.get("answer", ""))
        response = {
            "sequence": sequence,
            "reason": reason,
            "question": session.prompt,
            "answer": answer,
            "disposition": "answered" if answer else "silent",
            "media_start": round(media_start, 2),
            "media_end": round(media_end, 2),
            "frames": len(frames),
            "triggered_at": trigger_wall,
            "completed_at": completed_at,
            "total_latency_ms": completed_at - trigger_wall,
            "model_latency_ms": round(result.get("model_seconds", 0) * 1000),
        }
        session.responses.append(response)
        if answer:
            session.event(
                "answer",
                f"第 {sequence} 次回答完成",
                sequence=sequence,
                latency_ms=response["total_latency_ms"],
            )
        else:
            session.event(
                "silent",
                f"第 {sequence} 个窗口未生成可见文本",
                sequence=sequence,
                latency_ms=response["total_latency_ms"],
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        session.event("error", f"在线推理失败：{exc}")
    finally:
        session.inference_running = False


async def run_offline_job(job: dict[str, Any]) -> None:
    started = time.monotonic()
    gpu = job["gpu"]
    try:
        job.update(
            status="running",
            stage="preparing",
            message="检查视频编码",
            started_at=now_ms(),
        )
        video = await transcode_if_needed(Path(job["source_path"]), job)
        job.update(stage="loading", message="加载模型并生成回答")
        if MOCK_MODEL:
            await asyncio.sleep(0.15)
            result = {"answer": "[Mock] 离线推理链路正常。"}
        else:
            async with GPU_LOCKS[gpu]:
                result = await get_worker(gpu).infer(
                    video,
                    job["prompt"],
                    job["max_new_tokens"],
                    job["max_clips"],
                )
        answer = extract_answer(result.get("answer", ""))
        if not answer:
            raise RuntimeError("推理完成但未解析到模型输出")
        job.update(
            status="completed",
            stage="completed",
            message="推理完成",
            answer=answer,
            elapsed_seconds=round(time.monotonic() - started, 2),
            completed_at=now_ms(),
            media_url=f"/media/{video.name}",
        )
    except Exception as exc:
        job.update(
            status="failed",
            stage="failed",
            message="推理失败",
            error=str(exc),
            elapsed_seconds=round(time.monotonic() - started, 2),
            completed_at=now_ms(),
        )


async def health(_: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "model_ready": MODEL.exists(),
            "gpus": [0, 1],
            "mock_model": MOCK_MODEL,
            "online": {
                "webrtc": True,
                "transport": "WebRTC",
                "strategy": "sliding-window",
                "camera_required": False,
            },
            "samples": [
                {
                    "id": key,
                    "name": SAMPLE_NAMES[key],
                    "available": value.exists(),
                    "url": f"/sample/{key}",
                }
                for key, value in SAMPLES.items()
            ],
        }
    )


async def get_job(request: web.Request) -> web.Response:
    job = JOBS.get(request.match_info["job_id"])
    if not job:
        raise web.HTTPNotFound(text="任务不存在")
    return web.json_response(public_job(job))


async def create_offline_job(request: web.Request) -> web.Response:
    reader = await request.multipart()
    fields: dict[str, str] = {}
    uploaded_path: Path | None = None
    uploaded_name = ""
    async for part in reader:
        if part.filename:
            suffix = Path(part.filename).suffix.lower() or ".mp4"
            uploaded_name = Path(part.filename).name
            temporary_id = uuid.uuid4().hex[:12]
            uploaded_path = UPLOADS / f"{temporary_id}{suffix}"
            with uploaded_path.open("wb") as handle:
                while chunk := await part.read_chunk(1024 * 1024):
                    handle.write(chunk)
        else:
            fields[part.name] = await part.text()
    prompt = fields.get("prompt", "").strip()
    gpu = clamp(fields.get("gpu"), 0, 1, 0)
    if not prompt:
        raise web.HTTPBadRequest(text="请输入问题")
    sample_id = fields.get("sample_id", "")
    source = uploaded_path or SAMPLES.get(sample_id)
    source_name = uploaded_name or SAMPLE_NAMES.get(sample_id, "")
    if not source or not source.exists():
        raise web.HTTPBadRequest(text="请选择或上传视频")
    job_id = uuid.uuid4().hex[:12]
    if uploaded_path:
        target = UPLOADS / f"{job_id}{uploaded_path.suffix}"
        uploaded_path.rename(target)
        source = target
    job = {
        "id": job_id,
        "status": "created",
        "stage": "created",
        "message": "任务已创建",
        "prompt": prompt,
        "gpu": gpu,
        "max_new_tokens": clamp(fields.get("max_new_tokens"), 8, 512, 256),
        "max_clips": clamp(fields.get("max_clips"), 1, 8, 1),
        "source_name": source_name,
        "source_path": str(source),
        "created_at": now_ms(),
        "answer": "",
        "error": "",
    }
    JOBS[job_id] = job
    asyncio.create_task(run_offline_job(job))
    return web.json_response(public_job(job), status=202)


async def create_online_session(request: web.Request) -> web.Response:
    payload = await request.json()
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise web.HTTPBadRequest(text="请输入在线观察问题")
    if len(SESSIONS) >= 16:
        oldest = min(SESSIONS.values(), key=lambda item: item.created_at)
        await oldest.stop()
        SESSIONS.pop(oldest.id, None)
    session_id = uuid.uuid4().hex[:12]
    session = OnlineSession(
        id=session_id,
        prompt=prompt,
        gpu=clamp(payload.get("gpu"), 0, 1, 0),
        window_seconds=clamp(payload.get("window_seconds"), 4, 60, 8),
        interval_seconds=clamp(payload.get("interval_seconds"), 4, 60, 8),
        max_new_tokens=clamp(payload.get("max_new_tokens"), 8, 512, 128),
        max_clips=clamp(payload.get("max_clips"), 1, 2, 1),
        auto_query=bool(payload.get("auto_query", True)),
    )
    session.event("session", "在线会话已创建")
    SESSIONS[session_id] = session
    return web.json_response(session.public(), status=201)


def get_session(request: web.Request) -> OnlineSession:
    session = SESSIONS.get(request.match_info["session_id"])
    if not session:
        raise web.HTTPNotFound(text="在线会话不存在")
    return session


async def online_session_status(request: web.Request) -> web.Response:
    return web.json_response(get_session(request).public())


async def online_session_preview(request: web.Request) -> web.Response:
    session = get_session(request)
    if not session.frames:
        raise web.HTTPNotFound(text="服务端尚未收到可预览的视频帧")
    frame = session.frames[-1]
    output = io.BytesIO()
    Image.fromarray(frame.image).save(output, format="JPEG", quality=82)
    return web.Response(
        body=output.getvalue(),
        content_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Media-Time": f"{frame.media_time:.2f}",
            "X-Sampled-Frame": str(session.sampled_frames),
        },
    )


async def manual_online_query(request: web.Request) -> web.Response:
    session = get_session(request)
    if len(session.frames) < 4:
        raise web.HTTPConflict(text="缓冲帧不足，请先播放视频")
    if not session.schedule_inference("manual"):
        raise web.HTTPConflict(text="已有推理任务正在运行")
    return web.json_response({"ok": True}, status=202)


async def stop_online_session(request: web.Request) -> web.Response:
    session = get_session(request)
    await session.stop()
    return web.json_response(session.public())


async def rtc_offer(request: web.Request) -> web.Response:
    payload = await request.json()
    session = SESSIONS.get(str(payload.get("session_id", "")))
    if not session:
        raise web.HTTPNotFound(text="在线会话不存在")
    pc = RTCPeerConnection()
    session.pc = pc
    session.transport_state = "connecting"
    session.event("transport", "正在协商 WebRTC 连接")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        session.transport_state = pc.connectionState
        session.event("transport", f"WebRTC 状态：{pc.connectionState}")
        if pc.connectionState in {"failed", "closed"} and session.status != "stopped":
            session.status = "failed" if pc.connectionState == "failed" else "ended"

    @pc.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        session.event("transport", "遥测 DataChannel 已连接")

        @channel.on("message")
        def on_message(message: Any) -> None:
            try:
                stats = json.loads(message)
            except (TypeError, json.JSONDecodeError):
                return
            if stats.get("type") != "telemetry":
                return
            sent_at = int(stats.get("sent_at", now_ms()))
            session.source_duration = float(stats.get("duration") or 0)
            session.client_stats = {
                "fps": round(float(stats.get("fps") or 0), 1),
                "bitrate_kbps": round(float(stats.get("bitrate_kbps") or 0), 1),
                "packets_sent": int(stats.get("packets_sent") or 0),
                "transport_lag_ms": max(0, min(now_ms() - sent_at, 60_000)),
            }

    @pc.on("track")
    def on_track(track: Any) -> None:
        if track.kind == "video":
            session.consumer_task = asyncio.create_task(session.consume_video(track))

    offer = RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def serve_sample(request: web.Request) -> web.StreamResponse:
    path = SAMPLES.get(request.match_info["sample_id"])
    if not path or not path.exists():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def serve_media(request: web.Request) -> web.StreamResponse:
    name = Path(request.match_info["name"]).name
    candidates = [UPLOADS / name, *[path for path in SAMPLES.values() if path.name == name]]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if not path:
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def serve_index(_: web.Request) -> web.StreamResponse:
    return web.FileResponse(STATIC / "index.html")


async def serve_evaluation(_: web.Request) -> web.Response:
    return web.json_response(evaluation_summary())


async def json_error_middleware(
    request: web.Request, handler: Any
) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as exc:
        if request.path.startswith("/api/"):
            try:
                message = exc.text or exc.reason
            except Exception:
                message = exc.reason
            return web.json_response({"error": message}, status=exc.status)
        raise
    except Exception as exc:
        if request.path.startswith("/api/"):
            return web.json_response({"error": str(exc)}, status=500)
        raise


async def cleanup(_: web.Application) -> None:
    await asyncio.gather(
        *(session.stop() for session in list(SESSIONS.values())),
        return_exceptions=True,
    )
    await asyncio.gather(
        *(worker.stop() for worker in list(WORKERS.values())),
        return_exceptions=True,
    )


def build_app() -> web.Application:
    app = web.Application(
        client_max_size=4 * 1024**3,
        middlewares=[web.middleware(json_error_middleware)],
    )
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/evaluation", serve_evaluation)
    app.router.add_post("/api/infer", create_offline_job)
    app.router.add_get("/api/jobs/{job_id}", get_job)
    app.router.add_post("/api/online/sessions", create_online_session)
    app.router.add_get(
        "/api/online/sessions/{session_id}", online_session_status
    )
    app.router.add_get(
        "/api/online/sessions/{session_id}/preview.jpg",
        online_session_preview,
    )
    app.router.add_post(
        "/api/online/sessions/{session_id}/query", manual_online_query
    )
    app.router.add_post(
        "/api/online/sessions/{session_id}/stop", stop_online_session
    )
    app.router.add_post("/api/rtc/offer", rtc_offer)
    app.router.add_get("/sample/{sample_id}", serve_sample)
    app.router.add_get("/media/{name}", serve_media)
    app.router.add_get("/", serve_index)
    app.router.add_get("/index.html", serve_index)
    app.router.add_static("/", STATIC, show_index=False)
    app.on_cleanup.append(cleanup)
    return app


def main() -> None:
    global MOCK_MODEL
    parser = argparse.ArgumentParser(description="Online Video LLM Testbed")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--mock-model",
        action="store_true",
        help="Validate WebRTC and UI without loading the model",
    )
    args = parser.parse_args()
    MOCK_MODEL = args.mock_model
    UPLOADS.mkdir(parents=True, exist_ok=True)
    ONLINE_RUNTIME.mkdir(parents=True, exist_ok=True)
    print(
        f"Online Video LLM Testbed: http://{args.host}:{args.port} "
        f"(mock_model={MOCK_MODEL})",
        flush=True,
    )
    web.run_app(
        build_app(),
        host=args.host,
        port=args.port,
        print=None,
        access_log_format='%a "%r" %s %Tf',
    )


if __name__ == "__main__":
    main()
