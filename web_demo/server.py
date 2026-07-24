#!/usr/bin/env python3
import argparse
import cgi
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
RUNTIME = ROOT / "tmp" / "web_demo"
UPLOADS = RUNTIME / "uploads"
MODEL = ROOT / "checkpoints" / "Dispider"
PYTHON = ROOT / ".venv" / "bin" / "python"
INFERENCE = ROOT / "inference.py"
QUICK_RESULTS = ROOT / "tmp" / "videomme_quick_results"
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
    "test": "彩色电视", "animation": "动画女孩", "car": "缺少车轮",
    "football": "足球比赛", "life-tip": "生活技巧", "fashion": "时尚穿搭",
    "stage": "舞台表演", "law": "法律知识", "geography": "地理现象",
    "football-7": "7号球员",
}

JOBS = {}
JOBS_LOCK = threading.Lock()
GPU_LOCKS = {0: threading.Lock(), 1: threading.Lock()}
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dispider")


def now_ms():
    return int(time.time() * 1000)


def update_job(job_id, **values):
    with JOBS_LOCK:
        JOBS[job_id].update(values)


def public_job(job):
    return {k: v for k, v in job.items() if k not in {"source_path"}}


def transcode_if_needed(source, job_id):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(source)],
        capture_output=True, text=True, timeout=30,
    )
    codec = probe.stdout.strip().lower()
    if codec in {"h264", "mpeg4"}:
        return source, codec or "unknown"
    target = UPLOADS / f"{job_id}_h264.mp4"
    update_job(job_id, stage="transcoding", message=f"正在将 {codec or '未知编码'} 转换为 H.264")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", str(target)],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "视频转码失败")
    return target, codec or "unknown"


def extract_answer(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    ignored = ("Special tokens", "You are using", "Loading checkpoint", "We detected", "UserWarning", "warnings.warn", "FutureWarning")
    candidates = [line for line in lines if not line.startswith(ignored) and not re.match(r"^\d+%", line)]
    return candidates[-1] if candidates else ""


def run_inference(job_id):
    job = JOBS[job_id]
    gpu = int(job["gpu"])
    started = time.monotonic()
    try:
        update_job(job_id, status="queued", stage="queued", message=f"等待 GPU {gpu}")
        with GPU_LOCKS[gpu]:
            update_job(job_id, status="running", stage="preparing", message="检查视频编码", started_at=now_ms())
            video, original_codec = transcode_if_needed(Path(job["source_path"]), job_id)
            update_job(job_id, stage="loading", message="加载视觉编码器与 Dispider 模型", original_codec=original_codec)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(ROOT),
                "PYTHONWARNINGS": "ignore",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": str(ROOT / "tmp" / "hf_runtime_cache"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            command = [str(PYTHON), str(INFERENCE), "--model_path", str(MODEL), "--video_path", str(video), "--prompt", job["prompt"], "--max_new_tokens", str(job["max_new_tokens"]), "--max_clips", str(job["max_clips"])]
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode:
                tail = "\n".join(combined.strip().splitlines()[-12:])
                raise RuntimeError(tail or f"推理退出码 {result.returncode}")
            answer = extract_answer(result.stdout)
            if not answer:
                raise RuntimeError("推理完成但未解析到模型输出")
            update_job(
                job_id, status="completed", stage="completed", message="推理完成", answer=answer,
                elapsed_seconds=round(time.monotonic() - started, 2), completed_at=now_ms(),
                media_url=f"/media/{video.name}",
            )
    except Exception as exc:
        update_job(job_id, status="failed", stage="failed", message="推理失败", error=str(exc), elapsed_seconds=round(time.monotonic() - started, 2), completed_at=now_ms())


def evaluation_summary():
    videos, rows = 0, []
    for path in sorted(QUICK_RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        videos += len(data)
        for video in data:
            for q in video.get("questions", []):
                response = q.get("response", "")
                match = re.search(r"[ABCD]", response)
                prediction = match.group(0) if match else ""
                options = q.get("options", [])
                option_map = {option[:1]: option[3:] if len(option) > 3 and option[1:3] == ". " else option for option in options}
                rows.append({
                    "video_id": video.get("video_id"), "question_id": q.get("question_id"),
                    "task_type": q.get("task_type"), "question": q.get("question"),
                    "options": options,
                    "prediction": prediction, "answer": q.get("answer"), "response": response,
                    "prediction_text": option_map.get(prediction, "未识别到有效选项"),
                    "answer_text": option_map.get(q.get("answer"), ""),
                    "correct": prediction == q.get("answer"),
                })
    correct = sum(row["correct"] for row in rows)
    task_stats = {}
    for row in rows:
        stat = task_stats.setdefault(row["task_type"], {"correct": 0, "total": 0})
        stat["correct"] += int(row["correct"])
        stat["total"] += 1
    tasks = [{"name": name, **stat, "accuracy": round(stat["correct"] / stat["total"] * 100, 1)} for name, stat in sorted(task_stats.items())]
    return {"videos": videos, "questions": len(rows), "answered": sum(bool(r["prediction"]) for r in rows), "correct": correct, "accuracy": round(correct / len(rows) * 100, 1) if rows else 0, "paper_accuracy": 57.2, "comparable": False, "configuration": {"dataset": "VideoMME quick subset", "sampling": "1 clip / 16 frames", "subtitles": False, "gpus": 2}, "tasks": tasks, "rows": rows}


class Handler(SimpleHTTPRequestHandler):
    server_version = "DispiderDemo/0.1"

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def json_response(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            self.json_response({"ok": True, "model_ready": MODEL.exists(), "gpus": [0, 1], "samples": [{"id": k, "name": SAMPLE_NAMES[k], "available": v.exists(), "url": f"/sample/{k}"} for k, v in SAMPLES.items()]})
            return
        if path == "/api/evaluation":
            self.json_response(evaluation_summary())
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                self.json_response(public_job(job) if job else {"error": "任务不存在"}, 200 if job else 404)
            return
        if path.startswith("/sample/"):
            sample = SAMPLES.get(path.rsplit("/", 1)[-1])
            return self.serve_file(sample)
        if path.startswith("/media/"):
            name = Path(path.rsplit("/", 1)[-1]).name
            candidates = [UPLOADS / name, *[p for p in SAMPLES.values() if p.name == name]]
            target = next((p for p in candidates if p.exists()), None)
            return self.serve_file(target)
        if path == "/logo.png":
            return self.serve_file(ROOT / "img" / "logo.png", "image/png")
        if path in {"/", "/index.html"}:
            return self.serve_file(STATIC / "index.html", "text/html; charset=utf-8")
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() in target.parents and target.exists():
            return self.serve_file(target)
        self.send_error(404)

    def serve_file(self, path, content_type=None):
        if not path or not Path(path).exists():
            self.send_error(404)
            return
        path = Path(path)
        content_type = content_type or self.guess_type(str(path))
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_POST(self):
        if urlparse(self.path).path != "/api/infer":
            self.json_response({"error": "接口不存在"}, 404)
            return
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
            prompt = form.getfirst("prompt", "").strip()
            gpu = int(form.getfirst("gpu", "0"))
            max_new_tokens = min(max(int(form.getfirst("max_new_tokens", "256")), 8), 512)
            max_clips = min(max(int(form.getfirst("max_clips", "1")), 1), 8)
            sample_id = form.getfirst("sample_id", "")
            if not prompt:
                return self.json_response({"error": "请输入问题"}, 400)
            if gpu not in GPU_LOCKS:
                return self.json_response({"error": "GPU 只能选择 0 或 1"}, 400)
            job_id = uuid.uuid4().hex[:12]
            upload = form["video"] if "video" in form else None
            if upload is not None and getattr(upload, "filename", ""):
                suffix = Path(upload.filename).suffix.lower() or ".mp4"
                source = UPLOADS / f"{job_id}{suffix}"
                with source.open("wb") as f:
                    shutil.copyfileobj(upload.file, f)
                source_name = upload.filename
            else:
                source = SAMPLES.get(sample_id)
                source_name = SAMPLE_NAMES.get(sample_id, "")
            if not source or not source.exists():
                return self.json_response({"error": "请选择或上传视频"}, 400)
            job = {"id": job_id, "status": "created", "stage": "created", "message": "任务已创建", "prompt": prompt, "gpu": gpu, "max_new_tokens": max_new_tokens, "max_clips": max_clips, "source_name": source_name, "source_path": str(source), "created_at": now_ms(), "answer": "", "error": ""}
            with JOBS_LOCK:
                JOBS[job_id] = job
            EXECUTOR.submit(run_inference, job_id)
            self.json_response(public_job(job), HTTPStatus.ACCEPTED)
        except Exception as exc:
            self.json_response({"error": str(exc)}, 500)


def main():
    parser = argparse.ArgumentParser(description="Dispider web demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dispider demo: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        EXECUTOR.shutdown(wait=False, cancel_futures=True)
        server.server_close()


if __name__ == "__main__":
    main()
