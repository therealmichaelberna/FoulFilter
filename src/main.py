"""main.py - FoulFilter web service.

Single container (ADR-0003): FastAPI app hosting the batch UI, a sequential
Job worker, and the pipeline engines. Jobs are ephemeral (ADR-0002): state
lives in memory and scratch files; transcripts persist under /data/transcripts.
"""

import json
import logging
import os
import queue as _q
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import deque
from contextlib import asynccontextmanager

import metrics

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles

from pipeline import JobCancelled, run_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FoulFilter.web")

DATA_DIR = os.getenv("DATA_DIR") or "/data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
SCRATCH_DIR = os.path.join(DATA_DIR, "scratch")
TRANSCRIPT_DIR = os.getenv("TRANSCRIPT_DIR") or os.path.join(DATA_DIR, "transcripts")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
BAD_WORDS_PATH = os.getenv("BAD_WORDS_PATH", "/data/bad_words.txt")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB") or 4096)
ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".m4b", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
}
CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# Job manager: in-memory queue + worker thread + SSE fan-out
# ---------------------------------------------------------------------------

class JobManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.queue = deque()
        self.subscribers = set()
        self.cancelled = set()
        self.worker = threading.Thread(target=self._work, daemon=True)

    def start(self):
        self.worker.start()

    def publish(self, event):
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except Exception:  # noqa: BLE001 - a dead SSE client must not stall jobs
                pass

    def subscribe(self):
        q = _q.Queue()
        with self.lock:
            self.subscribers.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.subscribers.discard(q)

    def snapshot(self):
        with self.lock:
            return [dict(v) for v in self.jobs.values()]

    def add(self, job_id, record):
        with self.lock:
            self.jobs[job_id] = record
            self.queue.append(job_id)
        self.publish({"job_id": job_id, **record})

    def update(self, job_id, **fields):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job.update(fields)
            snapshot = dict(job)
        self.publish({"job_id": job_id, **snapshot})

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            status = job["status"]
        if status == "queued":
            with self.lock:
                try:
                    self.queue.remove(job_id)
                except ValueError:
                    pass
                job["status"] = "cancelled"
                snapshot = dict(job)
            self.publish({"job_id": job_id, **snapshot})
            _delete_job_files(job_id)
            return True
        if status == "processing":
            self.cancelled.add(job_id)  # checked at next progress checkpoint
            return True
        return False

    def check_cancel(self, job_id):
        if job_id in self.cancelled:
            self.cancelled.discard(job_id)
            raise JobCancelled()

    def _work(self):
        while True:
            with self.lock:
                job_id = self.queue[0] if self.queue else None
            if not job_id:
                threading.Event().wait(0.5)
                continue
            with self.lock:
                self.queue.popleft()
                job = dict(self.jobs[job_id])
            self._process(job)

    def _process(self, job):
        job_id = job["id"]
        started_at = time.time()
        self.update(job_id, status="processing", stage="queued", progress=0,
                    detail="Starting", started_at=started_at)
        stage_times = {}
        current_stage = [None]
        stage_start = [started_at]

        def _on_stage_change(new_stage):
            now = time.time()
            if current_stage[0] and current_stage[0] != "completed":
                elapsed = now - stage_start[0]
                stage_times[current_stage[0]] = stage_times.get(current_stage[0], 0) + elapsed
            current_stage[0] = new_stage
            stage_start[0] = now

        try:
            summary = run_job(
                input_path=job["input_path"],
                output_path=job["output_path"],
                bad_words_list_path=BAD_WORDS_PATH,
                transcript_dir=TRANSCRIPT_DIR,
                scratch_dir=os.path.join(SCRATCH_DIR, job_id),
                censor_method=job["censor_method"],
                debug=job["debug"],
                rescan=job["rescan"],
                progress=self._progress_for(job_id, _on_stage_change),
            )
            _on_stage_change("completed")
            self.update(
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                detail=f"{len(summary['hits'])} hit(s)"
                + (" (rescanned)" if summary["rescanned"] else ""),
                hits=summary["hits"],
                download_url=f"/download/{job_id}",
            )
        except JobCancelled:
            logger.info("Job %s cancelled", job_id)
            self.update(job_id, status="cancelled", detail="Cancelled by user")
            _delete_job_files(job_id)
            return
        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            logger.exception("Job %s failed", job_id)
            self.update(job_id, status="failed", detail=str(exc)[:500])
            _delete_job_files(job_id, keep_output=False)
            return

        # Record metrics for ETA learning
        try:
            clip_secs = summary.get("transcript_words", 0) * 0.3
            if not clip_secs:
                clip_secs = time.time() - started_at
            metrics.record_job(clip_secs, stage_times)
        except Exception:  # noqa: BLE001
            logger.debug("Metrics recording failed", exc_info=True)

    def _progress_for(self, job_id, on_stage_change=None):
        def cb(stage, percent, detail=""):
            self.check_cancel(job_id)
            if on_stage_change:
                on_stage_change(stage)
            self.update(job_id, stage=stage, progress=int(percent), detail=detail)
        return cb


manager = JobManager()


def _delete_job_files(job_id, keep_output=True):
    shutil.rmtree(os.path.join(SCRATCH_DIR, job_id), ignore_errors=True)
    with manager.lock:
        job = manager.jobs.get(job_id) or {}
    for key in ("input_path", "output_path"):
        path = job.get(key)
        if path and os.path.exists(path) and not (
            keep_output and key == "output_path"
        ):
            try:
                os.remove(path)
            except OSError:
                pass


@asynccontextmanager
async def lifespan(app):
    for d in (UPLOAD_DIR, OUTPUT_DIR, SCRATCH_DIR, TRANSCRIPT_DIR):
        os.makedirs(d, exist_ok=True)
    # ADR-0002: temp uploads/outputs from a previous run are stale - wipe them.
    # Transcripts persist.
    for d in (UPLOAD_DIR, SCRATCH_DIR):
        for name in os.listdir(d):
            path = os.path.join(d, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
    manager.start()
    yield


app = FastAPI(title="FoulFilter", docs_url="/docs", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename(filename: str) -> str:
    """Strip directory components and characters that break FFmpeg filters."""
    name = os.path.basename(filename or "")
    name, ext = os.path.splitext(name)
    ext = ext.lower()
    name = name.replace(":", " -")
    name = re.sub(r"[\[\]\"']", "", name).strip().strip(".")
    return f"{name or 'file'}{ext}"


def validate_extension(filename: str):
    if os.path.splitext(filename)[1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")


def save_upload(upload: UploadFile, dest_path: str):
    limit = MAX_UPLOAD_MB * (1 << 20)
    written = 0
    with open(dest_path, "wb") as buf:
        while chunk := upload.file.read(CHUNK):
            written += len(chunk)
            if written > limit:
                buf.close()
                os.remove(dest_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"{upload.filename} exceeds {MAX_UPLOAD_MB} MB limit",
                )
            buf.write(chunk)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_media(
    files: list[UploadFile] = File(...),
    censor_method: str = Form("silence"),
    debug: bool = Form(False),
    rescan: bool = Form(False),
):
    if censor_method not in ("silence", "bleep", "remove"):
        raise HTTPException(status_code=400, detail="Invalid censor_method")

    job_ids = []
    for upload in files:
        safe_name = sanitize_filename(upload.filename)
        validate_extension(safe_name)
        job_id = uuid.uuid4().hex[:12]
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}__{safe_name}")
        output_path = os.path.join(OUTPUT_DIR, f"censored_{safe_name}")

        save_upload(upload, input_path)

        record = {
            "id": job_id,
            "filename": safe_name,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "detail": "",
            "censor_method": censor_method,
            "debug": debug,
            "rescan": rescan,
            "input_path": input_path,
            "output_path": output_path,
        }
        manager.add(job_id, record)
        job_ids.append(job_id)

    return {"job_ids": job_ids, "message": f"{len(job_ids)} job(s) queued."}


@app.get("/jobs")
async def list_jobs():
    return [
        {k: v for k, v in job.items() if not k.endswith("_path")}
        for job in manager.snapshot()
    ]


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    for job in manager.snapshot():
        if job["id"] == job_id:
            return {k: v for k, v in job.items() if not k.endswith("_path")}
    raise HTTPException(status_code=404, detail="Job not found")


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    try:
        handled = manager.cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"cancelled": handled}


@app.get("/download/{job_id}")
async def download_censored_file(job_id: str):
    for job in manager.snapshot():
        if job["id"] == job_id:
            path = job["output_path"]
            if os.path.exists(path):
                return FileResponse(
                    path=path,
                    filename=os.path.basename(path),
                    media_type="application/octet-stream",
                )
            raise HTTPException(status_code=404, detail="Output not ready")
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/download_zip")
async def download_zip(ids: str):
    wanted = [i.strip() for i in ids.split(",") if i.strip()]
    paths = []
    with manager.lock:
        for jid in wanted:
            job = manager.jobs.get(jid)
            if job and job["status"] == "completed" and os.path.exists(job["output_path"]):
                paths.append(job["output_path"])
    if not paths:
        raise HTTPException(status_code=404, detail="No completed outputs to zip")

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
        for path in paths:
            zf.write(path, arcname=f"censored_{os.path.basename(path)}")
    return FileResponse(
        path=tmp.name,
        filename="foulfilter_results.zip",
        media_type="application/zip",
        background=BackgroundTask(_remove_file, tmp.name),
    )


def _remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


@app.get("/events")
async def events():
    import asyncio

    q = manager.subscribe()

    async def stream():
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    event = await asyncio.to_thread(q.get, timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except Exception:  # timeout -> heartbeat
                    yield ": keepalive\n\n"
        finally:
            manager.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/config")
async def config():
    return {
        "censor_methods": ["silence", "bleep", "remove"],
        "ai_enhance": (os.getenv("AI_ENHANCE") or "").lower() == "true"
        and bool(os.getenv("GOOGLE_API_KEY")),
        "whisper_model": os.getenv("WHISPER_MODEL") or "base",
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/metrics")
async def get_metrics():
    return metrics.get_metrics()


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
