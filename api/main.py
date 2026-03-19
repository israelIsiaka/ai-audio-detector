"""
api/main.py — FastAPI REST API for AI Audio Detector.

Start server:
    uvicorn api.main:app --reload --port 8000
    # accessible from other devices on your network:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                  — welcome
    GET  /health            — liveness + model status
    GET  /model/info        — model metadata
    POST /predict           — single file inference
    POST /predict/batch     — multiple files inference
    GET  /jobs/{job_id}     — poll async job status
"""

import asyncio
import datetime
import io
import logging
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Add project src/ to path so pipeline.py + feature_extractor.py are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline import AudioDetector  # type: ignore

# ---------------------------------------------------------------------------
# Logging — console + daily rotating file (logs/YYYY-MM-DD.log, append mode)
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def _get_file_handler() -> logging.FileHandler:
    """Return a FileHandler for today's log file (append mode)."""
    today = datetime.date.today().isoformat()
    log_path = LOG_DIR / f"{today}.log"
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(_log_formatter)
    handler.setLevel(logging.INFO)
    return handler

def _setup_logging() -> None:
    console = logging.StreamHandler()
    console.setFormatter(_log_formatter)
    console.setLevel(logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(_get_file_handler())

_setup_logging()
log = logging.getLogger("ai-audio-api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "detector.pkl"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_SIZE_MB = 50
MAX_BATCH_FILES = 20

# ---------------------------------------------------------------------------
# In-memory job store  (replace with Redis for production)
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# App lifespan — load model once at startup
# ---------------------------------------------------------------------------
_detector: AudioDetector | None = None


def _get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unavailable"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    log.info("Loading model from %s …", MODEL_PATH)
    try:
        _detector = AudioDetector(model_path=MODEL_PATH)
        log.info("Model loaded successfully.")
    except FileNotFoundError:
        log.warning("Model not found at '%s'. /predict will return 503 until model is trained.", MODEL_PATH)
        _detector = None

    import sys
    local_ip = _get_local_ip()
    today    = datetime.date.today().isoformat()
    sys.stdout.write("\n" + "=" * 45 + "\n")
    sys.stdout.write("  AI Audio Detector is running\n")
    sys.stdout.write("=" * 45 + "\n")
    sys.stdout.write(f"  Local:    http://localhost:8000\n")
    sys.stdout.write(f"  Network:  http://{local_ip}:8000\n")
    sys.stdout.write(f"  Log:      logs/{today}.log\n")
    sys.stdout.write("=" * 45 + "\n\n")
    sys.stdout.flush()
    log.info("Server started — local=http://localhost:8000  network=http://%s:8000", local_ip)

    yield
    log.info("Shutting down.")


app = FastAPI(
    title="AI Audio Detector API",
    description="Predict whether an audio file is natural speech or AI-generated.",
    version="0.1.0",
    lifespan=lifespan,
)

STATIC_DIR = PROJECT_ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Middleware — log every request and response
# ---------------------------------------------------------------------------
_current_log_date = datetime.date.today()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global _current_log_date

    # Rotate log file at midnight without restarting
    today = datetime.date.today()
    if today != _current_log_date:
        _current_log_date = today
        root = logging.getLogger()
        # Remove old file handlers and attach today's
        root.handlers = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]
        root.addHandler(_get_file_handler())
        log.info("Log rotated — new file: logs/%s.log", today.isoformat())

    client = request.client.host if request.client else "unknown"
    start  = time.time()

    # Log incoming request
    log.info("→ %s %s  client=%s", request.method, request.url.path, client)

    # Log query params if present
    if request.query_params:
        log.info("  params=%s", dict(request.query_params))

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.time() - start) * 1000, 1)
        log.error("✗ %s %s  ERROR=%s  (%sms)", request.method, request.url.path, exc, duration_ms)
        raise

    duration_ms = round((time.time() - start) * 1000, 1)
    status_icon = "✓" if response.status_code < 400 else "✗"
    log.info(
        "%s %s %s  status=%d  time=%sms  client=%s",
        status_icon, request.method, request.url.path,
        response.status_code, duration_ms, client,
    )
    return response


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_path: str
    feature_count: int
    feature_names: list[str]
    metadata: dict[str, Any]


class PredictionResponse(BaseModel):
    file: str
    label: str = Field(..., examples=["natural", "ai_generated"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    features_used: int
    inference_ms: float


class BatchPredictionResponse(BaseModel):
    total: int
    completed: int
    results: list[dict[str, Any]]
    summary: dict[str, Any]


class JobResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    created_at: float
    completed_at: float | None = None
    result: BatchPredictionResponse | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_start_time = time.time()


def _require_model() -> AudioDetector:
    if _detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run `python src/model.py` to train and save a model first.",
        )
    return _detector


def _validate_upload(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )


async def _save_upload_to_tmp(file: UploadFile) -> Path:
    """Stream upload in chunks to a temp file, enforcing size limit early."""
    suffix = Path(file.filename or "audio").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    size = 0
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    try:
        while True:
            chunk = await file.read(256 * 1024)  # 256 KB chunks
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large ({size / (1024*1024):.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB.",
                )
            tmp.write(chunk)
    finally:
        tmp.close()
    return Path(tmp.name)


def _batch_summary(results: list[dict]) -> dict:
    good = [r for r in results if "label" in r]
    errors = [r for r in results if "error" in r]
    ai = sum(1 for r in good if r["label"] == "ai_generated")
    nat = len(good) - ai
    avg_conf = round(sum(r["confidence"] for r in good) / len(good), 4) if good else None
    return {
        "natural": nat,
        "ai_generated": ai,
        "errors": len(errors),
        "avg_confidence": avg_conf,
    }


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def _run_batch_job(job_id: str, tmp_paths: list[Path]) -> None:
    _jobs[job_id]["status"] = "running"
    detector = _require_model()
    results = []

    for tmp_path in tmp_paths:
        try:
            # Run blocking I/O in thread pool so we don't block the event loop
            result = await asyncio.get_event_loop().run_in_executor(
                None, detector.predict, tmp_path
            )
            # Replace temp path with original filename stored in metadata
            result["file"] = _jobs[job_id]["filenames"][tmp_paths.index(tmp_path)]
            results.append(result)
        except Exception as exc:
            results.append({
                "file": _jobs[job_id]["filenames"][tmp_paths.index(tmp_path)],
                "error": str(exc),
            })
        finally:
            tmp_path.unlink(missing_ok=True)

    summary = _batch_summary(results)
    _jobs[job_id].update(
        status="done",
        completed_at=time.time(),
        result=BatchPredictionResponse(
            total=len(results),
            completed=sum(1 for r in results if "label" in r),
            results=results,
            summary=summary,
        ).model_dump(),
    )
    log.info("Job %s done — %d files, summary=%s", job_id, len(results), summary)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(
        status="ok" if _detector is not None else "degraded",
        model_loaded=_detector is not None,
        model_path=str(MODEL_PATH),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
async def model_info():
    detector = _require_model()
    return ModelInfoResponse(
        model_path=str(MODEL_PATH),
        feature_count=len(detector.feature_names) or getattr(detector.model, "n_features_in_", -1),
        feature_names=detector.feature_names,
        metadata=detector.metadata,
    )


# ---------------------------------------------------------------------------
# POST /predict  — synchronous single-file inference
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
    summary="Predict a single audio file",
)
async def predict(file: UploadFile = File(..., description="Audio file (.wav, .mp3, .flac, .ogg)")):
    _validate_upload(file)
    detector = _require_model()
    tmp_path = await _save_upload_to_tmp(file)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, detector.predict, tmp_path
        )
        result["file"] = file.filename
        log.info(
            "PREDICTION  file=%s  label=%s  confidence=%.2f%%  time=%.1fms",
            file.filename, result["label"],
            result["confidence"] * 100, result["inference_ms"],
        )
        return PredictionResponse(**result)
    except Exception as exc:
        log.exception("Prediction failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# POST /predict/batch  — async multi-file inference via background task
# ---------------------------------------------------------------------------

@app.post(
    "/predict/batch",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["inference"],
    summary="Predict multiple audio files (async)",
)
async def predict_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Up to 20 audio files"),
):
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Too many files. Max batch size: {MAX_BATCH_FILES}.",
        )

    _require_model()  # fail fast before saving anything

    # Validate + persist all uploads
    tmp_paths: list[Path] = []
    filenames: list[str] = []
    for f in files:
        _validate_upload(f)
        tmp_paths.append(await _save_upload_to_tmp(f))
        filenames.append(f.filename or f"file_{len(filenames)}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": time.time(),
        "completed_at": None,
        "filenames": filenames,
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_run_batch_job, job_id, tmp_paths)
    log.info("Batch job %s queued — %d files", job_id, len(files))

    return JobResponse(**_jobs[job_id])


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}  — poll async job
# ---------------------------------------------------------------------------

@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    tags=["inference"],
    summary="Poll batch job status",
)
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse(**job)


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)