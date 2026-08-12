from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging
import os
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.genvideo_routes import router as genvideo_router
from app.api.genvideo_flow_routes import router as genvideo_flow_router
from app.api.routes import compat_router, router as dubbing_router, stream_router
from app.api.short_video_routes import router as short_video_router
from app.utils.workspace import WorkspaceManager
from utils.model_cache import configure_model_cache
from utils.model_registry import model_registry
from utils.ninerouter import ensure_9router_running

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("auto_dubbing.main")
MODEL_CACHE_PATHS = configure_model_cache()
logger.info(
    "startup.model_cache root=%s hf_home=%s hf_hub_cache=%s whisperx_asr_cache=%s whisperx_align_cache=%s",
    MODEL_CACHE_PATHS.root,
    MODEL_CACHE_PATHS.hf_home,
    MODEL_CACHE_PATHS.hf_hub_cache,
    MODEL_CACHE_PATHS.whisperx_asr_cache,
    MODEL_CACHE_PATHS.whisperx_align_cache,
)
DEV_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3002",
    "http://172.16.0.2:3000",
    "http://172.16.0.2:3001",
    "http://172.16.0.2:3002",
]
DEV_CORS_REGEX = r"^http://(localhost|127\.0\.0\.1|172\.16\.\d+\.\d+|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+):300[0-9]$"

Path("output").mkdir(parents=True, exist_ok=True)
Path("temp").mkdir(parents=True, exist_ok=True)


def _hours_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        try:
            for line in Path(".env").read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(f"{name}="):
                    raw = line.split("=", 1)[1].strip()
                    break
        except OSError:
            pass
    try:
        return max(1.0 / 60.0, float(raw)) if raw else default
    except ValueError:
        return default


async def _temp_cleanup_loop() -> None:
    retention_hours = _hours_env("AUTODUB_TEMP_CLEANUP_MAX_AGE_HOURS", 24.0)
    interval_hours = _hours_env("AUTODUB_TEMP_CLEANUP_INTERVAL_HOURS", 24.0)
    output_retention_hours = _hours_env("AUTODUB_OUTPUT_CLEANUP_MAX_AGE_HOURS", 12.0)
    output_interval_hours = _hours_env("AUTODUB_OUTPUT_CLEANUP_INTERVAL_HOURS", 1.0)
    retention_seconds = retention_hours * 3600.0
    output_retention_seconds = output_retention_hours * 3600.0
    sweep_interval_seconds = min(interval_hours, output_interval_hours) * 3600.0
    while True:
        try:
            await asyncio.to_thread(
                WorkspaceManager.evict_stale_all,
                retention_seconds,
                output_retention_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("temp.cleanup.failed")
        await asyncio.sleep(sweep_interval_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cleanup_task = asyncio.create_task(_temp_cleanup_loop(), name="temp-workspace-cleanup")
    logger.info("startup.ninerouter.begin")
    if ensure_9router_running():
        logger.info("startup.ninerouter.ready")
    else:
        logger.warning("startup.ninerouter.unreachable_or_failed")
    logger.info("startup.model_registry.begin")
    model_registry.startup()
    logger.info("startup.model_registry.ready stats=%s", model_registry.stats())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("shutdown.model_registry.begin")
        model_registry.shutdown()


app = FastAPI(
    title="Auto-Dubbing Video API",
    version="1.0.0",
    description="Low-VRAM auto-dubbing backend with canonical app package routes.",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid4().hex[:12])
    started_at = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    origin = request.headers.get("origin", "-")
    logger.info(
        "request.start id=%s method=%s path=%s client=%s origin=%s",
        request_id,
        request.method,
        request.url.path,
        client,
        origin,
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request.error id=%s method=%s path=%s elapsed_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            (time.perf_counter() - started_at) * 1000,
        )
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request.end id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started_at) * 1000,
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_CORS_ORIGINS,
    allow_origin_regex=DEV_CORS_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Render-Job-ID"],
)

app.include_router(dubbing_router)
app.include_router(compat_router)
app.include_router(stream_router)
app.include_router(short_video_router)
app.include_router(genvideo_flow_router)
app.include_router(genvideo_router)
app.mount("/media", StaticFiles(directory="output"), name="media")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "auto-dubbing-api", "status": "running"}
