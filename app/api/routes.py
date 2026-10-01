from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.schemas import (
    AnalyzeResponse,
    BatchDubbingResponse,
    DouyinRequest,
    DubbingResponse,
    PipelineConfig,
    RenderScriptRequest,
    ShortenTextRequest,
    ShortenTextResponse,
    TranscribeRequest,
)
from app.services.dependency_service import DependencyService
from app.services.pipeline import AutoDubbingPipeline
from app.services.pipeline_manager import PipelineManager
from app.services.render_job_service import RenderJobDispatcher, RenderJobSubmission
from app.services.voice_reference_service import (
    VOICE_REFERENCE_SECONDS,
    VoiceReferenceService,
    VoiceReferenceValidationError,
)
from app.utils.cancel import ACTIVE_OPERATIONS, PipelineCancelledError, schedule_process_termination
from app.utils.files import UploadSizeLimitError, safe_filename, save_upload_file, save_upload_file_limited
from app.utils.vram import VRAMManager
from app.utils.workspace import WorkspaceManager
from utils.model_registry import model_registry
from utils.stt import list_stt_models, remote_stt_enabled, transcribe_audio_remote
from utils.translation import list_translation_models, shorten_text_for_duration

router = APIRouter(prefix="/api/v1", tags=["dubbing"])
compat_router = APIRouter(prefix="/api", tags=["dubbing-compat"])
stream_router = APIRouter(prefix="/api", tags=["dubbing-stream"])
logger = logging.getLogger("auto_dubbing.routes")
STREAM_WORKER_JOIN_TIMEOUT_SECONDS = 5.0


class CancelOperationRequest(BaseModel):
    request_id: str | None = None
    hard: bool = True


REMOTE_ASR_PROVIDERS = {"9router", "remote", "openai-compatible", "openai_compatible", "gemini"}
VOICE_REFERENCE_SUFFIXES = {".wav"}
VOICE_REFERENCE_DIR = Path("output")
DEFAULT_VOICE_REFERENCE_MAX_BYTES = 16 * 1024 * 1024
SOURCE_MEDIA_DIR = Path("temp") / "source_media"
TranslationProvider = Literal["9router", "google", "mock"]
ComputeType = Literal["int8", "float16"]
AsrEngine = Literal["auto", "whisper", "paraformer"]
CopyrightSource = Literal["unknown", "owned", "licensed", "public_domain", "permission", "platform_library"]


def _first_config_value(value: str | None, env_names: tuple[str, ...], fallback: str) -> str:
    clean_value = (value or "").strip()
    if clean_value:
        return clean_value
    for env_name in env_names:
        env_file_value = _dotenv_value(env_name)
        if env_file_value:
            return env_file_value
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return env_value
    return fallback


def _dotenv_value(name: str) -> str | None:
    env_path = Path(".env")
    if not env_path.exists():
        return None
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                clean_value = value.strip().strip('"').strip("'")
                return clean_value or None
    except OSError:
        return None
    return None


def _fallback_asr_model(value: str | None) -> str:
    clean_value = (value or "").strip()
    if clean_value and clean_value.lower() != "auto":
        return clean_value


    provider = os.environ.get("AUTODUB_ASR_PROVIDER", "whisperx").strip().lower()
    if provider in REMOTE_ASR_PROVIDERS:
        return _first_config_value(None, ("AUTODUB_STT_MODEL",), "gemini/gemini-2.5-flash")
    configured = _first_config_value(None, ("AUTODUB_ASR_MODEL",), "base")
    return "base" if configured.lower() == "auto" else configured


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name) or _dotenv_value(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        raw = os.environ.get(name) or _dotenv_value(name) or str(default)
        return max(minimum, min(maximum, int(raw)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        raw = os.environ.get(name) or _dotenv_value(name) or str(default)
        return max(minimum, min(maximum, float(raw)))
    except (TypeError, ValueError):
        return default


def _media_name(media_path: str, field_name: str) -> str:
    raw = media_path.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        from urllib.parse import urlparse

        raw = urlparse(raw).path

    name = Path(raw).name
    if not name:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name}")
    return name


def _source_media_cache_path(media_path: str) -> Path:
    name = _media_name(media_path, "source_video_path")
    cache_root = SOURCE_MEDIA_DIR.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    path = (cache_root / name).resolve()
    if path.parent != cache_root:
        raise HTTPException(status_code=403, detail="source_video_path must point to cached source media")
    return path


def _cache_source_media(source_path: Path, media_path: str) -> None:
    try:
        destination = _source_media_cache_path(media_path)
        source_resolved = source_path.resolve()
        if source_resolved == destination:
            return
        shutil.copy2(source_path, destination)
        logger.info(
            "source_media.cached source=%s destination=%s size=%s",
            source_path,
            destination,
            destination.stat().st_size,
        )
    except Exception:
        logger.warning("source_media.cache_failed source=%s media_path=%s", source_path, media_path, exc_info=True)


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _config_from_form(
    source_language: str | None,
    target_language: str,
    translation_provider: str,
    translation_model: str,
    asr_model: str,
    compute_type: str,
    word_timestamps: bool,
    voice_model: str,
    tts_device: str,
    background_volume: float,
    tts_volume: float,
    burn_subtitles: bool,
    mock_translation: bool,
    mock_tts: bool,
    ocr_fallback: bool = True,
    ocr_force: bool = False,
    ocr_model: str = "gemini/gemini-2.5-flash",
    ocr_interval_seconds: float = 0.75,
    ocr_crop_bottom_ratio: float = 0.35,
    flash_text_enabled: bool = False,
    flash_text_mode: str = "balanced",
    flash_text_min_confidence: float = 0.58,
    flash_text_max_duration_s: float = 3.0,
    copyright_confirmed: bool = False,
    copyright_source: str = "unknown",
    copyright_notes: str = "",
    vocal_separation: bool = False,
    original_vocal_gain: float = 0.0,
    accompaniment_gain: float = 1.0,
    asr_engine: str | None = None,
    whisper_model: str | None = None,
    whisper_beam_size: int | None = None,
    segment_language_detection: bool | None = None,
    soft_timing_fit: bool | None = None,
    timing_max_drift_s: float | None = None,
    timing_min_gap_s: float | None = None,
    timing_max_atempo: float | None = None,
    voice_speed: float | None = None,
) -> PipelineConfig:
    translation_provider_value = _first_config_value(
        translation_provider,
        ("AUTODUB_TRANSLATION_PROVIDER",),
        "9router",
    )
    resolved_translation_provider = cast(
        TranslationProvider,
        translation_provider_value if translation_provider_value in {"9router", "google", "mock"} else "9router",
    )
    resolved_translation_model = _first_config_value(
        translation_model,
        ("AUTODUB_TRANSLATION_MODEL",),
        "ag/gemini-3-flash-agent",
    )
    resolved_asr_model = _fallback_asr_model(asr_model)
    compute_type_value = _first_config_value(compute_type, ("AUTODUB_COMPUTE_TYPE",), "float16")
    resolved_compute_type = cast(
        ComputeType,
        compute_type_value if compute_type_value in {"int8", "float16"} else "float16",
    )
    resolved_voice_model = _first_config_value(voice_model, ("AUTODUB_VOICE_MODEL",), "Trúc Ly")
    resolved_tts_device = "cuda"
    resolved_ocr_model = _first_config_value(ocr_model, ("AUTODUB_OCR_MODEL",), "gemini/gemini-2.5-flash")
    resolved_copyright_source = cast(
        CopyrightSource,
        copyright_source if copyright_source in {"unknown", "owned", "licensed", "public_domain", "permission", "platform_library"} else "unknown",
    )
    asr_engine_value = _first_config_value(asr_engine, ("ASR_ENGINE", "AUTODUB_ASR_ENGINE"), "auto").strip().lower()
    resolved_asr_engine = cast(
        AsrEngine,
        asr_engine_value if asr_engine_value in {"auto", "whisper", "paraformer"} else "auto",
    )

    return PipelineConfig(
        source_language=source_language,
        target_language=target_language,
        translation_provider=resolved_translation_provider,
        translation_model=resolved_translation_model,
        asr_model=resolved_asr_model,
        compute_type=resolved_compute_type,
        word_timestamps=word_timestamps,
        voice_model=resolved_voice_model,
        tts_device=resolved_tts_device,
        background_volume=background_volume,
        tts_volume=tts_volume,
        burn_subtitles=burn_subtitles,
        mock_translation=mock_translation,
        mock_tts=mock_tts,
        copyright_confirmed=copyright_confirmed,
        copyright_source=resolved_copyright_source,
        copyright_notes=copyright_notes,
        ocr_fallback=ocr_fallback,
        ocr_force=ocr_force,
        ocr_model=resolved_ocr_model,
        ocr_interval_seconds=ocr_interval_seconds,
        ocr_crop_bottom_ratio=ocr_crop_bottom_ratio,
        flash_text_enabled=flash_text_enabled,
        flash_text_mode=flash_text_mode if flash_text_mode in {"balanced", "strict"} else "balanced",
        flash_text_min_confidence=max(0.2, min(0.98, flash_text_min_confidence)),
        flash_text_max_duration_s=max(0.1, min(10.0, flash_text_max_duration_s)),
        vocal_separation=vocal_separation,
        original_vocal_gain=original_vocal_gain,
        accompaniment_gain=accompaniment_gain,
        asr_engine=resolved_asr_engine,
        whisper_model=_first_config_value(whisper_model, ("WHISPER_MODEL", "AUTODUB_WHISPER_MODEL"), "auto"),
        whisper_beam_size=max(1, min(10, int(whisper_beam_size if whisper_beam_size is not None else _env_int("WHISPER_BEAM_SIZE", _env_int("AUTODUB_WHISPER_BEAM_SIZE", 1, 1, 10), 1, 10)))),
        whisper_batch_size=_env_int("AUTODUB_WHISPER_BATCH_SIZE", 8, 1, 32),
        whisper_vad_filter=_env_bool("AUTODUB_WHISPER_VAD_FILTER", True),
        segment_language_detection=(segment_language_detection if segment_language_detection is not None else True),
        fill_speech_gaps=_env_bool("AUTODUB_FILL_SPEECH_GAPS", True),
        speech_gap_max_s=_env_float("AUTODUB_SPEECH_GAP_MAX_S", 8.0, 0, 30),
        default_source_language=_first_config_value(None, ("DEFAULT_SOURCE_LANG", "AUTODUB_DEFAULT_SOURCE_LANG"), "zh-CN"),
        ocr_adaptive=_env_bool("AUTODUB_OCR_ADAPTIVE", True),
        ocr_scene_threshold=_env_float("AUTODUB_OCR_SCENE_THRESHOLD", 0.28, 0.02, 1.0),
        translate_batch_size=_env_int("TRANSLATE_BATCH_SIZE", 24, 1, 100),
        translate_analysis=_env_bool("TRANSLATE_ANALYSIS", True),
        translate_review=_env_bool("TRANSLATE_REVIEW", True),
        translate_cps_budget=_env_float("TRANSLATE_CPS_BUDGET", 12.5, 1, 80),
        video_speed=_env_float("VIDEO_SPEED", 1.0, 0.25, 2.0),
        voice_speed=max(0.5, min(2.0, float(voice_speed if voice_speed is not None else _env_float("VOICE_SPEED", 1.0, 0.5, 2.0)))),
        soft_timing_fit=(soft_timing_fit if soft_timing_fit is not None else _env_bool("SOFT_TIMING_FIT", True)),
        timing_max_drift_s=max(0.0, min(10.0, float(timing_max_drift_s if timing_max_drift_s is not None else _env_float("TIMING_MAX_DRIFT_S", 1.5, 0, 10)))),
        timing_min_gap_s=max(0.0, min(2.0, float(timing_min_gap_s if timing_min_gap_s is not None else _env_float("TIMING_MIN_GAP_S", 0.12, 0, 2)))),
        timing_max_atempo=max(0.5, min(2.0, float(timing_max_atempo if timing_max_atempo is not None else _env_float("TIMING_MAX_ATEMPO", 1.1, 0.5, 2)))),
        hq_background=_env_bool("HQ_BACKGROUND", True),
        voice_postprocess=_env_bool("VOICE_POSTPROCESS", True),
        voice_target_lufs=_env_float("VOICE_TARGET_LUFS", -16.0, -40, -1),
        bg_duck_voice_db=_env_float("BG_DUCK_VOICE_DB", -7.0, -30, 0),
        checkpoint_enabled=_env_bool("AUTODUB_CHECKPOINT_ENABLED", True),
        checkpoint_root=_first_config_value(None, ("AUTODUB_CHECKPOINT_ROOT",), "temp/checkpoints"),
    )



def _require_copyright_preflight(config: PipelineConfig) -> None:
    if not config.copyright_confirmed:
        raise HTTPException(status_code=422, detail="Copyright preflight required: confirm you have rights to use this media before processing.")
    if config.copyright_source == "unknown":
        raise HTTPException(status_code=422, detail="Copyright preflight required: choose a clear rights source before processing.")

def _dubbing_response(result) -> DubbingResponse:
    return DubbingResponse(
        request_id=result.request_id,
        status="completed",
        output_video_path=str(result.output_video_path),
        subtitle_path=str(result.subtitle_path) if result.subtitle_path else None,
        segments_count=len(result.segments),
        caption_suggestions=result.caption_suggestions,
    )


def _batch_response(result) -> BatchDubbingResponse:
    return BatchDubbingResponse(
        request_id=result.request_id,
        status="partial" if result.failed else "completed",
        completed_count=len(result.results),
        failed_count=len(result.failed),
        output_video_paths=[str(item.output_video_path) for item in result.results],
        failures=result.failed,
    )



def _streaming_pipeline_response(request: Request, workspace_manager: WorkspaceManager, workspace, runner_factory) -> StreamingResponse:
    cancel_event = threading.Event()
    event_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    errors: list[BaseException] = []
    request_id = getattr(workspace, "request_id", "unknown")
    ACTIVE_OPERATIONS.register(request_id, cancel_event, "pipeline-stream")

    def worker() -> None:
        logger.info("stream.worker.start request_id=%s", request_id)
        try:
            for event in runner_factory(cancel_event):
                event_queue.put(("data", event))
        except PipelineCancelledError:
            logger.info("stream.worker.cancelled request_id=%s cause=client_disconnected_or_reload", request_id)
        except BaseException as exc:
            error_message = str(exc)
            if isinstance(exc, ValueError) and "Copyright preflight required" in error_message:
                logger.warning(
                    "stream.worker.failed request_id=%s cause=copyright_preflight_missing error=%s",
                    request_id,
                    error_message,
                )
            else:
                logger.exception("stream.worker.failed request_id=%s error=%s", request_id, exc)
            errors.append(exc)
        finally:
            event_queue.put(("done", None))
            ACTIVE_OPERATIONS.unregister(request_id)
            logger.info("stream.worker.done request_id=%s cancelled=%s", request_id, cancel_event.is_set())

    thread = threading.Thread(target=worker, daemon=True, name=f"autodub-stream-{request_id}")
    thread.start()

    def get_stream_event() -> tuple[str, str | None]:
        return event_queue.get(block=True, timeout=0.2)

    async def event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("stream.client_disconnected request_id=%s cause=browser_reload_or_closed_tab", request_id)
                    cancel_event.set()
                    break
                try:
                    kind, payload = await asyncio.to_thread(get_stream_event)
                except queue.Empty:
                    if not thread.is_alive() and event_queue.empty():
                        break
                    continue
                if kind == "data" and payload is not None:
                    yield payload
                else:
                    break
            if errors and not cancel_event.is_set():
                message = str(errors[0]) or errors[0].__class__.__name__
                yield "data: " + json.dumps(
                    {"step": "error", "status": "error", "phase": "error", "progress": 0, "error": message},
                    ensure_ascii=False,
                ) + "\n\n"
        finally:
            cancel_event.set()
            thread.join(timeout=STREAM_WORKER_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                logger.warning(
                    "stream.worker.still_running_after_cancel request_id=%s cause=blocking_external_call hint=wait_for_timeout_or_check_9router_ffmpeg_tts",
                    request_id,
                )
                def cleanup_after_worker() -> None:
                    thread.join()
                    workspace_manager.cleanup(workspace)
                    logger.info("stream.workspace.cleaned_after_worker request_id=%s", request_id)

                threading.Thread(
                    target=cleanup_after_worker,
                    daemon=True,
                    name=f"autodub-cleanup-{request_id}",
                ).start()
            else:
                workspace_manager.cleanup(workspace)
                logger.info("stream.workspace.cleaned request_id=%s", request_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Operation-ID": request_id,
        },
    )


def _render_job_streaming_response(
    request: Request,
    dispatcher: RenderJobDispatcher,
    submission: RenderJobSubmission,
) -> StreamingResponse:
    async def event_stream():
        index = 0
        while True:
            events = await asyncio.to_thread(
                dispatcher.store.read_events,
                submission.job_id,
                submission.attempt,
            )
            while index < len(events):
                event = events[index]
                index += 1
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

            manifest = await asyncio.to_thread(dispatcher.store.read_manifest, submission.job_id)
            terminal = manifest.get("status") in {"complete", "failed", "cancelled"}
            if terminal:
                latest_events = await asyncio.to_thread(
                    dispatcher.store.read_events,
                    submission.job_id,
                    submission.attempt,
                )
                if len(latest_events) > index:
                    continue
                break
            if await request.is_disconnected():
                logger.info(
                    "render_job.stream_disconnected job_id=%s attempt=%d worker_continues=true",
                    submission.job_id,
                    submission.attempt,
                )
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Render-Job-ID": submission.job_id,
        },
    )


@compat_router.post("/cancel")
async def cancel_current_operation(payload: CancelOperationRequest) -> dict[str, object]:
    """Cancel the stream started by this UI and optionally terminate its backend worker."""

    operation = (
        ACTIVE_OPERATIONS.cancel(payload.request_id)
        if payload.request_id
        else ACTIVE_OPERATIONS.cancel_only_active()
    )
    if operation is None:
        raise HTTPException(
            status_code=404,
            detail="No active pipeline matches this request. The stream may already be finished.",
        )
    if payload.hard:
        schedule_process_termination(
            reason=f"ui_cancel:{operation.operation_id}",
            pid=operation.pid,
        )
    return {
        "status": "cancelling",
        "request_id": operation.operation_id,
        "hard": payload.hard,
        "pid": operation.pid,
    }


@stream_router.post("/dub")
async def stream_dub_video(
    request: Request,
    video: UploadFile = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="en"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    asr_model: str = Form(default=""),
    compute_type: str = Form(default=""),
    word_timestamps: bool = Form(default=True),
    voice_model: str = Form(default=""),
    tts_device: str = Form(default=""),
    background_volume: float = Form(default=0.0),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=False),
    mock_tts: bool = Form(default=False),
    ocr_fallback: bool = Form(default=True),
    ocr_force: bool = Form(default=False),
    ocr_model: str = Form(default=""),
    ocr_interval_seconds: float = Form(default=0.75),
    ocr_crop_bottom_ratio: float = Form(default=0.35),
    flash_text_enabled: bool = Form(default=False),
    flash_text_mode: str = Form(default="balanced"),
    flash_text_min_confidence: float = Form(default=0.58),
    flash_text_max_duration_s: float = Form(default=3.0),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
    vocal_separation: bool = Form(default=False),
    original_vocal_gain: float = Form(default=0.0),
    accompaniment_gain: float = Form(default=1.0),
    asr_engine: str = Form(default="auto"),
    whisper_model: str = Form(default="auto"),
    whisper_beam_size: int = Form(default=1),
    segment_language_detection: bool = Form(default=True),
    soft_timing_fit: bool = Form(default=True),
    timing_max_drift_s: float = Form(default=1.5),
    timing_min_gap_s: float = Form(default=0.12),
    timing_max_atempo: float = Form(default=1.1),
    voice_speed: float = Form(default=1.0),
) -> StreamingResponse:
    """SSE endpoint for the Next.js client.

    The upload is persisted before streaming begins. The final output is written
    to `output/`, while the UUID workspace is cleaned after the stream ends.
    """

    config = _config_from_form(
        None if source_language == "auto" else source_language,
        target_language,
        translation_provider,
        translation_model,
        asr_model,
        compute_type,
        word_timestamps,
        voice_model,
        tts_device,
        background_volume,
        tts_volume,
        burn_subtitles,
        mock_translation,
        mock_tts,
        ocr_fallback,
        ocr_force,
        ocr_model,
        ocr_interval_seconds,
        ocr_crop_bottom_ratio,
        flash_text_enabled=flash_text_enabled,
        flash_text_mode=flash_text_mode,
        flash_text_min_confidence=flash_text_min_confidence,
        flash_text_max_duration_s=flash_text_max_duration_s,
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
        vocal_separation=vocal_separation,
        original_vocal_gain=original_vocal_gain,
        accompaniment_gain=accompaniment_gain,
        asr_engine=asr_engine,
        whisper_model=whisper_model,
        whisper_beam_size=whisper_beam_size,
        segment_language_detection=segment_language_detection,
        soft_timing_fit=soft_timing_fit,
        timing_max_drift_s=timing_max_drift_s,
        timing_min_gap_s=timing_min_gap_s,
        timing_max_atempo=timing_max_atempo,
        voice_speed=voice_speed,
    )
    _require_copyright_preflight(config)

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
    except Exception:
        workspace_manager.cleanup(workspace)
        raise

    return _streaming_pipeline_response(
        request,
        workspace_manager,
        workspace,
        lambda cancel_event: AutoDubbingPipeline(config, cancel_event=cancel_event).run(workspace),
    )

@stream_router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video_script(
    video: UploadFile = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="vi"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    asr_model: str = Form(default=""),
    compute_type: str = Form(default=""),
    voice_model: str = Form(default=""),
    tts_device: str = Form(default=""),
    ocr_fallback: bool = Form(default=True),
    ocr_force: bool = Form(default=False),
    ocr_model: str = Form(default=""),
    ocr_interval_seconds: float = Form(default=0.75),
    ocr_crop_bottom_ratio: float = Form(default=0.35),
    flash_text_enabled: bool = Form(default=False),
    flash_text_mode: str = Form(default="balanced"),
    flash_text_min_confidence: float = Form(default=0.58),
    flash_text_max_duration_s: float = Form(default=3.0),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
    vocal_separation: bool = Form(default=False),
    original_vocal_gain: float = Form(default=0.0),
    accompaniment_gain: float = Form(default=1.0),
    asr_engine: str = Form(default="auto"),
    whisper_model: str = Form(default="auto"),
    whisper_beam_size: int = Form(default=1),
    segment_language_detection: bool = Form(default=True),
    soft_timing_fit: bool = Form(default=True),
    timing_max_drift_s: float = Form(default=1.5),
    timing_min_gap_s: float = Form(default=0.12),
    timing_max_atempo: float = Form(default=1.1),
    voice_speed: float = Form(default=1.0),
) -> AnalyzeResponse:
    config = _config_from_form(
        None if source_language == "auto" else source_language,
        target_language,
        translation_provider,
        translation_model,
        asr_model,
        compute_type,
        True,
        voice_model,
        tts_device,
        0.0,
        1.0,
        True,
        False,
        True,
        ocr_fallback,
        ocr_force,
        ocr_model,
        ocr_interval_seconds,
        ocr_crop_bottom_ratio,
        flash_text_enabled=flash_text_enabled,
        flash_text_mode=flash_text_mode,
        flash_text_min_confidence=flash_text_min_confidence,
        flash_text_max_duration_s=flash_text_max_duration_s,
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
        vocal_separation=vocal_separation,
        original_vocal_gain=original_vocal_gain,
        accompaniment_gain=accompaniment_gain,
        asr_engine=asr_engine,
        whisper_model=whisper_model,
        whisper_beam_size=whisper_beam_size,
        segment_language_detection=segment_language_detection,
        soft_timing_fit=soft_timing_fit,
        timing_max_drift_s=timing_max_drift_s,
        timing_min_gap_s=timing_min_gap_s,
        timing_max_atempo=timing_max_atempo,
        voice_speed=voice_speed,
    )
    config.word_timestamps = True
    _require_copyright_preflight(config)

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
        preview_path = workspace.output_dir / f"{workspace.request_id}_source.mp4"
        shutil.copy2(workspace.input_video, preview_path)
        _cache_source_media(workspace.input_video, preview_path.name)
        pipeline = AutoDubbingPipeline(config)
        segments = pipeline.analyze(workspace)
        return AnalyzeResponse(
            request_id=workspace.request_id,
            status="completed",
            source_video_path=f"/media/{preview_path.name}",
            segments=segments,
            flash_text_tracks=pipeline.last_flash_text_tracks,
            caption_suggestions=pipeline.last_caption_suggestions,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}") from exc
    finally:
        workspace_manager.cleanup(workspace)




@stream_router.post("/analyze-stream")
async def analyze_video_script_stream(
    request: Request,
    video: UploadFile = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="vi"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    asr_model: str = Form(default=""),
    compute_type: str = Form(default=""),
    voice_model: str = Form(default=""),
    tts_device: str = Form(default=""),
    word_timestamps: bool = Form(default=True),
    mock_translation: bool = Form(default=False),
    ocr_fallback: bool = Form(default=True),
    ocr_force: bool = Form(default=False),
    ocr_model: str = Form(default=""),
    ocr_interval_seconds: float = Form(default=0.75),
    ocr_crop_bottom_ratio: float = Form(default=0.35),
    flash_text_enabled: bool = Form(default=False),
    flash_text_mode: str = Form(default="balanced"),
    flash_text_min_confidence: float = Form(default=0.58),
    flash_text_max_duration_s: float = Form(default=3.0),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
    vocal_separation: bool = Form(default=False),
    original_vocal_gain: float = Form(default=0.0),
    accompaniment_gain: float = Form(default=1.0),
    asr_engine: str = Form(default="auto"),
    whisper_model: str = Form(default="auto"),
    whisper_beam_size: int = Form(default=1),
    segment_language_detection: bool = Form(default=True),
    soft_timing_fit: bool = Form(default=True),
    timing_max_drift_s: float = Form(default=1.5),
    timing_min_gap_s: float = Form(default=0.12),
    timing_max_atempo: float = Form(default=1.1),
    voice_speed: float = Form(default=1.0),
) -> StreamingResponse:
    config = _config_from_form(
        None if source_language == "auto" else source_language,
        target_language,
        translation_provider,
        translation_model,
        asr_model,
        compute_type,
        word_timestamps,
        voice_model,
        tts_device,
        0.0,
        1.0,
        True,
        mock_translation,
        True,
        ocr_fallback,
        ocr_force,
        ocr_model,
        ocr_interval_seconds,
        ocr_crop_bottom_ratio,
        flash_text_enabled=flash_text_enabled,
        flash_text_mode=flash_text_mode,
        flash_text_min_confidence=flash_text_min_confidence,
        flash_text_max_duration_s=flash_text_max_duration_s,
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
        vocal_separation=vocal_separation,
        original_vocal_gain=original_vocal_gain,
        accompaniment_gain=accompaniment_gain,
        asr_engine=asr_engine,
        whisper_model=whisper_model,
        whisper_beam_size=whisper_beam_size,
        segment_language_detection=segment_language_detection,
        soft_timing_fit=soft_timing_fit,
        timing_max_drift_s=timing_max_drift_s,
        timing_min_gap_s=timing_min_gap_s,
        timing_max_atempo=timing_max_atempo,
        voice_speed=voice_speed,
    )
    config.word_timestamps = word_timestamps
    _require_copyright_preflight(config)

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
        preview_path = workspace.output_dir / f"{workspace.request_id}_source.mp4"
        shutil.copy2(workspace.input_video, preview_path)
        _cache_source_media(workspace.input_video, preview_path.name)
    except Exception:
        workspace_manager.cleanup(workspace)
        raise

    return _streaming_pipeline_response(
        request,
        workspace_manager,
        workspace,
        lambda cancel_event: AutoDubbingPipeline(config, cancel_event=cancel_event).analyze_stream(workspace),
    )


def _output_media_path(media_path: str, field_name: str = "source_video_path") -> Path:
    name = _media_name(media_path, field_name)
    output_root = Path("output").resolve()
    path = (output_root / name).resolve()
    if path.parent != output_root:
        raise HTTPException(status_code=403, detail=f"{field_name} must point to output media")
    if field_name == "source_video_path" and not _is_nonempty_file(path):
        cached_path = _source_media_cache_path(name)
        if _is_nonempty_file(cached_path):
            logger.info("source_media.cache_hit name=%s cached_path=%s", name, cached_path)
            return cached_path
        raise HTTPException(
            status_code=404,
            detail=f"Source video not found: {path}. Re-upload the source video to rerun render.",
        )
    return path


def _voice_reference_media_path(media_path: str) -> Path:
    path = _output_media_path(media_path, field_name="clone_reference_audio_path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Clone reference audio not found.")
    try:
        VoiceReferenceService().validate_canonical(path)
    except VoiceReferenceValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return path


def _render_config(payload: RenderScriptRequest, clone_reference_audio_path: Path | None) -> PipelineConfig:
    return PipelineConfig(
        source_language=payload.source_language,
        target_language=payload.target_language,
        translation_provider=payload.translation_provider,
        translation_model=payload.translation_model,
        voice_model=payload.voice_model.strip(),
        voice_mode=payload.voice_mode,
        clone_reference_audio_path=(
            str(clone_reference_audio_path) if clone_reference_audio_path is not None else None
        ),
        tts_device="cuda",
        background_volume=payload.background_volume,
        tts_volume=payload.tts_volume,
        burn_subtitles=payload.burn_subtitles,
        mock_translation=False,
        mock_tts=False,
        copyright_confirmed=payload.copyright_confirmed,
        copyright_source=payload.copyright_source,
        copyright_notes=payload.copyright_notes,
        vocal_separation=payload.vocal_separation,
        original_vocal_gain=payload.original_vocal_gain,
        accompaniment_gain=payload.accompaniment_gain,
        ocr_fallback=payload.ocr_fallback,
        ocr_force=payload.ocr_force,
        ocr_model=payload.ocr_model,
        ocr_interval_seconds=payload.ocr_interval_seconds,
        ocr_crop_bottom_ratio=payload.ocr_crop_bottom_ratio,
        flash_text_enabled=payload.flash_text_enabled,
        flash_text_mode=payload.flash_text_mode,
        flash_text_min_confidence=payload.flash_text_min_confidence,
        flash_text_max_duration_s=payload.flash_text_max_duration_s,
        asr_engine=payload.asr_engine,
        whisper_model=payload.whisper_model,
        whisper_beam_size=payload.whisper_beam_size,
        segment_language_detection=payload.segment_language_detection,
        translate_batch_size=payload.translate_batch_size,
        translate_analysis=payload.translate_analysis,
        translate_review=payload.translate_review,
        translate_cps_budget=payload.translate_cps_budget,
        video_speed=payload.video_speed,
        voice_speed=payload.voice_speed,
        soft_timing_fit=payload.soft_timing_fit,
        timing_max_drift_s=payload.timing_max_drift_s,
        timing_min_gap_s=payload.timing_min_gap_s,
        timing_max_atempo=payload.timing_max_atempo,
        hq_background=payload.hq_background,
        voice_postprocess=payload.voice_postprocess,
        voice_target_lufs=payload.voice_target_lufs,
        bg_duck_voice_db=payload.bg_duck_voice_db,
    )


def _shorten_text_response(request: ShortenTextRequest) -> ShortenTextResponse:
    try:
        shortened, max_words, provider, model = shorten_text_for_duration(
            request.text,
            target_duration=request.target_duration,
            target_language=request.target_language,
            source_text=request.source_text,
            context=request.context,
            provider=request.translation_provider,
            model=request.translation_model,
            max_words=request.max_words,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Shorten failed: {exc}") from exc

    return ShortenTextResponse(
        text=shortened,
        max_words=max_words,
        target_duration=request.target_duration,
        provider=provider,
        model=model,
    )


@stream_router.post("/shorten-text", response_model=ShortenTextResponse)
def shorten_text(request: ShortenTextRequest) -> ShortenTextResponse:
    return _shorten_text_response(request)


@stream_router.post("/voice-reference")
async def upload_voice_reference(
    audio: UploadFile = File(...),
    clip_duration_seconds: float = Form(default=VOICE_REFERENCE_SECONDS),
    selection_start_seconds: float | None = Form(default=None),
    selection_end_seconds: float | None = Form(default=None),
) -> dict[str, object]:
    if not math.isfinite(clip_duration_seconds) or not math.isclose(
        clip_duration_seconds,
        VOICE_REFERENCE_SECONDS,
        rel_tol=0,
        abs_tol=0.001,
    ):
        raise HTTPException(status_code=422, detail="Clone reference clip duration must be exactly 3 seconds.")
    if (selection_start_seconds is None) != (selection_end_seconds is None):
        raise HTTPException(status_code=422, detail="Both clone selection boundaries are required together.")
    if selection_start_seconds is not None and selection_end_seconds is not None:
        selection_duration = selection_end_seconds - selection_start_seconds
        if (
            not math.isfinite(selection_start_seconds)
            or not math.isfinite(selection_end_seconds)
            or selection_start_seconds < 0
            or not math.isclose(selection_duration, VOICE_REFERENCE_SECONDS, rel_tol=0, abs_tol=0.002)
        ):
            raise HTTPException(status_code=422, detail="Clone selection must identify one exact 3-second window.")

    safe_name = safe_filename(audio.filename, fallback_suffix=".wav")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in VOICE_REFERENCE_SUFFIXES:
        allowed = ", ".join(sorted(VOICE_REFERENCE_SUFFIXES))
        raise HTTPException(status_code=422, detail=f"Unsupported voice reference format. Use: {allowed}")
    if audio.content_type and not (
        audio.content_type.startswith("audio/") or audio.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=422, detail="The clone reference must be an audio file.")

    try:
        max_bytes = int(os.environ.get("AUTODUB_VOICE_REFERENCE_MAX_BYTES", DEFAULT_VOICE_REFERENCE_MAX_BYTES))
    except (TypeError, ValueError):
        max_bytes = DEFAULT_VOICE_REFERENCE_MAX_BYTES
    max_bytes = max(1, max_bytes)

    output_root = VOICE_REFERENCE_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{uuid4().hex}_voice_reference.wav"
    Path("temp").mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="voice_reference_", dir="temp") as upload_dir:
            staged_upload = Path(upload_dir) / f"selected_clip{suffix}"
            try:
                await save_upload_file_limited(audio, staged_upload, max_bytes=max_bytes)
            except UploadSizeLimitError as exc:
                raise HTTPException(status_code=413, detail=f"Clone reference upload is too large (max {max_bytes} bytes).") from exc

            metadata = await asyncio.to_thread(
                VoiceReferenceService().canonicalize,
                staged_upload,
                destination,
            )
    except VoiceReferenceValidationError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "path": f"/media/{destination.name}",
        "filename": safe_name,
        "duration_seconds": metadata.duration_seconds,
        "sample_rate": metadata.sample_rate,
        "frame_count": metadata.frame_count,
        "sha256": metadata.sha256,
        "selection_start_seconds": selection_start_seconds,
        "selection_end_seconds": selection_end_seconds,
    }


@stream_router.post("/render-script")
async def render_edited_script(request: Request, payload: RenderScriptRequest) -> StreamingResponse:
    clone_reference_audio_path: Path | None = None
    if payload.voice_mode == "clone":
        clone_reference_audio_path = _voice_reference_media_path(payload.clone_reference_audio_path or "")

    config = _render_config(payload, clone_reference_audio_path)
    _require_copyright_preflight(config)
    logger.info(
        "script_render.voice_setup_received mode=%s default_voice=%s segment_voices=%s timing={soft:%s,max_drift:%.3f,min_gap:%.3f,max_atempo:%.3f,speed:%.3f}",
        payload.voice_mode,
        config.voice_model,
        len({segment.voice_model.strip() for segment in payload.segments if segment.voice_model.strip()}),
        config.soft_timing_fit,
        config.timing_max_drift_s,
        config.timing_min_gap_s,
        config.timing_max_atempo,
        config.voice_speed,
    )
    source_video_path = _output_media_path(payload.source_video_path)
    dispatcher = RenderJobDispatcher()
    try:
        submission = await asyncio.to_thread(
            dispatcher.submit,
            payload,
            config,
            source_video_path,
            clone_reference_audio_path,
        )
    except Exception as exc:
        logger.exception("render_job.submit_failed source=%s", source_video_path)
        raise HTTPException(status_code=503, detail=f"Unable to queue render job: {exc}") from exc
    return _render_job_streaming_response(request, dispatcher, submission)


@stream_router.post("/render-script-upload")
async def render_edited_script_with_video(
    request: Request,
    video: UploadFile = File(...),
    payload: str = Form(...),
) -> StreamingResponse:
    try:
        render_payload = RenderScriptRequest.model_validate_json(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid render payload: {exc}") from exc

    clone_reference_audio_path: Path | None = None
    if render_payload.voice_mode == "clone":
        clone_reference_audio_path = _voice_reference_media_path(
            render_payload.clone_reference_audio_path or ""
        )

    config = _render_config(render_payload, clone_reference_audio_path)
    _require_copyright_preflight(config)
    logger.info(
        "script_render.voice_setup_received mode=%s default_voice=%s segment_voices=%s timing={soft:%s,max_drift:%.3f,min_gap:%.3f,max_atempo:%.3f,speed:%.3f}",
        render_payload.voice_mode,
        config.voice_model,
        len({segment.voice_model.strip() for segment in render_payload.segments if segment.voice_model.strip()}),
        config.soft_timing_fit,
        config.timing_max_drift_s,
        config.timing_min_gap_s,
        config.timing_max_atempo,
        config.voice_speed,
    )

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
        if not _is_nonempty_file(workspace.input_video):
            raise HTTPException(status_code=422, detail="Source video upload is empty.")
        _cache_source_media(workspace.input_video, render_payload.source_video_path)
        logger.info(
            "script_render.upload_fallback_received request_id=%s source_video_path=%s bytes=%s",
            workspace.request_id,
            render_payload.source_video_path,
            workspace.input_video.stat().st_size,
        )
        dispatcher = RenderJobDispatcher()
        submission = await asyncio.to_thread(
            dispatcher.submit,
            render_payload,
            config,
            workspace.input_video,
            clone_reference_audio_path,
        )
    except Exception as exc:
        workspace_manager.cleanup(workspace)
        if isinstance(exc, HTTPException):
            raise
        logger.exception("render_job.upload_submit_failed request_id=%s", workspace.request_id)
        raise HTTPException(status_code=503, detail=f"Unable to queue render job: {exc}") from exc

    workspace_manager.cleanup(workspace)
    return _render_job_streaming_response(request, dispatcher, submission)


@stream_router.get("/render-jobs/{job_id}")
async def render_job_status(job_id: str) -> dict[str, object]:
    try:
        UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid render job ID.") from exc
    dispatcher = RenderJobDispatcher()
    try:
        manifest = await asyncio.to_thread(dispatcher.store.read_manifest, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        auto_retry_attempts = max(1, int(os.environ.get("AUTODUB_RENDER_AUTO_ATTEMPTS", "3")))
    except ValueError:
        auto_retry_attempts = 3
    should_resume = manifest.get("status") in {"pending", "queued", "running"} or (
        manifest.get("status") == "failed"
        and int(manifest.get("attempt", 0) or 0) < auto_retry_attempts
    )
    if should_resume:
        try:
            submission = await asyncio.to_thread(dispatcher.resume, job_id)
            if submission.enqueued:
                logger.warning(
                    "render_job.orphan_requeued job_id=%s attempt=%d",
                    job_id,
                    submission.attempt,
                )
            manifest = await asyncio.to_thread(dispatcher.store.read_manifest, job_id)
        except Exception:
            logger.exception("render_job.recovery_failed job_id=%s", job_id)
            manifest = await asyncio.to_thread(dispatcher.store.read_manifest, job_id)
    output_available = dispatcher.store.output_path(job_id).is_file()
    if manifest.get("status") == "complete" and not output_available:
        manifest = {
            **manifest,
            "status": "failed",
            "phase": "error",
            "error": "Rendered output is missing; submit the same render again to resume.",
        }
    return {**manifest, "output_available": output_available}


@stream_router.post("/render-jobs/{job_id}/cancel")
async def cancel_render_job(job_id: str) -> dict[str, object]:
    try:
        UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid render job ID.") from exc

    dispatcher = RenderJobDispatcher()
    try:
        manifest = await asyncio.to_thread(dispatcher.store.read_manifest, job_id)
        await asyncio.to_thread(dispatcher.cancel, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    worker_pid = int(manifest.get("worker_pid", 0) or manifest.get("owner_pid", 0) or 0)
    if worker_pid == os.getpid() and manifest.get("status") in {"queued", "running"}:
        schedule_process_termination(reason=f"ui_cancel_render:{job_id}", pid=worker_pid)
    return {
        "status": "cancelling",
        "job_id": job_id,
        "hard": worker_pid == os.getpid(),
        "worker_pid": worker_pid,
    }


@router.post("/shorten-text", response_model=ShortenTextResponse)
def shorten_text_v1(request: ShortenTextRequest) -> ShortenTextResponse:
    return _shorten_text_response(request)


@router.post("/dub", response_model=DubbingResponse)
async def dub_video(
    file: UploadFile = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="en"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    asr_model: str = Form(default=""),
    compute_type: str = Form(default=""),
    word_timestamps: bool = Form(default=True),
    voice_model: str = Form(default=""),
    tts_device: str = Form(default=""),
    background_volume: float = Form(default=0.0),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=False),
    mock_tts: bool = Form(default=False),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
    vocal_separation: bool = Form(default=False),
) -> DubbingResponse:
    try:
        config = _config_from_form(
            source_language,
            target_language,
            translation_provider,
            translation_model,
            asr_model,
            compute_type,
            word_timestamps,
            voice_model,
            tts_device,
            background_volume,
            tts_volume,
            burn_subtitles,
            mock_translation,
            mock_tts,
            copyright_confirmed=copyright_confirmed,
            copyright_source=copyright_source,
            copyright_notes=copyright_notes,
            vocal_separation=vocal_separation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _require_copyright_preflight(config)
    suffix = Path(safe_filename(file.filename)).suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="dub_upload_") as upload_dir:
        input_path = Path(upload_dir) / f"input{suffix}"
        await save_upload_file(file, input_path)

        try:
            result = PipelineManager().process(input_path, config)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return _dubbing_response(result)


@router.post("/dub-with-srt", response_model=DubbingResponse)
async def dub_video_with_srt(
    video: UploadFile = File(...),
    subtitle: UploadFile = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="en"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    asr_model: str = Form(default=""),
    compute_type: str = Form(default=""),
    voice_model: str = Form(default=""),
    tts_device: str = Form(default=""),
    background_volume: float = Form(default=0.0),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=False),
    mock_tts: bool = Form(default=False),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
    vocal_separation: bool = Form(default=False),
) -> DubbingResponse:
    try:
        config = _config_from_form(
            source_language,
            target_language,
            translation_provider,
            translation_model,
            asr_model,
            compute_type,
            True,
            voice_model,
            tts_device,
            background_volume,
            tts_volume,
            burn_subtitles,
            mock_translation,
            mock_tts,
            copyright_confirmed=copyright_confirmed,
            copyright_source=copyright_source,
            copyright_notes=copyright_notes,
            vocal_separation=vocal_separation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _require_copyright_preflight(config)
    video_suffix = Path(safe_filename(video.filename)).suffix or ".mp4"
    subtitle_suffix = Path(safe_filename(subtitle.filename)).suffix or ".srt"
    with tempfile.TemporaryDirectory(prefix="dub_srt_upload_") as upload_dir:
        video_path = Path(upload_dir) / f"input{video_suffix}"
        subtitle_path = Path(upload_dir) / f"input{subtitle_suffix}"
        await save_upload_file(video, video_path)
        await save_upload_file(subtitle, subtitle_path)

        try:
            result = PipelineManager().process_with_srt(video_path, subtitle_path, config)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SRT pipeline failed: {exc}") from exc

    return _dubbing_response(result)


@router.post("/batch-dub", response_model=BatchDubbingResponse)
async def batch_dub_videos(
    files: list[UploadFile] = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="en"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    asr_model: str = Form(default=""),
    compute_type: str = Form(default=""),
    word_timestamps: bool = Form(default=True),
    voice_model: str = Form(default=""),
    tts_device: str = Form(default=""),
    background_volume: float = Form(default=0.0),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=False),
    mock_tts: bool = Form(default=False),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
    vocal_separation: bool = Form(default=False),
) -> BatchDubbingResponse:
    try:
        config = _config_from_form(
            source_language,
            target_language,
            translation_provider,
            translation_model,
            asr_model,
            compute_type,
            word_timestamps,
            voice_model,
            tts_device,
            background_volume,
            tts_volume,
            burn_subtitles,
            mock_translation,
            mock_tts,
            copyright_confirmed=copyright_confirmed,
            copyright_source=copyright_source,
            copyright_notes=copyright_notes,
            vocal_separation=vocal_separation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _require_copyright_preflight(config)
    with tempfile.TemporaryDirectory(prefix="dub_batch_upload_") as upload_dir:
        video_paths: list[Path] = []
        for index, upload in enumerate(files):
            suffix = Path(safe_filename(upload.filename)).suffix or ".mp4"
            input_path = Path(upload_dir) / f"input_{index:04d}{suffix}"
            await save_upload_file(upload, input_path)
            video_paths.append(input_path)

        result = PipelineManager().process_batch(video_paths, config)

    return _batch_response(result)


@router.post("/douyin", response_model=BatchDubbingResponse)
def process_douyin(request: DouyinRequest) -> BatchDubbingResponse:
    try:
        max_items = 1 if request.mode == "single" else request.max_items
        result = PipelineManager().process_douyin(
            url=request.url,
            config=request.config,
            max_items=max_items,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Douyin pipeline failed: {exc}") from exc

    return _batch_response(result)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}



@router.get("/health/dependencies")
def health_dependencies() -> dict[str, bool | str | None]:
    return DependencyService().status()


@router.get("/stt/models")
def stt_models_v1() -> dict[str, object]:
    return list_stt_models(timeout=0.5)


@router.get("/translation/models")
def translation_models_v1() -> dict[str, object]:
    return list_translation_models(timeout=0.5)


@stream_router.get("/translation/models")
def translation_models() -> dict[str, object]:
    return list_translation_models(timeout=0.5)


@stream_router.get("/stt/models")
def stt_models() -> dict[str, object]:
    return list_stt_models(timeout=0.5)


def _pipeline_settings_payload() -> dict[str, object]:
    """UI contract for ASR/OCR/timing controls shared by both workflows."""
    return {
        "version": "pipeline-settings-v1",
        "source_languages": [
            {"value": "auto", "label": "Tự động nhận dạng"},
            {"value": "zh", "label": "Tiếng Trung"},
            {"value": "en", "label": "Tiếng Anh"},
            {"value": "vi", "label": "Tiếng Việt"},
            {"value": "ja", "label": "Tiếng Nhật"},
            {"value": "ko", "label": "Tiếng Hàn"},
        ],
        "asr_engines": ["auto", "whisper", "paraformer"],
        "whisper_models": ["auto", "tiny", "base", "small", "medium", "large-v3"],
        "ocr": {
            "enabled_by_default": False,
            "interval_seconds": 0.75,
            "crop_bottom_ratio": 0.35,
            "models": [
                {"id": "gemini/gemini-2.5-flash", "label": "Gemini 2.5 Flash OCR"},
                {"id": "gemini/gemini-2.5-pro", "label": "Gemini 2.5 Pro OCR"},
            ],
        },
        "timing": {
            "soft_timing_fit": True,
            "max_drift_s": 1.5,
            "min_gap_s": 0.12,
            "max_atempo": 1.1,
            "voice_speed": 1.0,
        },
        "speech_gap_repair": {
            "enabled": _env_bool("AUTODUB_FILL_SPEECH_GAPS", True),
            "max_gap_s": _env_float("AUTODUB_SPEECH_GAP_MAX_S", 8.0, 0, 30),
        },
        "vocal_separation": {
            "enabled_by_default": False,
            "device": os.environ.get("AUTODUB_DEMUCS_DEVICE", "auto").strip().lower() or "auto",
            "chunk_seconds": _env_float("AUTODUB_DEMUCS_CHUNK_SECONDS", 60.0, 10.0, 600.0),
        },
    }


@router.get("/pipeline/settings")
def pipeline_settings_v1() -> dict[str, object]:
    return _pipeline_settings_payload()


@stream_router.get("/pipeline/settings")
def pipeline_settings() -> dict[str, object]:
    return _pipeline_settings_payload()


@compat_router.post("/upload-and-extract")
async def upload_and_extract(video: UploadFile = File(...)) -> dict[str, str]:
    """Persist an uploaded video and extract a reviewable audio track."""

    request_uuid = uuid4().hex
    workspace = Path("temp") / request_uuid
    video_path = workspace / "original.mp4"
    audio_path = workspace / "audio.wav"
    workspace.mkdir(parents=True, exist_ok=False)

    try:
        with video_path.open("wb") as buffer:
            while chunk := await video.read(1024 * 1024):
                buffer.write(chunk)

        try:
            import ffmpeg
        except ImportError as exc:
            raise RuntimeError("ffmpeg-python is required. Install it with `pip install ffmpeg-python`.") from exc

        (
            ffmpeg.input(str(video_path))
            .output(str(audio_path), ac=1, ar="16000", vn=None, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )

        return {
            "uuid": request_uuid,
            "status": "audio_extracted",
            "audio_url": f"/temp/{request_uuid}/audio.wav",
            "video_url": f"/temp/{request_uuid}/original.mp4",
        }
    except Exception as exc:
        logger.exception("upload_extract.error uuid=%s workspace=%s", request_uuid, workspace)
        shutil.rmtree(workspace, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload/audio extraction failed: {exc}") from exc


@compat_router.post("/transcribe")
def transcribe(request: TranscribeRequest) -> list[dict[str, object]]:
    """Transcribe audio extracted by `/api/upload-and-extract` for HITL review."""

    workspace = _workspace_from_uuid(request.uuid)
    audio_path = workspace / "audio.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="audio.wav not found for this workspace")

    try:
        if remote_stt_enabled(""):
            return _normalize_transcript_segments(
                transcribe_audio_remote(
                    audio_path,
                    source_language=None,
                    model=None,
                )
            )

        try:
            import torch
            import whisperx
        except ImportError as exc:
            raise RuntimeError("torch and whisperx are required for transcription.") from exc

        DependencyService().require_cuda()
        device = "cuda"
        audio = whisperx.load_audio(str(audio_path))
        with model_registry.acquire_whisperx_asr(
            whisperx,
            whisper_arch="base",
            device=device,
            compute_type=_first_config_value(None, ("AUTODUB_COMPUTE_TYPE",), "float16"),
            language=None,
            beam_size=_env_int("AUTODUB_WHISPER_BEAM_SIZE", 1, 1, 10),
        ) as model:
            result = model.transcribe(
                audio,
                batch_size=_env_int("AUTODUB_WHISPER_BATCH_SIZE", 8, 1, 32),
                language=None,
                vad_filter=_env_bool("AUTODUB_WHISPER_VAD_FILTER", True),
                no_speech_threshold=0.4,
                condition_on_previous_text=False,
            )

        VRAMManager.cleanup()
        language_code = result.get("language") or "en"
        with model_registry.acquire_whisperx_align(
            whisperx,
            language_code=language_code,
            device=device,
        ) as (align_model, metadata):
            aligned = whisperx.align(
                result.get("segments", []),
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
        return _normalize_transcript_segments(aligned.get("segments", []))
    except HTTPException:
        raise
    except Exception as exc:
        if VRAMManager.is_cuda_error(exc):
            VRAMManager.reset_after_cuda_error()
            raise HTTPException(
                status_code=507,
                detail="CUDA failed during transcription. Model cache was reset and VRAM cache was cleared.",
            ) from exc
        logger.exception("transcribe.error uuid=%s", request.uuid)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        VRAMManager.cleanup()


@compat_router.post("/process-video")
async def process_video_stream(
    request: Request,
    video: UploadFile = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="en"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    word_timestamps: bool = Form(default=True),
    voice_model: str = Form(default=""),
    tts_device: str = Form(default=""),
    background_volume: float = Form(default=0.0),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=False),
    mock_tts: bool = Form(default=False),
    ocr_fallback: bool = Form(default=True),
    ocr_force: bool = Form(default=False),
    ocr_model: str = Form(default=""),
    ocr_interval_seconds: float = Form(default=0.75),
    ocr_crop_bottom_ratio: float = Form(default=0.35),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
) -> StreamingResponse:
    """Compatibility endpoint for browser direct fetch streaming.

    The frontend posts directly here instead of going through Next.js. The core
    pipeline remains synchronous and low-VRAM; this endpoint only wraps it in an
    SSE response so clients can keep the connection alive.
    """

    config = PipelineConfig(
        source_language=None if source_language == "auto" else source_language,
        target_language=target_language,
        translation_provider=translation_provider,
        translation_model=translation_model,
        word_timestamps=word_timestamps,
        voice_model=voice_model,
        tts_device="cuda",
        background_volume=background_volume,
        tts_volume=tts_volume,
        burn_subtitles=burn_subtitles,
        mock_translation=mock_translation,
        mock_tts=False,
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
        ocr_fallback=ocr_fallback,
        ocr_force=ocr_force,
        ocr_model=ocr_model,
        ocr_interval_seconds=ocr_interval_seconds,
        ocr_crop_bottom_ratio=ocr_crop_bottom_ratio,
    )
    _require_copyright_preflight(config)

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
    except Exception:
        workspace_manager.cleanup(workspace)
        raise

    return _streaming_pipeline_response(
        request,
        workspace_manager,
        workspace,
        lambda cancel_event: AutoDubbingPipeline(config, cancel_event=cancel_event).run(workspace),
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _workspace_from_uuid(raw_uuid: str) -> Path:
    try:
        parsed_uuid = UUID(raw_uuid)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid uuid") from exc

    temp_root = Path("temp").resolve()
    workspace = (temp_root / parsed_uuid.hex).resolve()
    if workspace.parent != temp_root:
        raise HTTPException(status_code=403, detail="Invalid workspace path")
    if not workspace.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


def _normalize_transcript_segments(raw_segments: list[dict]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for index, raw in enumerate(raw_segments, start=1):
        text = str(raw.get("text", "")).strip()
        if not text:
            continue

        start = float(raw.get("start", 0.0) or 0.0)
        end = float(raw.get("end", start) or start)
        words = [
            {
                "word": str(word.get("word", "")).strip(),
                "start": float(word.get("start", start) or start),
                "end": float(word.get("end", end) or end),
            }
            for word in raw.get("words", [])
            if str(word.get("word", "")).strip()
        ]

        segments.append(
            {
                "id": index,
                "start": start,
                "end": max(start, end),
                "text": text,
                "words": words,
            }
        )
    return segments
