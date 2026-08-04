from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

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
from app.utils.cancel import PipelineCancelledError
from app.utils.files import safe_filename, save_upload_file
from app.utils.vram import VRAMManager
from app.utils.workspace import WorkspaceManager
from utils.model_registry import model_registry
from utils.stt import list_stt_models, remote_stt_enabled, transcribe_audio_remote
from utils.translation import list_translation_models, shorten_text_for_duration

router = APIRouter(prefix="/api/v1", tags=["dubbing"])
compat_router = APIRouter(prefix="/api", tags=["dubbing-compat"])
stream_router = APIRouter(prefix="/api", tags=["dubbing-stream"])
logger = logging.getLogger("auto_dubbing.routes")


REMOTE_ASR_PROVIDERS = {"9router", "remote", "openai-compatible", "openai_compatible", "gemini"}
VOICE_REFERENCE_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


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
    if clean_value:
        return clean_value
    provider = os.environ.get("AUTODUB_ASR_PROVIDER", "whisperx").strip().lower()
    if provider in REMOTE_ASR_PROVIDERS:
        return _first_config_value(None, ("AUTODUB_STT_MODEL",), "gemini/gemini-2.5-flash")
    return _first_config_value(None, ("AUTODUB_ASR_MODEL",), "base")


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
    copyright_confirmed: bool = False,
    copyright_source: str = "unknown",
    copyright_notes: str = "",
) -> PipelineConfig:
    resolved_translation_provider = _first_config_value(
        translation_provider,
        ("AUTODUB_TRANSLATION_PROVIDER",),
        "9router",
    )
    resolved_translation_model = _first_config_value(
        translation_model,
        ("AUTODUB_TRANSLATION_MODEL",),
        "ag/gemini-3-flash-agent",
    )
    resolved_asr_model = _fallback_asr_model(asr_model)
    resolved_compute_type = _first_config_value(compute_type, ("AUTODUB_COMPUTE_TYPE",), "int8")
    resolved_voice_model = _first_config_value(voice_model, ("AUTODUB_VOICE_MODEL",), "Trúc Ly")
    resolved_tts_device = "cuda"
    resolved_ocr_model = _first_config_value(ocr_model, ("AUTODUB_OCR_MODEL",), "gemini/gemini-2.5-flash")

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
        mock_tts=False,
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
        ocr_fallback=ocr_fallback,
        ocr_force=ocr_force,
        ocr_model=resolved_ocr_model,
        ocr_interval_seconds=ocr_interval_seconds,
        ocr_crop_bottom_ratio=ocr_crop_bottom_ratio,
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
    event_queue: queue.Queue[object] = queue.Queue()
    errors: list[BaseException] = []
    request_id = getattr(workspace, "request_id", "unknown")

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
            logger.info("stream.worker.done request_id=%s cancelled=%s", request_id, cancel_event.is_set())

    thread = threading.Thread(target=worker, daemon=True, name=f"autodub-stream-{request_id}")
    thread.start()

    async def event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("stream.client_disconnected request_id=%s cause=browser_reload_or_closed_tab", request_id)
                    cancel_event.set()
                    break
                try:
                    kind, payload = await asyncio.to_thread(event_queue.get, True, 0.2)
                except queue.Empty:
                    if not thread.is_alive() and event_queue.empty():
                        break
                    continue
                if kind == "data":
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
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(
                    "stream.worker.still_running_after_cancel request_id=%s cause=blocking_external_call hint=wait_for_timeout_or_check_9router_ffmpeg_tts",
                    request_id,
                )
            workspace_manager.cleanup(workspace)
            logger.info("stream.workspace.cleaned request_id=%s", request_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
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
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
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
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
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
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
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
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
    )
    config.word_timestamps = True
    _require_copyright_preflight(config)

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
        preview_path = workspace.output_dir / f"{workspace.request_id}_source.mp4"
        shutil.copy2(workspace.input_video, preview_path)
        segments = AutoDubbingPipeline(config).analyze(workspace)
        return AnalyzeResponse(
            request_id=workspace.request_id,
            status="completed",
            source_video_path=f"/media/{preview_path.name}",
            segments=segments,
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
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
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
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
    )
    config.word_timestamps = word_timestamps
    _require_copyright_preflight(config)

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
        preview_path = workspace.output_dir / f"{workspace.request_id}_source.mp4"
        shutil.copy2(workspace.input_video, preview_path)
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
    raw = media_path.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        from urllib.parse import urlparse

        raw = urlparse(raw).path

    name = Path(raw).name
    if not name:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name}")

    output_root = Path("output").resolve()
    path = (output_root / name).resolve()
    if path.parent != output_root:
        raise HTTPException(status_code=403, detail=f"{field_name} must point to output media")
    return path


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
async def upload_voice_reference(audio: UploadFile = File(...)) -> dict[str, str]:
    safe_name = safe_filename(audio.filename, fallback_suffix=".wav")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in VOICE_REFERENCE_SUFFIXES:
        allowed = ", ".join(sorted(VOICE_REFERENCE_SUFFIXES))
        raise HTTPException(status_code=422, detail=f"Unsupported voice reference format. Use: {allowed}")
    if audio.content_type and not (
        audio.content_type.startswith("audio/") or audio.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=422, detail="The clone reference must be an audio file.")

    output_root = Path("output")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{uuid4().hex}_voice_reference{suffix}"
    await save_upload_file(audio, destination)
    if destination.stat().st_size <= 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="The clone reference audio is empty.")

    return {
        "path": f"/media/{destination.name}",
        "filename": safe_name,
    }


@stream_router.post("/render-script")
async def render_edited_script(request: Request, payload: RenderScriptRequest) -> StreamingResponse:
    clone_reference_audio_path: Path | None = None
    if payload.voice_mode == "clone":
        clone_reference_audio_path = _output_media_path(
            payload.clone_reference_audio_path or "",
            field_name="clone_reference_audio_path",
        )
        if not clone_reference_audio_path.is_file():
            raise HTTPException(status_code=404, detail="Clone reference audio not found.")

    config = PipelineConfig(
        target_language=payload.target_language,
        translation_provider=payload.translation_provider,
        translation_model=payload.translation_model,
        voice_model=payload.voice_model,
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
    )
    _require_copyright_preflight(config)
    source_video_path = _output_media_path(payload.source_video_path)
    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()

    return _streaming_pipeline_response(
        request,
        workspace_manager,
        workspace,
        lambda cancel_event: AutoDubbingPipeline(config, cancel_event=cancel_event).render_script(
            workspace=workspace,
            source_video_path=source_video_path,
            script_segments=payload.segments,
        ),
    )


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
            compute_type="int8",
            language=None,
        ) as model:
            result = model.transcribe(audio, batch_size=4, language=None)

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
        if VRAMManager.is_cuda_oom(exc):
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during transcription. Model was offloaded and CUDA cache was cleared.",
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
