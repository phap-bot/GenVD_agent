from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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
from app.utils.vram import VRAMManager
from app.utils.files import safe_filename, save_upload_file
from app.utils.workspace import WorkspaceManager
from utils.model_registry import model_registry
from utils.stt import list_stt_models, remote_stt_enabled, transcribe_audio_remote
from utils.translation import list_translation_models, shorten_text_for_duration

router = APIRouter(prefix="/api/v1", tags=["dubbing"])
compat_router = APIRouter(prefix="/api", tags=["dubbing-compat"])
stream_router = APIRouter(prefix="/api", tags=["dubbing-stream"])
logger = logging.getLogger("auto_dubbing.routes")


REMOTE_ASR_PROVIDERS = {"9router", "remote", "openai-compatible", "openai_compatible", "gemini"}


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
    resolved_tts_device = _first_config_value(tts_device, ("AUTODUB_TTS_DEVICE",), "cuda")
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
        mock_tts=mock_tts,
        ocr_fallback=ocr_fallback,
        ocr_force=ocr_force,
        ocr_model=resolved_ocr_model,
        ocr_interval_seconds=ocr_interval_seconds,
        ocr_crop_bottom_ratio=ocr_crop_bottom_ratio,
    )


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


@stream_router.post("/dub")
async def stream_dub_video(
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
    )

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
    except Exception:
        workspace_manager.cleanup(workspace)
        raise

    def events():
        try:
            yield from AutoDubbingPipeline(config).run(workspace)
        finally:
            workspace_manager.cleanup(workspace)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
    )
    config.word_timestamps = True

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
    )
    config.word_timestamps = word_timestamps

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
        preview_path = workspace.output_dir / f"{workspace.request_id}_source.mp4"
        shutil.copy2(workspace.input_video, preview_path)
    except Exception:
        workspace_manager.cleanup(workspace)
        raise

    def events():
        try:
            yield from AutoDubbingPipeline(config).analyze_stream(workspace)
        finally:
            workspace_manager.cleanup(workspace)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _output_media_path(media_path: str) -> Path:
    raw = media_path.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        from urllib.parse import urlparse

        raw = urlparse(raw).path

    name = Path(raw).name
    if not name:
        raise HTTPException(status_code=422, detail="Invalid source_video_path")

    output_root = Path("output").resolve()
    path = (output_root / name).resolve()
    if path.parent != output_root:
        raise HTTPException(status_code=403, detail="source_video_path must point to output media")
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


@stream_router.post("/render-script")
async def render_edited_script(request: RenderScriptRequest) -> StreamingResponse:
    config = PipelineConfig(
        target_language=request.target_language,
        translation_provider=request.translation_provider,
        translation_model=request.translation_model,
        voice_model=request.voice_model,
        tts_device=request.tts_device,
        background_volume=request.background_volume,
        tts_volume=request.tts_volume,
        burn_subtitles=request.burn_subtitles,
        mock_translation=False,
        mock_tts=request.mock_tts,
    )
    source_video_path = _output_media_path(request.source_video_path)
    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()

    def events():
        try:
            yield from AutoDubbingPipeline(config).render_script(
                workspace=workspace,
                source_video_path=source_video_path,
                script_segments=request.segments,
            )
        finally:
            workspace_manager.cleanup(workspace)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
    mock_tts: bool = Form(default=True),
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
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
    mock_tts: bool = Form(default=True),
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
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
    mock_tts: bool = Form(default=True),
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
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio = whisperx.load_audio(str(audio_path))
        with model_registry.acquire_whisperx_asr(
            whisperx,
            whisper_arch="base",
            device=device,
            compute_type="int8",
            language=None,
        ) as model:
            result = model.transcribe(audio, batch_size=4 if device == "cuda" else 1, language=None)

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
    mock_tts: bool = Form(default=True),
    ocr_fallback: bool = Form(default=True),
    ocr_force: bool = Form(default=False),
    ocr_model: str = Form(default=""),
    ocr_interval_seconds: float = Form(default=0.75),
    ocr_crop_bottom_ratio: float = Form(default=0.35),
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
        tts_device=tts_device,
        background_volume=background_volume,
        tts_volume=tts_volume,
        burn_subtitles=burn_subtitles,
        mock_translation=mock_translation,
        mock_tts=mock_tts,
        ocr_fallback=ocr_fallback,
        ocr_force=ocr_force,
        ocr_model=ocr_model,
        ocr_interval_seconds=ocr_interval_seconds,
        ocr_crop_bottom_ratio=ocr_crop_bottom_ratio,
    )

    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        await workspace_manager.save_upload(video, workspace.input_video)
    except Exception:
        workspace_manager.cleanup(workspace)
        raise

    def events():
        try:
            yield from AutoDubbingPipeline(config).run(workspace)
        finally:
            workspace_manager.cleanup(workspace)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
