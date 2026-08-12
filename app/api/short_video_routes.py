from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.routes import (
    _cache_source_media,
    _config_from_form,
    _render_config,
    _render_job_streaming_response,
    _require_copyright_preflight,
    _streaming_pipeline_response,
    _voice_reference_media_path,
)
from app.models.schemas import RenderScriptRequest, ShortVideoInspectResponse, ShortVideoProfilesResponse, ShortVideoRenderRequest
from app.services.pipeline import AutoDubbingPipeline
from app.services.render_job_service import RenderJobDispatcher
from app.services.short_video_service import ShortVideoService
from app.utils.files import safe_filename, save_upload_file
from app.utils.workspace import WorkspaceManager


router = APIRouter(prefix="/api/v1/short-video", tags=["short-video"])
service = ShortVideoService()


def _resolve_asr_model(requested: str, duration: float) -> str:
    clean = (requested or "").strip()
    if clean and clean.lower() != "auto":
        return clean
    return "base" if duration <= 60 else "small"


def _load_short_source(media_id: str) -> tuple[Path, object]:
    try:
        path = service.resolve(media_id)
        profile = service.require_short(path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return path, profile


async def _create_short_workspace(media_id: str) -> tuple[WorkspaceManager, object, Path, object]:
    source_path, profile = _load_short_source(media_id)
    workspace_manager = WorkspaceManager()
    workspace = workspace_manager.create()
    try:
        shutil.copy2(source_path, workspace.input_video)
        preview_path = workspace.output_dir / f"{workspace.request_id}_short_source{source_path.suffix.lower() or '.mp4'}"
        shutil.copy2(source_path, preview_path)
        _cache_source_media(source_path, preview_path.name)
        return workspace_manager, workspace, preview_path, profile
    except Exception:
        workspace_manager.cleanup(workspace)
        raise


def _short_config(
    profile,
    *,
    source_language: str | None,
    target_language: str,
    translation_provider: str,
    translation_model: str,
    asr_model: str,
    compute_type: str,
    word_timestamps: bool,
    voice_model: str,
    background_volume: float,
    tts_volume: float,
    burn_subtitles: bool,
    mock_translation: bool,
    mock_tts: bool,
    ocr_fallback: bool,
    ocr_force: bool,
    ocr_model: str,
    ocr_interval_seconds: float,
    ocr_crop_bottom_ratio: float,
    copyright_confirmed: bool,
    copyright_source: str,
    copyright_notes: str,
    vocal_separation: bool,
    voice_mode: str = "system",
    clone_reference_audio_path: str | None = None,
):
    config = _config_from_form(
        None if source_language in {None, "", "auto"} else source_language,
        target_language,
        translation_provider,
        translation_model,
        _resolve_asr_model(asr_model, profile.duration_seconds),
        compute_type,
        word_timestamps,
        voice_model,
        "cuda",
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
        # Short Video does not pay the Demucs cost during analysis by default.
        vocal_separation=vocal_separation,
    )
    # Short Video is intentionally hybrid: retain speech recognition while
    # also reading hard-subtitles across the full short timeline.
    config.source_mode = "hybrid"
    config.ocr_max_frames = 600
    config.voice_mode = voice_mode if voice_mode in {"system", "clone"} else "system"
    if config.voice_mode == "clone":
        if not clone_reference_audio_path:
            raise HTTPException(status_code=422, detail="Clone voice mode requires a reference audio file.")
        reference = _voice_reference_media_path(clone_reference_audio_path)
        config.clone_reference_audio_path = str(reference)
    return config


@router.get("/profiles", response_model=ShortVideoProfilesResponse)
def short_video_profiles() -> ShortVideoProfilesResponse:
    max_seconds = service.max_short_seconds
    return ShortVideoProfilesResponse(
        short_video_max_seconds=max_seconds,
        profiles=[
            service.resolve_profile(30),
            service.resolve_profile(120),
            service.resolve_profile(min(240, max_seconds)),
            service.resolve_profile(max_seconds + 1),
        ],
    )


@router.post("/inspect", response_model=ShortVideoInspectResponse)
async def inspect_short_video(video: UploadFile = File(...)) -> ShortVideoInspectResponse:
    service.evict_stale()
    safe_name = safe_filename(video.filename, fallback_suffix=".mp4")
    temp_dir = Path(tempfile.mkdtemp(prefix="short_upload_", dir="temp"))
    temp_path = temp_dir / safe_name
    try:
        await save_upload_file(video, temp_path)
        if temp_path.stat().st_size <= 0:
            raise HTTPException(status_code=422, detail="Source video upload is empty.")
        media_id, stored_path = service.store_upload(safe_name, temp_path)
        payload = service.inspect(media_id, stored_path)
        return ShortVideoInspectResponse.model_validate(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Short Video inspect failed: {exc}") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/analyze")
async def analyze_short_video(
    request: Request,
    media_id: str = Form(...),
    source_language: str = Form(default="auto"),
    target_language: str = Form(default="vi"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    asr_model: str = Form(default="auto"),
    compute_type: str = Form(default="int8"),
    voice_model: str = Form(default=""),
    ocr_fallback: bool = Form(default=True),
    ocr_force: bool = Form(default=False),
    ocr_model: str = Form(default=""),
    ocr_interval_seconds: float = Form(default=0.5),
    ocr_crop_bottom_ratio: float = Form(default=0.35),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
) -> StreamingResponse:
    _source_path, profile = _load_short_source(media_id)
    config = _short_config(
        profile,
        source_language=source_language,
        target_language=target_language,
        translation_provider=translation_provider,
        translation_model=translation_model,
        asr_model=asr_model,
        compute_type=compute_type,
        word_timestamps=True,
        voice_model=voice_model,
        background_volume=0.0,
        tts_volume=1.0,
        burn_subtitles=True,
        mock_translation=False,
        mock_tts=True,
        ocr_fallback=ocr_fallback,
        ocr_force=ocr_force,
        ocr_model=ocr_model,
        ocr_interval_seconds=ocr_interval_seconds,
        ocr_crop_bottom_ratio=ocr_crop_bottom_ratio,
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
        vocal_separation=False,
    )
    try:
        config.require_copyright_preflight()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    workspace_manager, workspace, _preview_path, _profile = await _create_short_workspace(media_id)

    return _streaming_pipeline_response(
        request,
        workspace_manager,
        workspace,
        lambda cancel_event: AutoDubbingPipeline(config, cancel_event=cancel_event).analyze_stream(workspace),
    )


@router.post("/dub")
async def dub_short_video(
    request: Request,
    media_id: str = Form(...),
    source_language: str = Form(default="auto"),
    target_language: str = Form(default="vi"),
    translation_provider: str = Form(default=""),
    translation_model: str = Form(default=""),
    asr_model: str = Form(default="auto"),
    compute_type: str = Form(default="int8"),
    voice_model: str = Form(default=""),
    background_volume: float = Form(default=0.0),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=False),
    mock_tts: bool = Form(default=False),
    ocr_fallback: bool = Form(default=True),
    ocr_force: bool = Form(default=False),
    ocr_model: str = Form(default=""),
    ocr_interval_seconds: float = Form(default=0.5),
    ocr_crop_bottom_ratio: float = Form(default=0.35),
    copyright_confirmed: bool = Form(default=False),
    copyright_source: str = Form(default="unknown"),
    copyright_notes: str = Form(default=""),
    vocal_separation: bool = Form(default=False),
    voice_mode: str = Form(default="system"),
    clone_reference_audio_path: str | None = Form(default=None),
) -> StreamingResponse:
    _source_path, profile = _load_short_source(media_id)
    config = _short_config(
        profile,
        source_language=source_language,
        target_language=target_language,
        translation_provider=translation_provider,
        translation_model=translation_model,
        asr_model=asr_model,
        compute_type=compute_type,
        word_timestamps=True,
        voice_model=voice_model,
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
        copyright_confirmed=copyright_confirmed,
        copyright_source=copyright_source,
        copyright_notes=copyright_notes,
        vocal_separation=vocal_separation,
        voice_mode=voice_mode,
        clone_reference_audio_path=clone_reference_audio_path,
    )
    try:
        config.require_copyright_preflight()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    workspace_manager, workspace, _preview_path, _profile = await _create_short_workspace(media_id)

    return _streaming_pipeline_response(
        request,
        workspace_manager,
        workspace,
        lambda cancel_event: AutoDubbingPipeline(config, cancel_event=cancel_event).run(workspace),
    )


@router.post("/render-script")
async def render_short_script(request: Request, payload: ShortVideoRenderRequest) -> StreamingResponse:
    """Render an analyzed/edited Short Video timeline without re-running ASR."""
    source_path, profile = _load_short_source(payload.media_id)
    if profile.route != "short_video":
        raise HTTPException(status_code=422, detail="This media belongs to the long-video Clone Video workflow.")

    clone_reference_path: Path | None = None
    if payload.voice_mode == "clone":
        clone_reference_path = _voice_reference_media_path(payload.clone_reference_audio_path or "")

    render_payload = RenderScriptRequest(
        source_video_path=f"/temp/short_video/{payload.media_id}/{source_path.name}",
        target_language=payload.target_language,
        translation_provider=payload.translation_provider,
        translation_model=payload.translation_model,
        voice_model=payload.voice_model,
        voice_mode=payload.voice_mode,
        clone_reference_audio_path=(
            f"/media/{clone_reference_path.name}" if clone_reference_path is not None else None
        ),
        tts_device=payload.tts_device,
        background_volume=payload.background_volume,
        tts_volume=payload.tts_volume,
        burn_subtitles=payload.burn_subtitles,
        mock_tts=payload.mock_tts,
        copyright_confirmed=payload.copyright_confirmed,
        copyright_source=payload.copyright_source,
        copyright_notes=payload.copyright_notes,
        vocal_separation=payload.vocal_separation,
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
        segments=payload.segments,
    )
    try:
        _require_copyright_preflight(render_payload)
        config = _render_config(render_payload, clone_reference_path)
        dispatcher = RenderJobDispatcher()
        submission = await asyncio.to_thread(
            dispatcher.submit,
            render_payload,
            config,
            source_path,
            clone_reference_path,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to queue Short Video render: {exc}") from exc
    return _render_job_streaming_response(request, dispatcher, submission)
