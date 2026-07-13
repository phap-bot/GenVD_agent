from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    AnalyzeResponse,
    BatchDubbingResponse,
    DouyinRequest,
    DubbingResponse,
    PipelineConfig,
    RenderScriptRequest,
)
from app.services.dependency_service import DependencyService
from app.services.pipeline import AutoDubbingPipeline
from app.services.pipeline_manager import PipelineManager
from app.utils.files import safe_filename, save_upload_file
from app.utils.workspace import WorkspaceManager

router = APIRouter(prefix="/api/v1", tags=["dubbing"])
compat_router = APIRouter(prefix="/api", tags=["dubbing-compat"])
stream_router = APIRouter(prefix="/api", tags=["dubbing-stream"])


def _config_from_form(
    source_language: str | None,
    target_language: str,
    asr_model: str,
    compute_type: str,
    voice_model: str,
    tts_device: str,
    background_volume: float,
    tts_volume: float,
    burn_subtitles: bool,
    mock_translation: bool,
    mock_tts: bool,
) -> PipelineConfig:
    return PipelineConfig(
        source_language=source_language,
        target_language=target_language,
        asr_model=asr_model,
        compute_type=compute_type,
        voice_model=voice_model,
        tts_device=tts_device,
        background_volume=background_volume,
        tts_volume=tts_volume,
        burn_subtitles=burn_subtitles,
        mock_translation=mock_translation,
        mock_tts=mock_tts,
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
    asr_model: str = Form(default="base"),
    compute_type: str = Form(default="int8"),
    voice_model: str = Form(default="Trúc Ly"),
    tts_device: str = Form(default="cuda"),
    background_volume: float = Form(default=0.35),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=True),
    mock_tts: bool = Form(default=False),
) -> StreamingResponse:
    """SSE endpoint for the Next.js client.

    The upload is persisted before streaming begins. The final output is written
    to `output/`, while the UUID workspace is cleaned after the stream ends.
    """

    config = _config_from_form(
        None if source_language == "auto" else source_language,
        target_language,
        asr_model,
        compute_type,
        voice_model,
        tts_device,
        background_volume,
        tts_volume,
        burn_subtitles,
        mock_translation,
        mock_tts,
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
    asr_model: str = Form(default="base"),
    compute_type: str = Form(default="int8"),
    voice_model: str = Form(default="Trúc Ly"),
    tts_device: str = Form(default="cuda"),
) -> AnalyzeResponse:
    config = _config_from_form(
        None if source_language == "auto" else source_language,
        target_language,
        asr_model,
        compute_type,
        voice_model,
        tts_device,
        0.35,
        1.0,
        True,
        True,
        True,
    )

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
    asr_model: str = Form(default="base"),
    compute_type: str = Form(default="int8"),
    voice_model: str = Form(default="Trúc Ly"),
    tts_device: str = Form(default="cuda"),
    word_timestamps: bool = Form(default=False),
) -> StreamingResponse:
    config = _config_from_form(
        None if source_language == "auto" else source_language,
        target_language,
        asr_model,
        compute_type,
        voice_model,
        tts_device,
        0.35,
        1.0,
        True,
        True,
        True,
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


@stream_router.post("/render-script")
async def render_edited_script(request: RenderScriptRequest) -> StreamingResponse:
    config = PipelineConfig(
        target_language=request.target_language,
        voice_model=request.voice_model,
        tts_device=request.tts_device,
        background_volume=request.background_volume,
        tts_volume=request.tts_volume,
        burn_subtitles=request.burn_subtitles,
        mock_translation=True,
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


@router.post("/dub", response_model=DubbingResponse)
async def dub_video(
    file: UploadFile = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="en"),
    asr_model: str = Form(default="base"),
    compute_type: str = Form(default="int8"),
    voice_model: str = Form(default="Trúc Ly"),
    tts_device: str = Form(default="cuda"),
    background_volume: float = Form(default=0.35),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=True),
    mock_tts: bool = Form(default=True),
) -> DubbingResponse:
    try:
        config = _config_from_form(
            source_language,
            target_language,
            asr_model,
            compute_type,
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
    asr_model: str = Form(default="base"),
    compute_type: str = Form(default="int8"),
    voice_model: str = Form(default="Trúc Ly"),
    tts_device: str = Form(default="cuda"),
    background_volume: float = Form(default=0.35),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=True),
    mock_tts: bool = Form(default=True),
) -> DubbingResponse:
    try:
        config = _config_from_form(
            source_language,
            target_language,
            asr_model,
            compute_type,
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
    asr_model: str = Form(default="base"),
    compute_type: str = Form(default="int8"),
    voice_model: str = Form(default="Trúc Ly"),
    tts_device: str = Form(default="cuda"),
    background_volume: float = Form(default=0.35),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=True),
    mock_tts: bool = Form(default=True),
) -> BatchDubbingResponse:
    try:
        config = _config_from_form(
            source_language,
            target_language,
            asr_model,
            compute_type,
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


@compat_router.post("/process-video")
async def process_video_stream(
    video: UploadFile = File(...),
    source_language: str | None = Form(default=None),
    target_language: str = Form(default="en"),
    voice_model: str = Form(default="Trúc Ly"),
    tts_device: str = Form(default="cuda"),
    background_volume: float = Form(default=0.35),
    tts_volume: float = Form(default=1.0),
    burn_subtitles: bool = Form(default=True),
    mock_translation: bool = Form(default=True),
    mock_tts: bool = Form(default=True),
) -> StreamingResponse:
    """Compatibility endpoint for browser direct fetch streaming.

    The frontend posts directly here instead of going through Next.js. The core
    pipeline remains synchronous and low-VRAM; this endpoint only wraps it in an
    SSE response so clients can keep the connection alive.
    """

    config = PipelineConfig(
        source_language=None if source_language == "auto" else source_language,
        target_language=target_language,
        voice_model=voice_model,
        tts_device=tts_device,
        background_volume=background_volume,
        tts_volume=tts_volume,
        burn_subtitles=burn_subtitles,
        mock_translation=mock_translation,
        mock_tts=mock_tts,
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
