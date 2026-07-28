from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from pathlib import Path

from app.models.schemas import BatchPipelineResult, PipelineConfig, PipelineResult
from app.services.asr_service import ASRService
from app.services.downloader_service import DownloaderService
from app.services.subtitle_service import SubtitleService
from app.services.timeline_service import TimelineService
from app.services.translation_service import TranslationService
from app.services.tts_service import TTSService
from app.services.video_service import VideoService
from app.utils.files import ensure_output_dir, make_request_id
from app.utils.memory import VRAMManager

logger = logging.getLogger(__name__)


class PipelineManager:
    """Synchronous auto-dubbing pipeline with strict sequential AI stages."""

    _pipeline_lock = threading.RLock()

    def __init__(self, output_dir: str | Path = "output") -> None:
        self.output_dir = ensure_output_dir(output_dir)

    def process(self, video_path: Path, config: PipelineConfig) -> PipelineResult:
        """Full video pipeline: extract audio -> ASR -> translate -> TTS -> mux video."""
        with self._pipeline_lock:
            return self._process_unlocked(video_path, config)

    def _process_unlocked(self, video_path: Path, config: PipelineConfig) -> PipelineResult:
        request_id = make_request_id()
        output_video_path = self.output_dir / f"{request_id}_dubbed.mp4"
        output_subtitle_path = self.output_dir / f"{request_id}.srt"

        with tempfile.TemporaryDirectory(prefix=f"dub_{request_id}_") as temp_root:
            work_dir = Path(temp_root)
            try:
                logger.info("Starting ASR stage for request %s", request_id)
                asr_service = ASRService(config)
                segments = TimelineService().from_transcript(
                    asr_service.transcribe(video_path, work_dir),
                    merge_semantic=True,
                    source_language=config.source_language,
                )
                del asr_service
                VRAMManager.cleanup()

                logger.info("Starting translation stage for request %s", request_id)
                translation_service = TranslationService(config)
                translated_segments = TimelineService().from_transcript(translation_service.translate(segments))
                del translation_service
                VRAMManager.cleanup()

                logger.info("Starting TTS stage for request %s", request_id)
                tts_service = TTSService(config)
                tts_tracks = tts_service.synthesize(translated_segments, work_dir)
                del tts_service
                VRAMManager.cleanup()

                logger.info("Starting video render stage for request %s", request_id)
                rendered_path, temp_subtitle_path = VideoService(config).render(
                    video_path=video_path,
                    segments=translated_segments,
                    tts_tracks=tts_tracks,
                    work_dir=work_dir,
                    output_path=output_video_path,
                )
                shutil.copy2(temp_subtitle_path, output_subtitle_path)

                return PipelineResult(
                    request_id=request_id,
                    output_video_path=rendered_path,
                    subtitle_path=output_subtitle_path,
                    segments=translated_segments,
                )
            finally:
                VRAMManager.cleanup()

    def process_with_srt(
        self,
        video_path: Path,
        subtitle_path: Path,
        config: PipelineConfig,
    ) -> PipelineResult:
        """Dub a video from an existing SRT instead of running ASR."""
        with self._pipeline_lock:
            return self._process_with_srt_unlocked(video_path, subtitle_path, config)

    def _process_with_srt_unlocked(
        self,
        video_path: Path,
        subtitle_path: Path,
        config: PipelineConfig,
    ) -> PipelineResult:
        request_id = make_request_id()
        output_video_path = self.output_dir / f"{request_id}_dubbed.mp4"
        output_subtitle_path = self.output_dir / f"{request_id}.srt"

        with tempfile.TemporaryDirectory(prefix=f"dub_srt_{request_id}_") as temp_root:
            work_dir = Path(temp_root)
            try:
                logger.info("Starting SRT parse stage for request %s", request_id)
                segments = TimelineService().from_transcript(SubtitleService().parse_srt(subtitle_path))
                if not segments:
                    raise ValueError("No valid subtitle segments found in SRT file")

                logger.info("Starting translation stage for request %s", request_id)
                translation_service = TranslationService(config)
                translated_segments = TimelineService().from_transcript(translation_service.translate(segments))
                del translation_service
                VRAMManager.cleanup()

                logger.info("Starting TTS stage for request %s", request_id)
                tts_service = TTSService(config)
                tts_tracks = tts_service.synthesize(translated_segments, work_dir)
                del tts_service
                VRAMManager.cleanup()

                logger.info("Starting video render stage for request %s", request_id)
                rendered_path, temp_subtitle_path = VideoService(config).render(
                    video_path=video_path,
                    segments=translated_segments,
                    tts_tracks=tts_tracks,
                    work_dir=work_dir,
                    output_path=output_video_path,
                )
                shutil.copy2(temp_subtitle_path, output_subtitle_path)

                return PipelineResult(
                    request_id=request_id,
                    output_video_path=rendered_path,
                    subtitle_path=output_subtitle_path,
                    segments=translated_segments,
                )
            finally:
                VRAMManager.cleanup()

    def process_batch(
        self,
        video_paths: list[Path],
        config: PipelineConfig,
    ) -> BatchPipelineResult:
        """Process many videos strictly one by one to avoid GPU OOM."""
        request_id = make_request_id()
        results: list[PipelineResult] = []
        failed: list[str] = []

        for video_path in video_paths:
            try:
                results.append(self.process(video_path, config))
            except Exception as exc:
                logger.exception("Batch item failed: %s", video_path)
                failed.append(f"{video_path.name}: {exc}")
            finally:
                VRAMManager.cleanup()

        return BatchPipelineResult(request_id=request_id, results=results, failed=failed)

    def process_douyin(
        self,
        url: str,
        config: PipelineConfig,
        max_items: int = 10,
    ) -> BatchPipelineResult:
        """Download Douyin videos, then process each file sequentially."""
        request_id = make_request_id()
        with tempfile.TemporaryDirectory(prefix=f"douyin_{request_id}_") as temp_root:
            download_dir = Path(temp_root) / "downloads"
            video_paths = DownloaderService().download_douyin(
                url=url,
                destination_dir=download_dir,
                max_items=max_items,
            )
            if not video_paths:
                raise ValueError("No videos were downloaded from Douyin URL")

            batch = self.process_batch(video_paths, config)
            return BatchPipelineResult(
                request_id=request_id,
                results=batch.results,
                failed=batch.failed,
            )
