from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from app.models.schemas import BatchPipelineResult, PipelineConfig, PipelineResult, TranscriptSegment
from app.services.asr_service import ASRService
from app.services.downloader_service import DownloaderService
from app.services.subtitle_service import SubtitleService
from app.services.timeline_service import TimelineService
from app.services.translation_service import TranslationService
from app.services.checkpoint_service import CheckpointStore
from app.services.pipeline import TTS_POLICY_VERSION
from app.services.tts_service import TTSService
from app.services.video_service import VideoService
from app.utils.files import ensure_output_dir, make_request_id
from app.utils.memory import VRAMManager
from utils.translation import TRANSLATION_POLICY_VERSION

logger = logging.getLogger(__name__)


class PipelineManager:
    """Synchronous auto-dubbing pipeline with strict sequential AI stages."""

    _pipeline_lock = threading.RLock()

    def __init__(self, output_dir: str | Path = "output") -> None:
        self.output_dir = ensure_output_dir(output_dir)

    def process(self, video_path: Path, config: PipelineConfig) -> PipelineResult:
        """Full video pipeline: extract audio -> ASR -> translate -> TTS -> mux video."""
        config.require_copyright_preflight()
        with self._pipeline_lock:
            return self._process_unlocked(video_path, config)

    def _process_unlocked(self, video_path: Path, config: PipelineConfig) -> PipelineResult:
        request_id = make_request_id()
        output_video_path = self.output_dir / f"{request_id}_dubbed.mp4"
        output_subtitle_path = self.output_dir / f"{request_id}.srt"

        with tempfile.TemporaryDirectory(prefix=f"dub_{request_id}_") as temp_root:
            work_dir = Path(temp_root)
            try:
                checkpoint = CheckpointStore(
                    video_path,
                    root=config.checkpoint_root,
                    enabled=config.checkpoint_enabled,
                )
                accompaniment_path = None
                asr_input = video_path
                if config.vocal_separation:
                    try:
                        logger.info("Starting vocal separation stage for request %s", request_id)
                        from app.services.vocal_separation_service import VocalSeparationService
                        from app.services.dependency_service import DependencyService
                        DependencyService().require_ffmpeg()
                        import ffmpeg
                        source_audio = work_dir / "source_audio.wav"
                        (
                            ffmpeg.input(str(video_path))
                            .output(
                                str(source_audio),
                                ac=2,
                                ar="44100",
                                acodec="pcm_s16le",
                                vn=None,
                                format="wav",
                            )
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True)
                        )
                        sep_result = VocalSeparationService().separate(
                            audio_path=source_audio,
                            output_dir=work_dir / "separated",
                            device=os.environ.get("AUTODUB_DEMUCS_DEVICE", "auto"),
                        )
                        asr_input = sep_result.vocals_path
                        accompaniment_path = sep_result.accompaniment_path
                    except Exception as exc:
                        logger.exception("Vocal separation failed for request %s", request_id)
                        raise RuntimeError(
                            "Vocal separation is enabled but failed; pipeline stopped to prevent "
                            f"original voice bleed. {exc}"
                        ) from exc

                logger.info("Starting ASR stage for request %s", request_id)
                asr_service = None
                asr_payload = {"stage": "asr", **config.model_dump(mode="json", exclude={"clone_reference_audio_path"})}
                cached_asr = checkpoint.load("asr", asr_payload)
                if isinstance(cached_asr, list):
                    segments = [TranscriptSegment.model_validate(item) for item in cached_asr]
                    logger.info("checkpoint.hit manager stage=asr request_id=%s segments=%s", request_id, len(segments))
                else:
                    asr_service = ASRService(config)
                    segments = TimelineService().from_transcript(
                        asr_service.transcribe(asr_input, work_dir),
                        merge_semantic=True,
                        source_language=config.source_language,
                    )
                    checkpoint.save("asr", asr_payload, [item.model_dump(mode="json") for item in segments])
                if asr_service is not None:
                    del asr_service
                VRAMManager.cleanup()

                logger.info("Starting translation stage for request %s", request_id)
                translation_service = None
                translation_payload = {
                    "stage": "translation",
                    "policy_version": TRANSLATION_POLICY_VERSION,
                    **config.model_dump(mode="json", exclude={"clone_reference_audio_path"}),
                    "segments": [item.model_dump(mode="json") for item in segments],
                }
                cached_translation = checkpoint.load("translation", translation_payload)
                if isinstance(cached_translation, list):
                    translated_segments = [TranscriptSegment.model_validate(item) for item in cached_translation]
                    logger.info("checkpoint.hit manager stage=translation request_id=%s segments=%s", request_id, len(translated_segments))
                else:
                    translation_service = TranslationService(config)
                    translated_segments = TimelineService().from_transcript(translation_service.translate(segments))
                    checkpoint.save("translation", translation_payload, [item.model_dump(mode="json") for item in translated_segments])
                if translation_service is not None:
                    del translation_service
                VRAMManager.cleanup()

                logger.info("Starting TTS stage for request %s", request_id)
                self._checkpoint_voice_setup(checkpoint, config, translated_segments, request_id)
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
                    accompaniment_path=accompaniment_path,
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

    def _checkpoint_voice_setup(
        self,
        checkpoint: CheckpointStore,
        config: PipelineConfig,
        segments: list[TranscriptSegment],
        request_id: str,
    ) -> None:
        """Persist the selected voice/timing identity before TTS starts."""
        payload = {
            "stage": "voice_setup",
            "policy_version": TTS_POLICY_VERSION,
            "voice_mode": config.voice_mode,
            "voice_model": config.voice_model.strip(),
            "clone_reference_audio_path": config.clone_reference_audio_path,
            "voice_speed": round(config.voice_speed, 3),
            "soft_timing_fit": bool(config.soft_timing_fit),
            "timing_max_drift_s": round(config.timing_max_drift_s, 3),
            "timing_min_gap_s": round(config.timing_min_gap_s, 3),
            "timing_max_atempo": round(config.timing_max_atempo, 3),
            "segments": [item.model_dump(mode="json") for item in segments],
        }
        cached = checkpoint.load("voice_setup", payload)
        if isinstance(cached, dict):
            logger.info(
                "checkpoint.hit manager stage=voice_setup request_id=%s mode=%s voice=%s segments=%s",
                request_id,
                config.voice_mode,
                config.voice_model,
                len(segments),
            )
            return
        checkpoint_path = checkpoint.save(
            "voice_setup",
            payload,
            {
                "voice_mode": config.voice_mode,
                "default_voice": config.voice_model.strip(),
                "segments": len(segments),
            },
        )
        if checkpoint_path is not None:
            logger.info(
                "checkpoint.saved manager stage=voice_setup request_id=%s mode=%s voice=%s segments=%s",
                request_id,
                config.voice_mode,
                config.voice_model,
                len(segments),
            )

    def process_with_srt(
        self,
        video_path: Path,
        subtitle_path: Path,
        config: PipelineConfig,
    ) -> PipelineResult:
        """Dub a video from an existing SRT instead of running ASR."""
        config.require_copyright_preflight()
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
                accompaniment_path = None
                if config.vocal_separation:
                    try:
                        logger.info("Starting vocal separation stage for request %s", request_id)
                        from app.services.vocal_separation_service import VocalSeparationService
                        from app.services.dependency_service import DependencyService
                        DependencyService().require_ffmpeg()
                        import ffmpeg
                        source_audio = work_dir / "source_audio.wav"
                        (
                            ffmpeg.input(str(video_path))
                            .output(
                                str(source_audio),
                                ac=2,
                                ar="44100",
                                acodec="pcm_s16le",
                                vn=None,
                                format="wav",
                            )
                            .overwrite_output()
                            .run(capture_stdout=True, capture_stderr=True)
                        )
                        sep_result = VocalSeparationService().separate(
                            audio_path=source_audio,
                            output_dir=work_dir / "separated",
                            device=os.environ.get("AUTODUB_DEMUCS_DEVICE", "auto"),
                        )
                        accompaniment_path = sep_result.accompaniment_path
                    except Exception as exc:
                        logger.exception("Vocal separation failed for request %s", request_id)
                        raise RuntimeError(
                            "Vocal separation is enabled but failed; pipeline stopped to prevent "
                            f"original voice bleed. {exc}"
                        ) from exc

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
                self._checkpoint_voice_setup(
                    CheckpointStore(
                        video_path,
                        root=config.checkpoint_root,
                        enabled=config.checkpoint_enabled,
                    ),
                    config,
                    translated_segments,
                    request_id,
                )
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
                    accompaniment_path=accompaniment_path,
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
