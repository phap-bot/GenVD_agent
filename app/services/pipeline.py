from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

from app.models.schemas import DubbingScriptSegment, PipelineConfig, TranscriptSegment, WordTimestamp
from app.services.dependency_service import DependencyService
from app.services.timeline_service import TimelineService
from app.services.translation_service import TranslationService
from app.utils.media_probe import probe_duration, probe_video_dimensions
from app.utils.vram import VRAMManager
from app.utils.workspace import Workspace
from app.utils.cancel import PipelineCancelledError
from utils.model_cache import configure_model_cache
from utils.model_registry import model_registry
from utils.ocr import extract_video_ocr_segments
from utils.stt import remote_stt_enabled, transcribe_audio_remote
from utils.tts_voice import (
    encode_cloned_vieneu_voice,
    infer_stable_cloned_vieneu_audio,
    infer_stable_vieneu_audio,
    resolve_vieneu_voice,
)

logger = logging.getLogger(__name__)
MODEL_CACHE_PATHS = configure_model_cache()
MAX_SEGMENT_DURATION = 8.0
MAX_CJK_SEGMENT_DURATION = 7.0
MAX_SEGMENT_CHARS = 180
MAX_CJK_SEGMENT_CHARS = 72
HARD_MAX_SEGMENT_DURATION = 14.0
HARD_MAX_SEGMENT_CHARS = 280
HARD_MAX_CJK_SEGMENT_CHARS = 120
MIN_SEGMENT_DURATION = 0.2
PUNCTUATION = set(".!?;,\u3002\uff01\uff1f\uff1b\uff0c\u3001")
SENTENCE_TERMINATORS = set(".!?;\u3002\uff01\uff1f\uff1b")
CLAUSE_PUNCTUATION = set(",:\uff0c\u3001\uff1a")
GPU_LOCK_POLL_SECONDS = 1.0
GPU_LOCK_WAIT_TIMEOUT_SECONDS = 180.0
TTS_CACHE_MAX_AGE_SECONDS = 2 * 24 * 3600  # 2 days
AUDIO_STEM_CACHE_VERSION = "demucs-normalized-residual-v2"
ACCOMPANIMENT_DEFAULT_VOLUME = 0.92
ORIGINAL_FALLBACK_VOLUME = 0.12

# ── TTS segment merging constants ──
TTS_MERGE_MAX_GAP = 0.5           # max silence gap (seconds) between segments to merge
TTS_MERGE_MAX_SEG_DURATION = 2.0  # only merge segments shorter than this (seconds)
TTS_MERGE_MAX_GROUP_DURATION = 6.0  # max total duration of a merged group
TTS_MERGE_MAX_CHARS = 160         # max total characters of merged text


@dataclass(frozen=True)
class AudioChunk:
    segment_id: int
    path: Path
    start: float
    end: float


@dataclass(frozen=True)
class _TTSGroup:
    """One or more adjacent short segments merged for a single TTS call."""
    text: str
    start: float
    end: float
    first_segment_id: int
    segment_count: int
    voice_model: str  # resolved voice key (empty for clone mode)


class AutoDubbingPipeline:
    """SSE-producing orchestrator for the auto-dubbing pipeline.

    All model stages are guarded by a process-wide lock so two requests cannot
    load GPU models at the same time on a 4GB VRAM machine.
    """

    _gpu_lock = threading.RLock()

    def __init__(self, config: PipelineConfig, cancel_event: threading.Event | None = None) -> None:
        self.config = config
        self.cancel_event = cancel_event
        self.dependencies = DependencyService()
        self.last_source_engine = "unknown"

    def run(self, workspace: Workspace) -> Generator[str, None, Path]:
        output_path = workspace.output_dir / f"{workspace.request_id}_dubbed.mp4"
        output_subtitle_path = output_path.with_suffix(".srt")
        self.config.require_copyright_preflight()

        try:
            yield self._event(
                "processing",
                "Initializing workspace...",
                request_id=workspace.request_id,
                phase="prepare",
                progress=3,
            )
            self.dependencies.require_ffmpeg()
            video_duration = self._safe_probe_duration(workspace.input_video)
            yield self._event(
                "processing",
                "Media ready",
                phase="prepare",
                progress=8,
                stats={"video_duration": video_duration},
            )

            self._acquire_gpu_lock(workspace)
            try:
                yield self._event(
                    "processing",
                    "Extracting audio and transcribing...",
                    phase="recognize",
                    progress=14,
                    detail="Running ASR/OCR from backend",
                )
                segments = self._source_timeline(self._extract_source_segments(workspace))
                source_stats = self._segment_stats(segments)
                yield self._event(
                    "processing",
                    "Source timeline extracted",
                    phase="recognize",
                    progress=38,
                    detail=f"{self.last_source_engine} returned {len(segments)} segment(s)",
                    stats={**source_stats, "engine": self.last_source_engine},
                )
                VRAMManager.cleanup()
                self._raise_if_cancelled(workspace)

                yield self._event(
                    "processing",
                    "Translating script...",
                    phase="translate",
                    progress=46,
                    stats=source_stats,
                )
                translated_segments = self._canonical_timeline(self._translate_segments(segments))
                yield self._event(
                    "processing",
                    "Script translated",
                    phase="translate",
                    progress=62,
                    stats=self._segment_stats(translated_segments),
                )
                VRAMManager.cleanup()
                self._raise_if_cancelled(workspace)

                yield self._event(
                    "processing",
                    "Generating AI voice...",
                    phase="voice",
                    progress=70,
                    stats=self._segment_stats(translated_segments),
                )
                chunks = yield from self._run_tts(translated_segments, workspace)
                yield self._event(
                    "processing",
                    "Voice tracks generated",
                    phase="voice",
                    progress=84,
                    stats={"chunks": len(chunks), **self._segment_stats(translated_segments)},
                )
                VRAMManager.cleanup()
            finally:
                self._gpu_lock.release()

            yield self._event("processing", "Rendering final video...", phase="render", progress=92)
            subtitle_path = workspace.root / "translated.srt"
            self._write_srt(translated_segments, subtitle_path)
            tts_mix_path = workspace.root / "tts_mix.wav"
            self._combine_audio_chunks(chunks, tts_mix_path, total_duration=video_duration)
            yield self._event("processing", "Muxing subtitles and audio...", phase="render", progress=96)
            accompaniment_path = self._ensure_accompaniment_audio(workspace, workspace.input_video) if self.config.vocal_separation else None
            self._render_video(
                video_path=workspace.input_video,
                subtitle_path=subtitle_path,
                tts_mix_path=tts_mix_path,
                output_path=output_path,
                accompaniment_path=accompaniment_path,
            )
            self._write_srt(translated_segments, output_subtitle_path, video_duration=video_duration)

            yield self._event(
                "success",
                "Completed",
                phase="complete",
                progress=100,
                video_url=f"/media/{output_path.name}",
                subtitle_url=f"/media/{output_subtitle_path.name}",
            )
            return output_path
        except PipelineCancelledError:
            raise
        except Exception as exc:
            VRAMManager.cleanup()
            if VRAMManager.is_cuda_error(exc):
                VRAMManager.reset_after_cuda_error()
                logger.exception("CUDA failure during dubbing pipeline")
                yield self._event(
                    "error",
                    "CUDA bi loi trong luc tao giong. Model cache da reset va VRAM da duoc don; thu render lai, neu con lap thi restart backend.",
                    error=str(exc),
                )
                return output_path

            logger.exception("Dubbing pipeline failed")
            yield self._event("error", "Pipeline failed", error=str(exc))
            return output_path
        finally:
            VRAMManager.cleanup()

    def analyze(self, workspace: Workspace) -> list[DubbingScriptSegment]:
        """Extract script/timeline without rendering final video."""
        self.config.require_copyright_preflight()
        with self._gpu_lock:
            try:
                self.dependencies.require_ffmpeg()
                segments = self._source_timeline(self._extract_source_segments(workspace))
                VRAMManager.cleanup()
                translated = self._canonical_timeline(self._translate_segments(segments))
                return [
                    DubbingScriptSegment(
                        id=segment.id,
                        start=segment.start,
                        end=segment.end,
                        original_text=segments[index].text if index < len(segments) else segment.text,
                        translated_text=segment.text,
                        voice_model=self.config.voice_model,
                    )
                    for index, segment in enumerate(translated)
                ]
            finally:
                VRAMManager.cleanup()

    def analyze_stream(self, workspace: Workspace) -> Generator[str, None, list[DubbingScriptSegment]]:
        """Extract script/timeline while streaming progress to the client."""
        self.config.require_copyright_preflight()
        try:
            yield self._event(
                "processing",
                "Initializing analyze workspace...",
                request_id=workspace.request_id,
                phase="prepare",
                progress=5,
            )
            self.dependencies.require_ffmpeg()
            video_duration = self._safe_probe_duration(workspace.input_video)
            yield self._event(
                "processing",
                "Media ready",
                phase="prepare",
                progress=10,
                stats={"video_duration": video_duration},
            )
            yield self._event(
                "processing",
                "Waiting for recognition engine...",
                request_id=workspace.request_id,
                phase="recognize",
                progress=12,
                detail="Đang kiểm tra hàng đợi model/GPU",
            )
            lock_acquired = yield from self._acquire_gpu_lock_for_stream(
                workspace,
                phase="recognize",
                progress=15,
                operation="analyze",
            )
            if not lock_acquired:
                return []

            try:
                yield self._event(
                    "processing",
                    "Extracting audio and transcribing...",
                    request_id=workspace.request_id,
                    phase="recognize",
                    progress=18,
                    detail="Running selected ASR/OCR engine",
                )
                segments = self._source_timeline(self._extract_source_segments(workspace))
                source_stats = self._segment_stats(segments)
                yield self._event(
                    "processing",
                    "Source timeline extracted",
                    phase="recognize",
                    progress=58,
                    detail=f"{self.last_source_engine} returned {len(segments)} segment(s)",
                    stats={**source_stats, "engine": self.last_source_engine},
                )
                VRAMManager.cleanup()

                yield self._event(
                    "processing",
                    "Translating script...",
                    request_id=workspace.request_id,
                    phase="translate",
                    progress=70,
                    stats=source_stats,
                )
                translated = self._canonical_timeline(self._translate_segments(segments))
                yield self._event(
                    "processing",
                    "Script translated",
                    phase="translate",
                    progress=92,
                    stats=self._segment_stats(translated),
                )
                analyzed = [
                    DubbingScriptSegment(
                        id=segment.id,
                        start=segment.start,
                        end=segment.end,
                        original_text=segments[index].text if index < len(segments) else segment.text,
                        translated_text=segment.text,
                        voice_model=self.config.voice_model,
                    )
                    for index, segment in enumerate(translated)
                ]
            finally:
                self._gpu_lock.release()

            yield self._event(
                "success",
                "Analyze completed",
                phase="complete",
                progress=100,
                source_video_path=f"/media/{workspace.request_id}_source.mp4",
                segments=[segment.model_dump() for segment in analyzed],
            )
            return analyzed
        except PipelineCancelledError:
            raise
        except Exception as exc:
            VRAMManager.cleanup()
            logger.exception("Analyze stream failed")
            yield self._event("error", "Analyze failed", error=str(exc))
            return []
        finally:
            VRAMManager.cleanup()

    def render_script(
        self,
        workspace: Workspace,
        source_video_path: Path,
        script_segments: list[DubbingScriptSegment],
    ) -> Generator[str, None, Path]:
        """Render a final video from user-edited script/timeline segments."""
        output_path = workspace.output_dir / f"{workspace.request_id}_script_dubbed.mp4"
        output_subtitle_path = output_path.with_suffix(".srt")
        timeline_segments = TimelineService().from_script(script_segments)
        self.config.require_copyright_preflight()

        try:
            yield self._event(
                "processing",
                "Initializing render workspace...",
                request_id=workspace.request_id,
                phase="prepare",
                progress=5,
                stats=self._script_stats(script_segments),
            )
            self.dependencies.require_ffmpeg()
            self._raise_if_cancelled(workspace)
            if not source_video_path.is_file() or source_video_path.stat().st_size <= 0:
                raise FileNotFoundError(f"Source video not found: {source_video_path}")
            render_source_path = workspace.root / "render_source.mp4"
            shutil.copy2(source_video_path, render_source_path)
            logger.info(
                "script_render.source_staged request_id=%s source=%s staged=%s bytes=%s",
                workspace.request_id,
                source_video_path,
                render_source_path,
                render_source_path.stat().st_size,
            )
            video_duration = self._safe_probe_duration(render_source_path)
            timeline_segments = self._clip_timeline_to_video(timeline_segments, video_duration)

            self._acquire_gpu_lock(workspace)
            try:
                yield self._event(
                    "processing",
                    "Generating AI voice from edited script...",
                    phase="voice",
                    progress=18,
                    stats=self._script_stats(script_segments),
                )
                chunks = yield from self._run_tts_from_script(script_segments, timeline_segments, workspace)
                yield self._event(
                    "processing",
                    "Voice tracks generated",
                    phase="voice",
                    progress=72,
                    stats={"chunks": len(chunks), **self._script_stats(script_segments)},
                )
                VRAMManager.cleanup()
            finally:
                self._gpu_lock.release()

            yield self._event("processing", "Rendering final video...", phase="render", progress=84)
            subtitle_path = workspace.root / "edited_script.ass"
            video_width, video_height = self._video_dimensions(render_source_path)
            self._write_ass(script_segments, subtitle_path, video_width=video_width, video_height=video_height, video_duration=video_duration)
            tts_mix_path = workspace.root / "tts_mix.wav"
            self._combine_audio_chunks(chunks, tts_mix_path, total_duration=video_duration)
            yield self._event("processing", "Muxing subtitles and audio...", phase="render", progress=94)
            accompaniment_path = self._ensure_accompaniment_audio(workspace, render_source_path) if self.config.vocal_separation else None
            self._render_video(
                video_path=render_source_path,
                subtitle_path=subtitle_path,
                tts_mix_path=tts_mix_path,
                output_path=output_path,
                script_segments=script_segments,
                video_width=video_width,
                video_height=video_height,
                video_duration=video_duration,
                accompaniment_path=accompaniment_path,
            )
            self._write_srt(timeline_segments, output_subtitle_path, video_duration=video_duration)

            yield self._event(
                "success",
                "Completed",
                phase="complete",
                progress=100,
                video_url=f"/media/{output_path.name}",
                subtitle_url=f"/media/{output_subtitle_path.name}",
            )
            return output_path
        except PipelineCancelledError:
            raise
        except Exception as exc:
            VRAMManager.cleanup()
            if VRAMManager.is_cuda_error(exc):
                VRAMManager.reset_after_cuda_error()
                logger.exception("CUDA failure during script render")
                yield self._event(
                    "error",
                    "CUDA bi loi trong luc tao giong. Model cache da reset va VRAM da duoc don; thu render lai, neu con lap thi restart backend.",
                    error=str(exc),
                )
                return output_path

            logger.exception("Script render failed")
            yield self._event("error", "Script render failed", error=str(exc))
            return output_path
        finally:
            VRAMManager.cleanup()

    def _extract_source_segments(self, workspace: Workspace) -> list[TranscriptSegment]:
        if self.config.ocr_force:
            logger.info("source_extract.ocr_forced request_id=%s", workspace.request_id)
            return self._run_ocr(workspace)

        has_audio = self._has_audio_stream(workspace.input_video)
        if not has_audio:
            logger.info("source_extract.no_audio_using_ocr request_id=%s", workspace.request_id)
            return self._run_ocr(workspace)

        segments = self._run_asr(workspace)
        if self._should_use_ocr_fallback(segments):
            logger.info(
                "source_extract.asr_sparse_using_ocr request_id=%s asr_segments=%s asr_chars=%s",
                workspace.request_id,
                len(segments),
                sum(len(segment.text.strip()) for segment in segments),
            )
            ocr_segments = self._run_ocr(workspace)
            if ocr_segments:
                return ocr_segments

        return segments

    def _should_use_ocr_fallback(self, segments: list[TranscriptSegment]) -> bool:
        if not self.config.ocr_fallback:
            return False
        text_chars = sum(len(segment.text.strip()) for segment in segments)
        speech_duration = sum(max(0.0, segment.end - segment.start) for segment in segments)
        return not segments or text_chars < 8 or speech_duration < 0.5

    def _run_ocr(self, workspace: Workspace) -> list[TranscriptSegment]:
        self._raise_if_cancelled(workspace)
        self.last_source_engine = "OCR"
        ocr_segments = extract_video_ocr_segments(
            workspace.input_video,
            source_language=self.config.source_language,
            model=self.config.ocr_model or self.config.translation_model,
            interval_seconds=self.config.ocr_interval_seconds,
            crop_bottom_ratio=self.config.ocr_crop_bottom_ratio,
            cancel_event=self.cancel_event,
        )
        return [
            TranscriptSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=[],
            )
            for segment in ocr_segments
        ]

    def _ensure_accompaniment_audio(self, workspace: Workspace, video_path: Path) -> Path | None:
        if not self.config.vocal_separation:
            return None
        acc_path = workspace.root / "accompaniment.wav"
        if acc_path.is_file() and acc_path.stat().st_size > 0:
            return acc_path

        if not self._has_audio_stream(video_path):
            return None

        try:
            from app.services.vocal_separation_service import VocalSeparationService

            cache_acc_path, cache_vocals_path = self._stem_cache_paths(video_path)
            self._evict_stale_stem_cache(cache_acc_path.parent)
            if cache_acc_path.is_file() and cache_acc_path.stat().st_size > 0:
                shutil.copy2(cache_acc_path, acc_path)
                if cache_vocals_path.is_file() and cache_vocals_path.stat().st_size > 0:
                    sep_dir = workspace.root / "separated"
                    sep_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cache_vocals_path, sep_dir / "vocals.wav")
                logger.info(
                    "vocal_separation.cache_hit source=%s accompaniment=%s",
                    video_path,
                    cache_acc_path,
                )
                return acc_path

            # Keep source separation independent from the 16 kHz mono ASR
            # file. Demucs must receive stereo full-band audio.
            source_audio = workspace.root / "separation_source.wav"
            if not source_audio.is_file() or source_audio.stat().st_size <= 0:
                ffmpeg = self._ffmpeg()
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
            sep_dir = workspace.root / "separated"
            device = os.environ.get("AUTODUB_DEMUCS_DEVICE", "cpu").strip().lower()
            if device not in {"cpu", "cuda"}:
                logger.warning("vocal_separation.invalid_device value=%s fallback=cpu", device)
                device = "cpu"
            logger.info("vocal_separation.ensure_start device=%s path=%s", device, source_audio)
            sep_result = VocalSeparationService().separate(
                audio_path=source_audio,
                output_dir=sep_dir,
                device=device,
                cancel_event=self.cancel_event,
            )
            shutil.copy2(str(sep_result.accompaniment_path), str(acc_path))
            try:
                shutil.copy2(sep_result.accompaniment_path, cache_acc_path)
                shutil.copy2(sep_result.vocals_path, cache_vocals_path)
                logger.info(
                    "vocal_separation.cache_saved accompaniment=%s vocals=%s",
                    cache_acc_path,
                    cache_vocals_path,
                )
            except OSError:
                logger.warning("vocal_separation.cache_save_failed source=%s", video_path, exc_info=True)
            return acc_path
        except PipelineCancelledError:
            raise
        except Exception as exc:
            logger.exception("vocal_separation.ensure_failed error=%s", exc)
            raise RuntimeError(
                "Vocal separation is enabled but failed; render stopped to prevent original voice bleed. "
                f"{exc}"
            ) from exc

    def _stem_cache_paths(self, video_path: Path) -> tuple[Path, Path]:
        digest = hashlib.sha256()
        digest.update(AUDIO_STEM_CACHE_VERSION.encode("ascii"))
        stat = video_path.stat()
        digest.update(str(stat.st_size).encode("ascii"))
        sample_size = 2 * 1024 * 1024
        with video_path.open("rb") as handle:
            offsets = (0, max(0, stat.st_size // 2 - sample_size // 2), max(0, stat.st_size - sample_size))
            for offset in offsets:
                handle.seek(offset)
                digest.update(handle.read(sample_size))

        cache_dir = Path("audio_cache") / "stems"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = digest.hexdigest()[:24]
        return cache_dir / f"{key}_accompaniment.wav", cache_dir / f"{key}_vocals.wav"

    def _evict_stale_stem_cache(self, cache_dir: Path) -> None:
        now = time.time()
        try:
            for entry in cache_dir.glob("*.wav"):
                if now - entry.stat().st_mtime > TTS_CACHE_MAX_AGE_SECONDS:
                    entry.unlink(missing_ok=True)
        except OSError:
            logger.warning("vocal_separation.cache_eviction_failed dir=%s", cache_dir, exc_info=True)

    def _run_asr(self, workspace: Workspace) -> list[TranscriptSegment]:
        self._raise_if_cancelled(workspace)
        audio_path = workspace.root / "source_audio.wav"
        if not self._has_audio_stream(workspace.input_video):
            logger.info("asr.skip.no_audio request_id=%s input=%s", workspace.request_id, workspace.input_video)
            self.last_source_engine = "no audio"
            return []

        ffmpeg = self._ffmpeg()
        (
            ffmpeg.input(str(workspace.input_video))
            .output(str(audio_path), ac=1, ar="16000", vn=None, format="wav")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        if self.config.vocal_separation:
            self._ensure_accompaniment_audio(workspace, workspace.input_video)
            vocals_path = workspace.root / "separated" / "vocals.wav"
            if vocals_path.is_file():
                logger.info("asr.using_separated_vocals path=%s", vocals_path)
                audio_path = vocals_path

        try:
            if remote_stt_enabled(self.config.asr_model):
                self.last_source_engine = f"remote STT ({self.config.asr_model})"
                logger.info(
                    "legacy_pipeline.asr.remote.start request_id=%s model=%s language=%s audio=%s",
                    workspace.request_id,
                    self.config.asr_model,
                    self.config.source_language or "auto",
                    audio_path,
                )
                return self._normalize_segments(
                    transcribe_audio_remote(
                        audio_path,
                        source_language=self.config.source_language,
                        model=self.config.asr_model,
                    )
                )

            try:
                import whisperx
            except ImportError as exc:
                raise RuntimeError("WhisperX is required for GPU ASR. Install whisperx, then restart the backend.") from exc

            self.dependencies.require_cuda()
            device = "cuda"
            self.last_source_engine = f"WhisperX {self.config.asr_model} on {device}"
            logger.info(
                "legacy_pipeline.asr.model.load request_id=%s model=%s compute_type=%s device=%s language=%s cache=%s",
                workspace.request_id,
                self.config.asr_model,
                self.config.compute_type,
                device,
                self.config.source_language or "auto",
                MODEL_CACHE_PATHS.whisperx_asr_cache,
            )
            audio = whisperx.load_audio(str(audio_path))
            with model_registry.acquire_whisperx_asr(
                whisperx,
                whisper_arch=self.config.asr_model,
                device=device,
                compute_type=self.config.compute_type,
                language=self.config.source_language,
            ) as model:
                result = model.transcribe(
                    audio,
                    batch_size=4,
                    language=self.config.source_language,
                )

            VRAMManager.cleanup()

            if not self.config.word_timestamps:
                return self._normalize_segments(result.get("segments", []))

            language_code = result.get("language") or self.config.source_language or "en"
            logger.info(
                "legacy_pipeline.asr.align.load request_id=%s language=%s device=%s cache=%s",
                workspace.request_id,
                language_code,
                device,
                MODEL_CACHE_PATHS.whisperx_align_cache,
            )
            raw_segs = result.get("segments", [])
            valid_segs = [s for s in raw_segs if s.get("text", "").strip()]
            if valid_segs:
                try:
                    with model_registry.acquire_whisperx_align(
                        whisperx,
                        language_code=language_code,
                        device=device,
                    ) as (align_model, metadata):
                        aligned = whisperx.align(
                            valid_segs,
                            align_model,
                            metadata,
                            audio,
                            device,
                            return_char_alignments=False,
                        )
                    return self._normalize_segments(aligned.get("segments", []))
                except Exception as exc:
                    logger.warning(
                        "legacy_pipeline.asr.align.failed request_id=%s language=%s error=%s, falling back to ASR segments",
                        workspace.request_id,
                        language_code,
                        exc,
                    )

            return self._normalize_segments(raw_segs)
        finally:
            VRAMManager.cleanup()

    def _translate_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        self._raise_if_cancelled()
        return TranslationService(self.config, self.cancel_event).translate(segments)

    def _source_timeline(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        return TimelineService().from_transcript(
            segments,
            merge_semantic=True,
            source_language=self.config.source_language,
        )

    def _canonical_timeline(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        return TimelineService().from_transcript(segments)

    def _run_tts(self, segments: list[TranscriptSegment], workspace: Workspace) -> Generator[str, None, list[AudioChunk]]:
        chunks: list[AudioChunk] = []
        self._raise_if_cancelled(workspace)

        try:
            if self.config.mock_tts:
                raise RuntimeError("Mock TTS is disabled because GPU TTS is required.")
            if self.config.tts_device != "cuda":
                raise RuntimeError("CPU TTS is disabled. Use CUDA TTS only.")
            cache_paths = configure_model_cache()
            logger.info(
                "legacy_pipeline.tts.model_cache request_id=%s device=%s hf_home=%s hf_hub_cache=%s",
                workspace.request_id,
                self.config.tts_device,
                cache_paths.hf_home,
                cache_paths.hf_hub_cache,
            )
            self.dependencies.require_cuda()
            with model_registry.acquire_vieneu(device="cuda", backend="pytorch") as model:
                if self.config.voice_mode == "clone":
                    voice_source = encode_cloned_vieneu_voice(
                        model,
                        self.config.clone_reference_audio_path or "",
                    )
                    logger.info(
                        "legacy_pipeline.tts.voice.clone request_id=%s reference=%s",
                        workspace.request_id,
                        self.config.clone_reference_audio_path,
                    )
                else:
                    voice_source = resolve_vieneu_voice(model, self.config.voice_model)
                    logger.info(
                        "legacy_pipeline.tts.voice.system request_id=%s requested=%s resolved=%s",
                        workspace.request_id,
                        self.config.voice_model,
                        voice_source,
                    )

                total_segments = max(1, len(segments))
                for index, segment in enumerate(segments):
                    self._raise_if_cancelled(workspace)
                    raw_path = workspace.chunks_dir / f"{segment.id:04d}_raw.wav"
                    final_path = workspace.chunks_dir / f"{segment.id:04d}.wav"
                    duration = max(segment.end - segment.start, 0.1)
                    logger.info(
                        "legacy_pipeline.tts.segment.start request_id=%s segment_id=%s index=%s total=%s mode=%s chars=%s duration=%.3f",
                        workspace.request_id,
                        segment.id,
                        index + 1,
                        total_segments,
                        self.config.voice_mode,
                        len(segment.text),
                        duration,
                    )

                    if self.config.voice_mode == "clone":
                        audio = infer_stable_cloned_vieneu_audio(model, segment.text, voice_source)
                    else:
                        audio = infer_stable_vieneu_audio(model, segment.text, voice_source)
                    model.save(audio, str(raw_path))

                    self._fit_audio_duration(raw_path, final_path, duration)
                    chunks.append(AudioChunk(segment.id, final_path, segment.start, segment.end))
                    progress = min(84, 70 + round(((index + 1) / total_segments) * 14))
                    yield self._event(
                        "processing",
                        f"Đã tạo voice {index + 1}/{total_segments}",
                        phase="voice",
                        progress=progress,
                        stats={
                            "chunks": len(chunks),
                            "segments": total_segments,
                            "segment_id": segment.id,
                        },
                    )

            return chunks
        finally:
            VRAMManager.cleanup()

    # ── TTS chunk cache helpers ──────────────────────────────────────────

    def _tts_cache_dir(self) -> Path:
        cache_dir = Path("tts_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _clone_reference_cache_id(self, clone_ref: str | None) -> str:
        if self.config.voice_mode != "clone":
            return ""

        clean_ref = (clone_ref or "").strip()
        if not clean_ref:
            return ""

        path = Path(clean_ref)
        try:
            stat = path.stat()
        except OSError:
            logger.warning("tts_cache.clone_ref_stat_failed path=%s", clean_ref)
            return clean_ref

        signature = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        cached_signature = getattr(self, "_clone_reference_cache_signature", None)
        cached_id = getattr(self, "_clone_reference_cache_id_value", None)
        if cached_signature == signature and isinstance(cached_id, str):
            return cached_id

        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            logger.warning("tts_cache.clone_ref_hash_failed path=%s", clean_ref)
            return clean_ref

        cache_id = f"sha256:{digest.hexdigest()}"
        self._clone_reference_cache_signature = signature
        self._clone_reference_cache_id_value = cache_id
        logger.info(
            "tts_cache.clone_ref_identity path=%s bytes=%s identity=%s",
            clean_ref,
            stat.st_size,
            cache_id[:24],
        )
        return cache_id

    def _tts_cache_key(
        self,
        text: str,
        voice_mode: str,
        voice_model: str,
        clone_ref: str | None,
        duration: float,
    ) -> str:
        payload = json.dumps(
            {
                "fit_version": 2,
                "text": text.strip(),
                "voice_mode": voice_mode,
                "voice_model": voice_model,
                "clone_ref": self._clone_reference_cache_id(clone_ref),
                "duration": round(duration, 3),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _evict_stale_tts_cache(self) -> None:
        """Remove cached TTS chunks older than TTS_CACHE_MAX_AGE_SECONDS."""
        cache_dir = self._tts_cache_dir()
        now = time.time()
        evicted = 0
        try:
            for entry in cache_dir.iterdir():
                if entry.is_file() and entry.suffix == ".wav":
                    age = now - entry.stat().st_mtime
                    if age > TTS_CACHE_MAX_AGE_SECONDS:
                        entry.unlink(missing_ok=True)
                        evicted += 1
        except Exception:
            logger.exception("tts_cache.eviction.failed")
        if evicted:
            logger.info("tts_cache.evicted count=%d max_age_days=%.1f", evicted, TTS_CACHE_MAX_AGE_SECONDS / 86400)

    def _tts_cache_lookup(
        self,
        segment: TranscriptSegment,
        voice_segments: list[DubbingScriptSegment],
        index: int,
    ) -> Path | None:
        """Return the cache path if a fitted WAV already exists for this segment."""
        duration = max(segment.end - segment.start, 0.1)
        voice_model = ""
        if self.config.voice_mode == "system" and index < len(voice_segments):
            voice_model = voice_segments[index].voice_model.strip()
        cache_key = self._tts_cache_key(
            segment.text,
            self.config.voice_mode,
            voice_model,
            self.config.clone_reference_audio_path,
            duration,
        )
        cache_path = self._tts_cache_dir() / f"{cache_key}.wav"
        exists = cache_path.is_file() and cache_path.stat().st_size > 0
        if index < 3 or not exists:
            logger.info(
                "tts_cache.lookup segment_id=%s index=%d key=%s duration=%.3f text_len=%d exists=%s",
                segment.id, index, cache_key, duration, len(segment.text), exists,
            )
        if exists:
            return cache_path
        return None

    def _tts_group_cache_key(self, group: _TTSGroup) -> str:
        """Cache key for a merged TTS group."""
        duration = max(group.end - group.start, 0.1)
        return self._tts_cache_key(
            group.text,
            self.config.voice_mode,
            group.voice_model,
            self.config.clone_reference_audio_path,
            duration,
        )

    def _tts_group_cache_lookup(self, group: _TTSGroup) -> Path | None:
        """Return the cache path if a fitted WAV already exists for this group."""
        cache_key = self._tts_group_cache_key(group)
        cache_path = self._tts_cache_dir() / f"{cache_key}.wav"
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            return cache_path
        return None

    # ── TTS segment merging ────────────────────────────────────────────

    def _build_tts_groups(
        self,
        timeline_segments: list[TranscriptSegment],
        voice_segments: list[DubbingScriptSegment],
    ) -> list[_TTSGroup]:
        """Merge adjacent short segments into groups for fewer TTS calls.

        Rules:
        - Only merge segments shorter than TTS_MERGE_MAX_SEG_DURATION.
        - Gap between adjacent segments must be < TTS_MERGE_MAX_GAP.
        - Total group duration must be < TTS_MERGE_MAX_GROUP_DURATION.
        - Total group text length must be < TTS_MERGE_MAX_CHARS.
        - In system voice mode, only merge segments with the same voice model.
        """
        if not timeline_segments:
            return []

        groups: list[_TTSGroup] = []
        # Accumulator for current group being built.
        cur_indices: list[int] = []
        cur_texts: list[str] = []

        def _voice_key(idx: int) -> str:
            if self.config.voice_mode != "system":
                return ""
            if idx < len(voice_segments):
                return voice_segments[idx].voice_model.strip()
            return ""

        def _flush() -> None:
            if not cur_indices:
                return
            first_idx = cur_indices[0]
            last_idx = cur_indices[-1]
            first_seg = timeline_segments[first_idx]
            last_seg = timeline_segments[last_idx]
            merged_text = " ".join(cur_texts)
            groups.append(_TTSGroup(
                text=merged_text,
                start=first_seg.start,
                end=last_seg.end,
                first_segment_id=first_seg.id,
                segment_count=len(cur_indices),
                voice_model=_voice_key(first_idx),
            ))

        for i, seg in enumerate(timeline_segments):
            seg_dur = seg.end - seg.start
            seg_text = seg.text.strip()

            if not cur_indices:
                # Start a new group.
                cur_indices = [i]
                cur_texts = [seg_text]
                continue

            prev_seg = timeline_segments[cur_indices[-1]]
            gap = seg.start - prev_seg.end
            group_start = timeline_segments[cur_indices[0]].start
            merged_dur = seg.end - group_start
            merged_text_len = sum(len(t) for t in cur_texts) + len(seg_text) + len(cur_texts)

            can_merge = (
                seg_dur <= TTS_MERGE_MAX_SEG_DURATION
                and gap <= TTS_MERGE_MAX_GAP
                and merged_dur <= TTS_MERGE_MAX_GROUP_DURATION
                and merged_text_len <= TTS_MERGE_MAX_CHARS
                and not self._ends_with_terminal_punctuation(prev_seg.text)
                and _voice_key(i) == _voice_key(cur_indices[0])
                # The segments already in the group must also be short.
                and all(
                    timeline_segments[j].end - timeline_segments[j].start <= TTS_MERGE_MAX_SEG_DURATION
                    for j in cur_indices
                )
            )

            if can_merge:
                cur_indices.append(i)
                cur_texts.append(seg_text)
            else:
                _flush()
                cur_indices = [i]
                cur_texts = [seg_text]

        _flush()
        return groups

    # ── TTS generation with cache-first + merged groups ───────────────

    def _run_tts_from_script(
        self,
        script_segments: list[DubbingScriptSegment],
        timeline_segments: list[TranscriptSegment],
        workspace: Workspace,
    ) -> Generator[str, None, list[AudioChunk]]:
        chunks: list[AudioChunk] = []
        self._raise_if_cancelled(workspace)
        voice_segments = sorted(script_segments, key=lambda item: (item.start, item.end))

        # Evict stale cache entries on each render.
        self._evict_stale_tts_cache()

        # Build merged TTS groups.
        tts_groups = self._build_tts_groups(timeline_segments, voice_segments)
        total_original = len(timeline_segments)
        total_groups = len(tts_groups)
        merged_count = sum(1 for g in tts_groups if g.segment_count > 1)
        logger.info(
            "tts_merge.summary request_id=%s original_segments=%d groups=%d merged_groups=%d",
            workspace.request_id,
            total_original,
            total_groups,
            merged_count,
        )

        # Fast path: if ALL groups are already cached, skip GPU entirely.
        all_cached = all(
            self._tts_group_cache_lookup(group) is not None
            for group in tts_groups
        )
        if all_cached:
            logger.info(
                "tts_cache.full_hit request_id=%s groups=%d — skipping GPU model load",
                workspace.request_id,
                total_groups,
            )
            for gi, group in enumerate(tts_groups):
                self._raise_if_cancelled(workspace)
                final_path = workspace.chunks_dir / f"{group.first_segment_id:04d}.wav"
                cached = self._tts_group_cache_lookup(group)
                shutil.copy2(str(cached), str(final_path))
                chunks.append(AudioChunk(group.first_segment_id, final_path, group.start, group.end))
                progress = min(72, 18 + round(((gi + 1) / total_groups) * 54))
                yield self._event(
                    "processing",
                    f"Đã tạo voice {gi + 1}/{total_groups} (cached)",
                    phase="voice",
                    progress=progress,
                    stats={"chunks": len(chunks), "segments": total_original, "groups": total_groups, "cached": True},
                )
            return chunks

        # Normal path: load model, but skip cached groups.
        try:
            cache_paths = configure_model_cache()
            logger.info(
                "legacy_pipeline.tts.model_cache request_id=%s device=%s hf_home=%s hf_hub_cache=%s",
                workspace.request_id,
                self.config.tts_device,
                cache_paths.hf_home,
                cache_paths.hf_hub_cache,
            )
            self.dependencies.require_cuda()
            with model_registry.acquire_vieneu(device="cuda", backend="pytorch") as model:
                if self.config.voice_mode == "clone":
                    clone_voice_reference = encode_cloned_vieneu_voice(
                        model,
                        self.config.clone_reference_audio_path or "",
                    )
                    logger.info(
                        "legacy_pipeline.tts_script.voice.clone request_id=%s reference=%s",
                        workspace.request_id,
                        self.config.clone_reference_audio_path,
                    )
                else:
                    clone_voice_reference = None

                if self.config.voice_mode == "system":
                    character_voice_map: dict[str, str] = {}
                else:
                    character_voice_map = {}

                cache_hits = 0
                for gi, group in enumerate(tts_groups):
                    self._raise_if_cancelled(workspace)
                    final_path = workspace.chunks_dir / f"{group.first_segment_id:04d}.wav"
                    duration = max(group.end - group.start, 0.1)

                    # ── Cache hit: copy and skip inference ──
                    cached = self._tts_group_cache_lookup(group)
                    if cached is not None:
                        shutil.copy2(str(cached), str(final_path))
                        chunks.append(AudioChunk(group.first_segment_id, final_path, group.start, group.end))
                        cache_hits += 1
                        progress = min(72, 18 + round(((gi + 1) / total_groups) * 54))
                        yield self._event(
                            "processing",
                            f"Đã tạo voice {gi + 1}/{total_groups} (cached)",
                            phase="voice",
                            progress=progress,
                            stats={"chunks": len(chunks), "segments": total_original, "groups": total_groups, "cached": True},
                        )
                        continue

                    # ── Cache miss: run TTS inference ──
                    raw_path = workspace.chunks_dir / f"{group.first_segment_id:04d}_raw.wav"
                    segs_label = f"({group.segment_count} segs)" if group.segment_count > 1 else ""
                    logger.info(
                        "legacy_pipeline.tts_script.group.start request_id=%s group=%s/%s segment_id=%s segs=%s mode=%s chars=%s duration=%.3f",
                        workspace.request_id,
                        gi + 1,
                        total_groups,
                        group.first_segment_id,
                        group.segment_count,
                        self.config.voice_mode,
                        len(group.text),
                        duration,
                    )

                    if self.config.voice_mode == "clone":
                        audio = infer_stable_cloned_vieneu_audio(model, group.text, clone_voice_reference)
                    else:
                        character_key = group.voice_model
                        if not character_key:
                            raise ValueError(f"System voice is missing for segment {group.first_segment_id}.")
                        voice = character_voice_map.setdefault(
                            character_key,
                            resolve_vieneu_voice(model, character_key),
                        )
                        audio = infer_stable_vieneu_audio(model, group.text, voice)

                    model.save(audio, str(raw_path))
                    self._fit_audio_duration(raw_path, final_path, duration)

                    # ── Save to cache ──
                    cache_key = self._tts_group_cache_key(group)
                    try:
                        cache_dest = self._tts_cache_dir() / f"{cache_key}.wav"
                        shutil.copy2(str(final_path), str(cache_dest))
                    except Exception:
                        logger.warning("tts_cache.save_failed segment_id=%s cache_key=%s", group.first_segment_id, cache_key)

                    chunks.append(AudioChunk(group.first_segment_id, final_path, group.start, group.end))
                    progress = min(72, 18 + round(((gi + 1) / total_groups) * 54))
                    yield self._event(
                        "processing",
                        f"Đã tạo voice {gi + 1}/{total_groups} {segs_label}",
                        phase="voice",
                        progress=progress,
                        stats={
                            "chunks": len(chunks),
                            "segments": total_original,
                            "groups": total_groups,
                            "segment_id": group.first_segment_id,
                        },
                    )

                if cache_hits:
                    logger.info(
                        "tts_cache.summary request_id=%s total_groups=%d hits=%d misses=%d",
                        workspace.request_id,
                        total_groups,
                        cache_hits,
                        total_groups - cache_hits,
                    )

            return chunks
        finally:
            VRAMManager.cleanup()

    def _fit_audio_duration(self, source: Path, destination: Path, target_duration: float) -> None:
        if self._fit_audio_duration_with_numpy(source, destination, target_duration):
            return

        ffmpeg = self._ffmpeg()
        current_duration = self._probe_duration(source)
        if current_duration <= 0:
            self._write_silent_wav(destination, target_duration)
            return

        try:
            natural_target = max(target_duration, 0.1)
            ratio = max(0.1, current_duration / natural_target)
            tempo = ratio if current_duration > natural_target else 1.0
            if tempo > 1.35:
                logger.warning(
                    "tts.audio_fit.high_speed source=%s raw_duration=%.3f target_duration=%.3f tempo=%.3f",
                    source,
                    current_duration,
                    natural_target,
                    tempo,
                )
            stream = ffmpeg.input(str(source)).audio
            for value in self._atempo_filters(tempo):
                stream = stream.filter("atempo", value)

            stream = stream.filter("apad").filter("atrim", duration=natural_target)
            command = ffmpeg.output(stream, str(destination), ac=1, ar="24000", format="wav").overwrite_output()
            self._run_ffmpeg_command(command, "audio_fit")
            return
        except Exception:
            logger.exception("ffmpeg duration fitting failed, falling back to pydub")

        if self._fit_audio_duration_with_pydub(source, destination, target_duration):
            return
        raise RuntimeError(f"Could not fit TTS audio to {target_duration:.3f}s")

    def _fit_audio_duration_with_numpy(self, source: Path, destination: Path, target_duration: float) -> bool:
        try:
            with wave.open(str(source), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.getnframes()
                raw_bytes = wf.readframes(frames)

            if frames == 0 or sample_rate <= 0:
                self._write_silent_wav(destination, target_duration, sample_rate=sample_rate or 24000)
                return True

            import numpy as np

            if sample_width == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16)
            elif sample_width == 4:
                samples = np.frombuffer(raw_bytes, dtype=np.int32)
            else:
                return False

            current_duration = frames / float(sample_rate)
            natural_target = max(target_duration, 0.1)
            if current_duration > natural_target:
                # Never truncate spoken audio. Let FFmpeg atempo compress the
                # complete utterance while preserving pitch.
                return False

            target_samples = math.ceil(natural_target * sample_rate * channels)
            if len(samples) < target_samples:
                samples = np.pad(samples, (0, target_samples - len(samples)), mode="constant")

            with wave.open(str(destination), "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())
            return True
        except Exception:
            logger.exception("numpy audio duration fitting failed")
            return False

    def _fit_audio_duration_with_pydub(self, source: Path, destination: Path, target_duration: float) -> bool:
        try:
            from pydub import AudioSegment
        except ImportError:
            return False

        try:
            audio = AudioSegment.from_file(source)
            target_ms = max(100, round(target_duration * 1000))
            if len(audio) <= 0:
                self._write_silent_wav(destination, target_duration)
                return True

            speed = len(audio) / target_ms
            natural_speed = speed if len(audio) > target_ms else 1.0
            fitted = audio
            if natural_speed > 1.0:
                fitted = audio._spawn(
                    audio.raw_data,
                    overrides={"frame_rate": max(1, round(audio.frame_rate * natural_speed))},
                ).set_frame_rate(audio.frame_rate)

            if len(fitted) > target_ms:
                fitted = fitted[:target_ms]
            elif len(fitted) < target_ms:
                fitted += AudioSegment.silent(duration=target_ms - len(fitted), frame_rate=audio.frame_rate)

            fitted.export(destination, format="wav")
            return True
        except Exception:
            logger.exception("pydub duration fitting failed, falling back to ffmpeg atempo")
            return False

    def _combine_audio_chunks(self, chunks: list[AudioChunk], destination: Path, *, total_duration: float | None = None) -> None:
        """Mix TTS audio chunks into a single WAV at sample-accurate offsets.

        Uses direct PCM buffer mixing instead of an FFmpeg CLI filter graph to
        avoid Windows ``WinError 206`` (command line too long) when the segment
        count is large.  Output format (44100 Hz, stereo, 16-bit WAV) and
        additive-sum semantics (equivalent to ``amix normalize=0``) are
        preserved exactly.
        """
        target_duration = max(float(total_duration or 0.0), 0.1)
        if not chunks:
            self._write_silent_wav(destination, target_duration, sample_rate=44100)
            return

        import numpy as np

        out_sample_rate = 44100
        out_channels = 2
        total_frames = math.ceil(target_duration * out_sample_rate)

        # int32 accumulator avoids overflow when overlapping chunks are summed.
        buffer = np.zeros((total_frames, out_channels), dtype=np.int32)
        mixed_count = 0

        for chunk in chunks:
            # --- read chunk WAV ---
            try:
                with wave.open(str(chunk.path), "rb") as wav_file:
                    chunk_sample_rate = wav_file.getframerate()
                    chunk_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    raw_bytes = wav_file.readframes(wav_file.getnframes())
            except Exception:
                logger.warning(
                    "audio_mix.chunk_read_failed path=%s segment_id=%s",
                    chunk.path,
                    chunk.segment_id,
                )
                continue

            # --- decode to int32 samples ---
            if sample_width == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.int32)
            elif sample_width == 4:
                samples = np.frombuffer(raw_bytes, dtype=np.int32).copy()
            else:
                logger.warning(
                    "audio_mix.unsupported_sample_width path=%s width=%s",
                    chunk.path,
                    sample_width,
                )
                continue

            if len(samples) == 0:
                continue

            # --- downmix to mono if multi-channel ---
            if chunk_channels > 1:
                samples = samples.reshape(-1, chunk_channels)[:, 0].copy()

            # --- resample to output rate (linear interpolation) ---
            if chunk_sample_rate != out_sample_rate:
                resampled_length = max(1, round(len(samples) * out_sample_rate / chunk_sample_rate))
                indices = np.linspace(0, len(samples) - 1, resampled_length)
                samples = np.interp(
                    indices,
                    np.arange(len(samples)),
                    samples.astype(np.float64),
                ).astype(np.int32)

            # --- place chunk at correct sample offset ---
            sample_offset = max(0, round(chunk.start * out_sample_rate))
            end_sample = min(total_frames, sample_offset + len(samples))
            usable = end_sample - sample_offset
            if usable <= 0:
                continue

            # Additive mix into both stereo channels (same as amix normalize=0).
            buffer[sample_offset:end_sample, 0] += samples[:usable]
            buffer[sample_offset:end_sample, 1] += samples[:usable]
            mixed_count += 1

        # --- clamp and write output WAV ---
        buffer = np.clip(buffer, -32768, 32767).astype(np.int16)
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(out_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(out_sample_rate)
            wav_file.writeframes(buffer.tobytes())

        logger.info(
            "audio_mix.completed destination=%s chunks=%s mixed=%s duration=%.2f frames=%s",
            destination,
            len(chunks),
            mixed_count,
            target_duration,
            total_frames,
        )

    def _render_video(
        self,
        video_path: Path,
        subtitle_path: Path,
        tts_mix_path: Path,
        output_path: Path,
        *,
        script_segments: Iterable[DubbingScriptSegment] | None = None,
        video_width: int | None = None,
        video_height: int | None = None,
        video_duration: float | None = None,
        accompaniment_path: Path | None = None,
    ) -> None:
        ffmpeg = self._ffmpeg()
        video_input = ffmpeg.input(str(video_path))
        tts_input = ffmpeg.input(str(tts_mix_path))

        video_stream = video_input.video
        if script_segments:
            video_stream = self._apply_blur_boxes(
                ffmpeg,
                video_stream,
                script_segments,
                video_width=video_width or 0,
                video_height=video_height or 0,
                video_duration=video_duration,
            )
        if self.config.burn_subtitles:
            subtitle_filter_path = self._ffmpeg_filter_path(subtitle_path)
            logger.info("legacy_pipeline.render.subtitle_filter path=%s", subtitle_filter_path)
            video_stream = video_stream.filter("subtitles", subtitle_filter_path)

        tts_audio = tts_input.audio.filter("volume", self.config.tts_volume)
        tts_split = tts_audio.filter_multi_output("asplit")
        tts_sidechain = tts_split[0]
        tts_for_mix = tts_split[1]
        has_acc = accompaniment_path and accompaniment_path.is_file() and accompaniment_path.stat().st_size > 0
        if self.config.vocal_separation and not has_acc:
            raise RuntimeError(
                "Vocal separation is enabled but no accompaniment track is available; "
                "render stopped to prevent original voice bleed."
            )
        if has_acc:
            bg_vol = self.config.background_volume if self.config.background_volume > 0 else ACCOMPANIMENT_DEFAULT_VOLUME
            logger.info(
                "legacy_pipeline.render.accompaniment_ducked path=%s volume=%.2f ratio=12 release_ms=320",
                accompaniment_path,
                bg_vol,
            )
            bg_audio = ffmpeg.input(str(accompaniment_path)).audio.filter("volume", bg_vol)
            # Keep the original ambience near full level between speech, then
            # duck it while dubbed speech is active. This masks residual vocal
            # bleed without flattening music and environmental sound globally.
            background_audio = ffmpeg.filter(
                [bg_audio, tts_sidechain],
                "sidechaincompress",
                threshold=0.02,
                ratio=12,
                attack=10,
                release=320,
                makeup=1,
            )
            mixed_audio = ffmpeg.filter(
                [background_audio, tts_for_mix],
                "amix",
                inputs=2,
                duration="first",
                dropout_transition=0,
                normalize=0,
            )
        else:
            bg_vol = self.config.background_volume if self.config.background_volume > 0 else ORIGINAL_FALLBACK_VOLUME
            logger.info(
                "legacy_pipeline.render.original_audio_emergency_duck video=%s volume=%.2f ratio=20 release_ms=420",
                video_path,
                bg_vol,
            )
            original_audio = video_input.audio.filter("volume", bg_vol)
            background_audio = ffmpeg.filter(
                [original_audio, tts_sidechain],
                "sidechaincompress",
                threshold=0.01,
                ratio=20,
                attack=5,
                release=420,
                makeup=1,
            )
            mixed_audio = ffmpeg.filter(
                [background_audio, tts_for_mix],
                "amix",
                inputs=2,
                duration="first",
                dropout_transition=0,
                normalize=0,
            )
        mixed_audio = mixed_audio.filter("alimiter", limit=0.95)

        command = ffmpeg.output(
            video_stream,
            mixed_audio,
            str(output_path),
            vcodec="libx264",
            acodec="aac",
        ).overwrite_output()
        self._run_ffmpeg_command(command, "render")

    def _apply_blur_boxes(
        self,
        ffmpeg,
        video_stream,
        segments: Iterable[DubbingScriptSegment],
        *,
        video_width: int,
        video_height: int,
        video_duration: float | None = None,
    ):
        width = max(1, int(video_width))
        height = max(1, int(video_height))
        ordered_segments = sorted(segments, key=lambda item: (item.start, item.end))
        render_until = max(float(video_duration or 0.0), ordered_segments[-1].end if ordered_segments else 0.0)
        for index, segment in enumerate(ordered_segments):
            style = segment.blur_style
            effective_end = ordered_segments[index + 1].start if index + 1 < len(ordered_segments) else render_until
            effective_end = max(segment.end, effective_end)
            if not style.enabled or style.width <= 0 or style.height <= 0 or effective_end <= segment.start:
                continue

            box_width = max(2, round(width * style.width / 100))
            box_height = max(2, round(height * style.height / 100))
            left = round(width * style.x / 100)
            top = round(height * style.y / 100)
            left = max(0, min(left, width - box_width))
            top = max(0, min(top, height - box_height))
            max_blur_radius = max(0, (min(box_width, box_height) // 2) - 1)
            blur_radius = max(0, min(int(style.blur), 48, max_blur_radius))
            opacity = max(0.0, min(float(style.opacity), 0.95))
            enable = f"between(t,{segment.start:.3f},{effective_end:.3f})"

            if blur_radius > 0:
                split_streams = video_stream.filter_multi_output("split")
                base_stream = split_streams[0]
                crop_source = split_streams[1]
                blurred_crop = crop_source.crop(
                    left,
                    top,
                    box_width,
                    box_height,
                ).filter("boxblur", luma_radius=blur_radius, luma_power=1)
                video_stream = ffmpeg.overlay(base_stream, blurred_crop, x=left, y=top, enable=enable)

            if opacity > 0:
                video_stream = video_stream.filter(
                    "drawbox",
                    x=left,
                    y=top,
                    w=box_width,
                    h=box_height,
                    color=f"black@{opacity:.3f}",
                    t="fill",
                    enable=enable,
                )
        return video_stream

    def _clip_timeline_to_video(
        self,
        segments: Iterable[TranscriptSegment],
        video_duration: float | None,
    ) -> list[TranscriptSegment]:
        canonical = TimelineService().from_transcript(segments)
        duration = max(0.0, float(video_duration or 0.0))
        if duration <= 0:
            return canonical

        clipped: list[TranscriptSegment] = []
        for segment in canonical:
            if segment.start >= duration:
                continue
            end = min(segment.end, duration)
            if end <= segment.start:
                continue
            clipped.append(
                segment.model_copy(
                    update={
                        "id": len(clipped),
                        "end": end,
                    }
                )
            )
        return clipped

    def _write_srt(
        self,
        segments: Iterable[TranscriptSegment],
        destination: Path,
        *,
        video_duration: float | None = None,
    ) -> None:
        lines: list[str] = []
        capcut_timeline = self._clip_timeline_to_video(segments, video_duration)
        for index, segment in enumerate(capcut_timeline, start=1):
            lines.extend(
                [
                    str(index),
                    f"{self._srt_time(segment.start)} --> {self._srt_time(segment.end)}",
                    segment.text,
                    "",
                ]
            )
        # CapCut reliably detects Vietnamese when SubRip uses a UTF-8 BOM. CRLF
        # also keeps the file compatible with its Windows importer/editor.
        content = "\r\n".join(lines)
        if content:
            content += "\r\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content.encode("utf-8-sig"))

    def _write_ass(
        self,
        segments: Iterable[DubbingScriptSegment],
        destination: Path,
        *,
        video_width: int,
        video_height: int,
        video_duration: float | None = None,
    ) -> None:
        width = max(1, int(video_width))
        height = max(1, int(video_height))
        header = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,42,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,3,0,2,20,20,20,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        lines = [*header]
        ordered_segments = sorted(segments, key=lambda item: (item.start, item.end))
        render_until = max(float(video_duration or 0.0), ordered_segments[-1].end if ordered_segments else 0.0)
        for index, segment in enumerate(ordered_segments):
            text = (segment.translated_text or segment.original_text).strip()
            effective_end = ordered_segments[index + 1].start if index + 1 < len(ordered_segments) else render_until
            effective_end = max(segment.end, effective_end)
            if not text or effective_end <= segment.start:
                continue

            style = segment.subtitle_style
            box_width_px = max(40, round(width * style.width / 100))
            box_height_px = max(24, round(height * style.height / 100))
            left_px = round(width * style.x / 100)
            top_px = round(height * style.y / 100)
            align = self._ass_alignment(style.align)
            if style.align == "left":
                pos_x = left_px
            elif style.align == "right":
                pos_x = left_px + box_width_px
            else:
                pos_x = left_px + box_width_px // 2
            pos_y = top_px + box_height_px // 2
            wrapped_text = self._wrap_ass_text(text, box_width_px=box_width_px, font_size=style.font_size)
            override = (
                "{"
                f"\\an{align}"
                f"\\pos({pos_x},{pos_y})"
                f"\\fs{style.font_size}"
                f"\\c{self._ass_color(style.color)}"
                f"\\3c{self._ass_color(style.outline_color)}"
                f"\\bord{style.outline_width}"
                "}"
            )
            lines.append(
                "Dialogue: 0,"
                f"{self._ass_time(segment.start)},"
                f"{self._ass_time(effective_end)},"
                "Default,,0,0,0,,"
                f"{override}{wrapped_text}"
            )
        destination.write_text("\n".join(lines), encoding="utf-8")

    def _normalize_segments(self, raw_segments: list[dict]) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        for index, raw in enumerate(raw_segments):
            words = [
                WordTimestamp(
                    word=str(word.get("word", "")).strip(),
                    start=float(word.get("start", raw.get("start", 0.0)) or 0.0),
                    end=float(word.get("end", raw.get("end", 0.0)) or 0.0),
                )
                for word in raw.get("words", [])
                if str(word.get("word", "")).strip()
            ]
            words = self._expand_cjk_word_timestamps(words)
            text = str(raw.get("text", "")).strip()
            if not text:
                continue

            base_segment = TranscriptSegment(
                id=index,
                start=float(raw.get("start", 0.0) or 0.0),
                end=float(raw.get("end", 0.0) or 0.0),
                text=text,
                words=words,
            )
            for split_segment in self._split_transcript_segment(base_segment):
                segments.append(split_segment.model_copy(update={"id": len(segments)}))
        return segments

    def _split_transcript_segment(self, segment: TranscriptSegment) -> list[TranscriptSegment]:
        duration = max(0.0, segment.end - segment.start)
        text_limit = self._segment_text_limit(segment.text)
        duration_limit = self._segment_duration_limit(segment.text)

        if self._has_useful_word_timestamps(segment):
            split_by_words = self._split_segment_by_words(segment)
            if split_by_words:
                return split_by_words

        has_multiple_sentences = sum(char in SENTENCE_TERMINATORS for char in segment.text) > 1
        if duration <= duration_limit and len(segment.text) <= text_limit and not has_multiple_sentences:
            return [segment]

        return self._split_segment_by_text(segment)

    def _has_useful_word_timestamps(self, segment: TranscriptSegment) -> bool:
        return any(word.end > word.start and word.end - word.start < segment.end - segment.start for word in segment.words)

    def _expand_cjk_word_timestamps(self, words: list[WordTimestamp]) -> list[WordTimestamp]:
        expanded: list[WordTimestamp] = []
        for word in words:
            token = word.word.strip()
            compact_chars = [char for char in token if self._is_compact_script_char(char)]
            if len(compact_chars) <= 1 or word.end <= word.start:
                expanded.append(word)
                continue

            pieces = [char for char in token if char.strip()]
            step = (word.end - word.start) / max(1, len(pieces))
            for index, piece in enumerate(pieces):
                piece_start = word.start + step * index
                expanded.append(
                    WordTimestamp(
                        word=piece,
                        start=piece_start,
                        end=min(word.end, piece_start + step),
                    )
                )
        return expanded

    def _split_segment_by_words(self, segment: TranscriptSegment) -> list[TranscriptSegment]:
        chunks: list[TranscriptSegment] = []
        current_words: list[WordTimestamp] = []
        text_limit = self._segment_text_limit(segment.text)
        duration_limit = self._segment_duration_limit(segment.text)
        contains_cjk = self._contains_cjk(segment.text)
        hard_text_limit = HARD_MAX_CJK_SEGMENT_CHARS if contains_cjk else HARD_MAX_SEGMENT_CHARS

        for word in segment.words:
            if current_words:
                projected_text = self._join_word_tokens([*[item.word for item in current_words], word.word])
                projected_duration = max(0.0, word.end - current_words[0].start)
                gap = max(0.0, word.start - current_words[-1].end)
                if (
                    (projected_duration >= duration_limit or len(projected_text) >= text_limit)
                    and gap >= 0.3
                ):
                    chunks.append(self._segment_from_words(segment, current_words))
                    current_words = []

            current_words.append(word)
            current_text = self._join_word_tokens([item.word for item in current_words])
            current_duration = max(0.0, current_words[-1].end - current_words[0].start)
            reached_soft_limit = current_duration >= duration_limit or len(current_text) >= text_limit
            reached_hard_limit = (
                current_duration >= HARD_MAX_SEGMENT_DURATION
                or len(current_text) >= hard_text_limit
            )
            should_close = (
                self._ends_with_terminal_punctuation(word.word)
                or (reached_soft_limit and self._ends_with_clause_punctuation(word.word))
                or reached_hard_limit
            )
            if should_close and current_duration >= MIN_SEGMENT_DURATION:
                chunks.append(self._segment_from_words(segment, current_words))
                current_words = []

        if current_words:
            chunks.append(self._segment_from_words(segment, current_words))

        return self._merge_tiny_segments(chunks)

    def _segment_from_words(self, original: TranscriptSegment, words: list[WordTimestamp]) -> TranscriptSegment:
        return TranscriptSegment(
            id=original.id,
            start=max(original.start, words[0].start),
            end=min(original.end, max(words[-1].end, words[0].start + MIN_SEGMENT_DURATION)),
            text=self._join_word_tokens([word.word for word in words]),
            words=words,
        )

    def _split_segment_by_text(self, segment: TranscriptSegment) -> list[TranscriptSegment]:
        duration = max(segment.end - segment.start, MIN_SEGMENT_DURATION)
        total_chars = max(1, len(segment.text))
        text_limit = self._segment_text_limit(segment.text)
        text_chunks = self._chunk_text(segment.text, text_limit)
        if len(text_chunks) <= 1:
            return [segment]

        chunks: list[TranscriptSegment] = []
        consumed_chars = 0
        for index, text in enumerate(text_chunks):
            clean_text = text.strip()
            if not clean_text:
                continue
            previous_chars = consumed_chars
            consumed_chars += len(clean_text)
            start = segment.start + duration * (previous_chars / total_chars)
            end = segment.end if index == len(text_chunks) - 1 else segment.start + duration * (consumed_chars / total_chars)
            chunks.append(
                TranscriptSegment(
                    id=segment.id,
                    start=start,
                    end=max(start + MIN_SEGMENT_DURATION, min(segment.end, end)),
                    text=clean_text,
                    words=[],
                )
            )
        return self._merge_tiny_segments(chunks)

    def _chunk_text(self, text: str, char_limit: int) -> list[str]:
        # Sentence punctuation is the primary boundary. Keeping each complete
        # sentence independent gives translation and TTS the same semantic
        # unit and avoids unnatural pauses inside a sentence.
        return self._text_units(text, char_limit)

    def _text_units(self, text: str, char_limit: int) -> list[str]:
        units: list[str] = []
        current = ""
        for char in text:
            current += char
            if char in SENTENCE_TERMINATORS:
                units.extend(self._hard_wrap(current.strip(), char_limit))
                current = ""
        if current.strip():
            units.extend(self._hard_wrap(current.strip(), char_limit))
        return [unit for unit in units if unit]

    def _hard_wrap(self, text: str, char_limit: int) -> list[str]:
        if len(text) <= char_limit:
            return [text]

        clause_units: list[str] = []
        current_clause = ""
        for char in text:
            current_clause += char
            if char in CLAUSE_PUNCTUATION or char in SENTENCE_TERMINATORS:
                clause_units.append(current_clause.strip())
                current_clause = ""
        if current_clause.strip():
            clause_units.append(current_clause.strip())

        if len(clause_units) > 1:
            chunks: list[str] = []
            current = ""
            for clause in clause_units:
                separator = "" if self._contains_cjk(current + clause) else " "
                candidate = f"{current}{separator}{clause}".strip() if current else clause
                if current and len(candidate) > char_limit:
                    chunks.append(current)
                    current = clause
                else:
                    current = candidate
            if current:
                chunks.append(current)
            if all(len(chunk) <= char_limit * 2 for chunk in chunks):
                return chunks

        if not self._contains_cjk(text):
            chunks: list[str] = []
            current = ""
            for word in text.split():
                candidate = f"{current} {word}".strip() if current else word
                if current and len(candidate) > char_limit:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                chunks.append(current)
            return chunks
        return [text[index : index + char_limit] for index in range(0, len(text), char_limit)]

    def _segment_text_limit(self, text: str) -> int:
        return MAX_CJK_SEGMENT_CHARS if self._contains_cjk(text) else MAX_SEGMENT_CHARS

    def _segment_duration_limit(self, text: str) -> float:
        return MAX_CJK_SEGMENT_DURATION if self._contains_cjk(text) else MAX_SEGMENT_DURATION

    def _merge_tiny_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        merged: list[TranscriptSegment] = []
        for segment in segments:
            if merged and segment.end - segment.start < MIN_SEGMENT_DURATION:
                previous = merged[-1]
                merged[-1] = TranscriptSegment(
                    id=previous.id,
                    start=previous.start,
                    end=segment.end,
                    text=self._join_text_chunks(previous.text, segment.text),
                    words=[*previous.words, *segment.words],
                )
            else:
                merged.append(segment)
        return merged

    def _join_word_tokens(self, tokens: list[str]) -> str:
        clean_tokens = [token.strip() for token in tokens if token.strip()]
        if self._contains_cjk("".join(clean_tokens)):
            return "".join(clean_tokens)
        return " ".join(clean_tokens)

    def _join_text_chunks(self, left: str, right: str) -> str:
        if self._contains_cjk(left + right):
            return f"{left}{right}"
        return f"{left} {right}".strip()

    def _contains_cjk(self, text: str) -> bool:
        return any(self._is_compact_script_char(char) for char in text)

    def _is_compact_script_char(self, char: str) -> bool:
        return (
            "\u3040" <= char <= "\u30ff"
            or "\u3400" <= char <= "\u9fff"
            or "\uac00" <= char <= "\ud7af"
        )

    def _ends_with_terminal_punctuation(self, text: str) -> bool:
        clean_text = text.strip()
        return bool(clean_text and clean_text[-1] in SENTENCE_TERMINATORS)

    def _ends_with_clause_punctuation(self, text: str) -> bool:
        clean_text = text.strip()
        return bool(clean_text and clean_text[-1] in CLAUSE_PUNCTUATION)

    def _mock_segments(self) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                id=0,
                start=0.0,
                end=3.0,
                text="Hello, this is a test transcription.",
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5),
                    WordTimestamp(word="this", start=0.8, end=1.1),
                    WordTimestamp(word="is", start=1.1, end=1.25),
                    WordTimestamp(word="a", start=1.25, end=1.35),
                    WordTimestamp(word="test", start=1.35, end=1.8),
                    WordTimestamp(word="transcription", start=1.8, end=3.0),
                ],
            )
        ]

    def _probe_duration(self, path: Path) -> float:
        return probe_duration(path, ffmpeg_module=self._ffmpeg(), logger=logger)

    def _safe_probe_duration(self, path: Path) -> float:
        try:
            return round(self._probe_duration(path), 2)
        except Exception:
            logger.exception("media_probe.duration.failed path=%s", path)
            return 0.0

    def _video_dimensions(self, path: Path) -> tuple[int, int]:
        try:
            return probe_video_dimensions(
                path,
                ffmpeg_module=self._ffmpeg(),
                logger=logger,
                default=(1920, 1080),
            )
        except Exception:
            logger.exception("media_probe.dimensions.failed path=%s", path)
            return 1920, 1080

    def _has_audio_stream(self, path: Path) -> bool:
        try:
            probe = self._ffmpeg().probe(str(path))
        except Exception as exc:
            if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 4551:
                logger.warning(
                    "media_probe.audio_probe_blocked path=%s reason=Windows Application Control; assuming audio may exist",
                    path,
                )
                return True
            logger.exception("media_probe.audio.failed path=%s", path)
            return False
        return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))

    def _atempo_filters(self, tempo: float) -> list[float]:
        values: list[float] = []
        while tempo > 2.0:
            values.append(2.0)
            tempo /= 2.0
        while tempo < 0.5:
            values.append(0.5)
            tempo /= 0.5
        values.append(round(tempo, 4))
        return values

    def _write_silent_wav(self, path: Path, duration: float, sample_rate: int = 24000) -> None:
        frames = max(1, math.ceil(duration * sample_rate))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x00" * frames)

    def _srt_time(self, seconds: float) -> str:
        millis = round(seconds * 1000)
        hours, remainder = divmod(millis, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def _ass_time(self, seconds: float) -> str:
        centis = round(max(0.0, seconds) * 100)
        hours, remainder = divmod(centis, 360_000)
        minutes, remainder = divmod(remainder, 6_000)
        secs, cs = divmod(remainder, 100)
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"

    def _ass_color(self, hex_color: str) -> str:
        clean = hex_color.strip().lstrip("#")
        if len(clean) != 6:
            clean = "FFFFFF"
        red, green, blue = clean[0:2], clean[2:4], clean[4:6]
        return f"&H00{blue}{green}{red}&"

    def _ass_alignment(self, align: str) -> int:
        if align == "left":
            return 4
        if align == "right":
            return 6
        return 5

    def _wrap_ass_text(self, text: str, *, box_width_px: int, font_size: int) -> str:
        escaped = self._ass_escape(text)
        if not escaped:
            return escaped
        if self._contains_cjk(escaped):
            chars_per_line = max(2, int(box_width_px / max(1, font_size)))
            return r"\N".join(
                escaped[index : index + chars_per_line]
                for index in range(0, len(escaped), chars_per_line)
            )

        words = escaped.split()
        if not words:
            return escaped
        chars_per_line = max(8, int(box_width_px / max(1, font_size * 0.55)))
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if current and len(candidate) > chars_per_line:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return r"\N".join(lines)

    def _ass_escape(self, text: str) -> str:
        return (
            text.replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\r\n", r"\N")
            .replace("\n", r"\N")
        )

    def _ffmpeg_filter_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            display_path = resolved.relative_to(Path.cwd().resolve())
        except ValueError:
            display_path = resolved

        escaped = str(display_path).replace("\\", "/").replace("'", r"\'")
        if ":" in escaped:
            escaped = escaped.replace(":", r"\:")
        return escaped

    def _run_ffmpeg_command(self, command, stage: str) -> None:
        try:
            command.run(capture_stdout=True, capture_stderr=True)
        except Exception as exc:
            stderr = self._ffmpeg_stderr(exc)
            logger.error("legacy_pipeline.ffmpeg.%s.error stderr=%s", stage, stderr or "<empty>")
            message = self._last_log_lines(stderr) or str(exc)
            raise RuntimeError(f"FFmpeg {stage} failed: {message}") from exc

    def _ffmpeg_stderr(self, exc: BaseException) -> str:
        stderr = getattr(exc, "stderr", None)
        if stderr is None:
            return ""
        if isinstance(stderr, bytes):
            return stderr.decode("utf-8", errors="replace").strip()
        return str(stderr).strip()

    def _last_log_lines(self, text: str, limit: int = 8) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-limit:])

    def _segment_stats(self, segments: list[TranscriptSegment]) -> dict[str, object]:
        duration = sum(max(0.0, segment.end - segment.start) for segment in segments)
        characters = sum(len(segment.text.strip()) for segment in segments)
        return {
            "segments": len(segments),
            "speech_duration": round(duration, 2),
            "characters": characters,
        }

    def _script_stats(self, segments: list[DubbingScriptSegment]) -> dict[str, object]:
        duration = sum(max(0.0, segment.end - segment.start) for segment in segments)
        characters = sum(len(segment.translated_text.strip()) for segment in segments)
        return {
            "segments": len(segments),
            "speech_duration": round(duration, 2),
            "characters": characters,
        }

    def _ffmpeg(self):
        self.dependencies.require_ffmpeg()
        try:
            import ffmpeg
        except ImportError as exc:
            raise RuntimeError("ffmpeg-python is required. Install it with `pip install ffmpeg-python`.") from exc
        return ffmpeg

    def _raise_if_cancelled(self, workspace: Workspace | None = None) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            request_id = getattr(workspace, "request_id", None)
            if request_id:
                logger.info("pipeline.cancelled request_id=%s", request_id)
            else:
                logger.info("pipeline.cancelled")
            raise PipelineCancelledError("Pipeline cancelled")

    def _acquire_gpu_lock(self, workspace: Workspace | None = None) -> None:
        while True:
            self._raise_if_cancelled(workspace)
            if self._gpu_lock.acquire(timeout=GPU_LOCK_POLL_SECONDS):
                return

    def _acquire_gpu_lock_for_stream(
        self,
        workspace: Workspace,
        *,
        phase: str,
        progress: int,
        operation: str,
    ) -> Generator[str, None, bool]:
        start = time.monotonic()
        timeout_seconds = self._env_float("AUTODUB_GPU_LOCK_WAIT_TIMEOUT", GPU_LOCK_WAIT_TIMEOUT_SECONDS)
        timeout_seconds = max(0.0, timeout_seconds)

        while True:
            self._raise_if_cancelled(workspace)
            if self._gpu_lock.acquire(timeout=GPU_LOCK_POLL_SECONDS):
                waited_seconds = time.monotonic() - start
                if waited_seconds >= 1.0:
                    logger.info(
                        "pipeline.lock.acquired_after_wait request_id=%s operation=%s waited=%.1fs",
                        workspace.request_id,
                        operation,
                        waited_seconds,
                    )
                    yield self._event(
                        "processing",
                        "Recognition engine acquired",
                        request_id=workspace.request_id,
                        phase=phase,
                        progress=progress,
                        detail=f"Waited {waited_seconds:.0f}s for previous GPU/model job, starting now",
                    )
                return True

            waited_seconds = time.monotonic() - start
            logger.info(
                "pipeline.lock.wait request_id=%s operation=%s waited=%.1fs timeout=%.1fs",
                workspace.request_id,
                operation,
                waited_seconds,
                timeout_seconds,
            )
            yield self._event(
                "processing",
                "Waiting for previous GPU/model job...",
                request_id=workspace.request_id,
                phase=phase,
                progress=progress,
                detail=f"Waiting for previous GPU/model job to release resources ({waited_seconds:.0f}s)",
            )

            if timeout_seconds and waited_seconds >= timeout_seconds:
                message = (
                    f"Backend is still busy after {timeout_seconds:.0f}s. "
                    "A previous job may be stuck in ASR/OCR; cancel it or restart the backend, then retry."
                )
                logger.warning(
                    "pipeline.lock.timeout request_id=%s operation=%s waited=%.1fs",
                    workspace.request_id,
                    operation,
                    waited_seconds,
                )
                yield self._event(
                    "error",
                    "Analyze failed",
                    request_id=workspace.request_id,
                    phase=phase,
                    progress=progress,
                    error=message,
                )
                return False

    def _env_float(self, name: str, default: float) -> float:
        raw_value = os.environ.get(name, "").strip()
        if not raw_value:
            return default
        try:
            return float(raw_value)
        except ValueError:
            logger.warning("Invalid float env %s=%r; using %.1f", name, raw_value, default)
            return default

    def _event(self, status: str, step: str, **extra: object) -> str:
        payload = {"status": status, "step": step, **extra}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
