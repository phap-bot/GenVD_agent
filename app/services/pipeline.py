from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Generator, Iterable

from app.models.schemas import DubbingScriptSegment, FlashTextTrack, PipelineConfig, TranscriptSegment, WordTimestamp
from app.services.dependency_service import DependencyService
from app.services.timeline_service import TimelineService
from app.services.translation_service import TranslationService
from app.services.streaming_pipeline import StreamingPipeline
from app.services.checkpoint_service import CheckpointStore
from app.services.audio_timing import repair_continuous_speech_gaps
from app.utils.media_probe import probe_duration, probe_video_dimensions
from app.utils.vram import VRAMManager
from app.utils.workspace import Workspace
from app.utils.cancel import PipelineCancelledError
from app.utils.video_encoder import (
    HARDWARE_ENCODERS,
    cpu_encoder_plan,
    is_hardware_encoder_runtime_error,
    select_video_encoder,
)
from utils.model_cache import configure_model_cache
from utils.model_registry import model_registry
from utils.ocr import extract_video_ocr_segments
from utils.flash_text import detect_flash_text_tracks
from utils.stt import remote_stt_enabled, transcribe_audio_remote
from utils.translation import TRANSLATION_POLICY_VERSION, generate_caption_suggestions
from utils.tts_voice import (
    ClonedVieneuVoice,
    encode_cloned_vieneu_voice,
    infer_stable_cloned_vieneu_audio_batch,
    infer_stable_cloned_vieneu_audio,
    infer_stable_vieneu_audio_batch,
    infer_stable_vieneu_audio,
    resolve_vieneu_voice,
)
from utils.language import detect_language

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
AUDIO_STEM_CACHE_VERSION = "demucs-fullband-stereo-normalized-residual-v3"
ASR_RECOGNITION_POLICY_VERSION = "asr-language-policy-v8-streaming-faster-whisper"
ACCOMPANIMENT_DEFAULT_VOLUME = 0.92
ORIGINAL_FALLBACK_VOLUME = 0.12
DEFAULT_X264_PRESET = "superfast"
DEFAULT_X264_CRF = 22
DEFAULT_FFMPEG_PROGRESS_INTERVAL_SECONDS = 15.0
DEFAULT_FFMPEG_RENDER_STALL_TIMEOUT_SECONDS = 180.0
DEFAULT_FFMPEG_RENDER_FINALIZE_TIMEOUT_SECONDS = 300.0
DEFAULT_FFMPEG_RENDER_TIMEOUT_FACTOR = 3.0
DEFAULT_FFMPEG_RENDER_MIN_TIMEOUT_SECONDS = 600.0
DEFAULT_FFMPEG_RENDER_UNKNOWN_TIMEOUT_SECONDS = 1800.0
VALID_X264_PRESETS = frozenset(
    {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}
)


def _env_value(name: str, default: str = "") -> str:
    """Read a setting from the process environment, then the local .env."""
    value = os.environ.get(name, "").strip()
    if not value:
        try:
            for raw_line in Path(".env").read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line.startswith(f"{name}="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        except OSError:
            pass
    return value or default


def _env_truthy(name: str, default: bool = False) -> bool:
    return _env_value(name, "1" if default else "").lower() in {"1", "true", "yes", "on"}

# ── TTS segment merging constants ──
TTS_MERGE_MAX_GAP = 0.5           # max silence gap (seconds) between segments to merge
TTS_MERGE_MAX_SEG_DURATION = 2.0  # only merge segments shorter than this (seconds)
TTS_MERGE_MAX_GROUP_DURATION = 6.0  # max total duration of a merged group
TTS_MERGE_MAX_CHARS = 160         # max total characters of merged text
TTS_CONTIGUOUS_MAX_GROUP_DURATION = 16.0
TTS_CONTIGUOUS_MAX_CHARS = 384
TTS_MIN_NATURAL_STRETCH_RATIO = 0.82  # do not slow very short utterances into unnatural speech
TIMELINE_CONTIGUOUS_TOLERANCE_S = 0.04  # absorb timestamp quantization at touching cue boundaries
TTS_POLICY_VERSION = "tts-contiguous-boundary-v2"
CAPTION_POLICY_VERSION = "grounded-caption-suggestions-v1"


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


@dataclass(frozen=True)
class _FFmpegMonitorConfig:
    heartbeat_seconds: float
    stall_seconds: float
    finalize_seconds: float
    hard_timeout_seconds: float


def _parse_ffmpeg_progress_time(value: str) -> float | None:
    """Parse FFmpeg's HH:MM:SS.microseconds progress value."""
    try:
        hours, minutes, seconds = value.strip().split(":", 2)
        return max(0.0, int(hours) * 3600 + int(minutes) * 60 + float(seconds))
    except (TypeError, ValueError):
        return None


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
        self.last_flash_text_tracks: list[FlashTextTrack] = []
        self.last_caption_suggestions: list[str] = []
        self.runtime_event_callback: Callable[[dict[str, object]], None] | None = None

    def set_runtime_event_callback(
        self,
        callback: Callable[[dict[str, object]], None] | None,
    ) -> None:
        """Attach a durable event sink for long blocking stages such as FFmpeg."""
        self.runtime_event_callback = callback

    def _publish_runtime_event(self, payload: dict[str, object]) -> None:
        callback = self.runtime_event_callback
        if callback is None:
            return
        try:
            callback(payload)
        except Exception:
            logger.exception("pipeline.runtime_event.publish_failed phase=%s", payload.get("phase"))

    def _checkpoint_store(self, workspace: Workspace) -> CheckpointStore:
        return CheckpointStore(
            workspace.input_video,
            root=self.config.checkpoint_root,
            enabled=self.config.checkpoint_enabled,
        )

    def _checkpoint_config(self, stage: str) -> dict[str, object]:
        """Only include settings that can change a stage's result."""
        if stage == "asr":
            return {
                "stage": stage,
                "policy_version": ASR_RECOGNITION_POLICY_VERSION,
                "asr_model": self.config.asr_model,
                "asr_engine": self.config.asr_engine,
                "auto_paraformer": _env_truthy("AUTODUB_AUTO_PARAFORMER"),
                "paraformer_model": _env_value("AUTODUB_PARAFORMER_MODEL", "paraformer-zh"),
                "paraformer_venv": _env_value("AUTODUB_PARAFORMER_VENV", ".venv-asr"),
                "whisper_model": self.config.whisper_model,
                "beam_size": self.config.whisper_beam_size,
                "batch_size": self.config.whisper_batch_size,
                "vad_filter": self.config.whisper_vad_filter,
                "compute_type": self.config.compute_type,
                "streaming": _env_truthy("AUTODUB_STREAMING_PIPELINE", True),
                "block_seconds": self._env_float("AUTODUB_ASR_BLOCK_SECONDS", 120.0),
                "block_overlap_seconds": self._env_float("AUTODUB_ASR_BLOCK_OVERLAP_SECONDS", 1.5),
                "source_language": self.config.source_language,
                "vocal_separation": self.config.vocal_separation,
                "segment_language_detection": self.config.segment_language_detection,
                "fill_speech_gaps": self.config.fill_speech_gaps,
                "short_video": self.config.short_video,
            }
        if stage == "ocr":
            return {
                "stage": stage,
                "source_language": self.config.source_language,
                "model": self.config.ocr_model,
                "interval": self.config.ocr_interval_seconds,
                "crop": self.config.ocr_crop_bottom_ratio,
                "max_frames": self.config.ocr_max_frames,
                "adaptive": self.config.ocr_adaptive,
                "scene_threshold": self.config.ocr_scene_threshold,
                "short_video": self.config.short_video,
            }
        if stage == "flash_text":
            return {
                "stage": stage,
                "policy_version": "flash-text-detector-v1",
                "enabled": self.config.flash_text_enabled,
                "mode": self.config.flash_text_mode,
                "min_confidence": self.config.flash_text_min_confidence,
                "max_duration_s": self.config.flash_text_max_duration_s,
                "top_ratio": 0.04,
                "bottom_exclusion_ratio": 0.28,
                "short_video": self.config.short_video,
            }
        if stage == "tts":
            return {
                "stage": stage,
                "policy_version": TTS_POLICY_VERSION,
                "voice_mode": self.config.voice_mode,
                "voice_model": self.config.voice_model.strip(),
                "clone_ref": self._clone_reference_cache_id(self.config.clone_reference_audio_path),
                "voice_speed": round(self.config.voice_speed, 3),
                "soft_timing_fit": bool(self.config.soft_timing_fit),
                "timing_max_drift_s": round(self.config.timing_max_drift_s, 3),
                "timing_min_gap_s": round(self.config.timing_min_gap_s, 3),
                "timing_max_atempo": round(self.config.timing_max_atempo, 3),
                "short_video": self.config.short_video,
            }
        if stage == "captions":
            return {
                "stage": stage,
                "policy_version": CAPTION_POLICY_VERSION,
                "source_language": self.config.source_language,
                "target_language": self.config.target_language,
                "provider": self.config.translation_provider,
                "model": self.config.translation_model,
                "short_video": self.config.short_video,
            }
        return {
            "stage": stage,
            "policy_version": TRANSLATION_POLICY_VERSION,
            "source_language": self.config.source_language,
            "target_language": self.config.target_language,
            "provider": self.config.translation_provider,
            "model": self.config.translation_model,
            "batch_size": self.config.translate_batch_size,
            "analysis": self.config.translate_analysis,
            "review": self.config.translate_review,
            "cps_budget": self.config.translate_cps_budget,
            "short_video": self.config.short_video,
        }

    def run(self, workspace: Workspace) -> Generator[str, None, Path]:
        output_path = workspace.output_dir / f"{workspace.request_id}_dubbed.mp4"
        output_subtitle_path = output_path.with_suffix(".srt")
        caption_suggestions: list[str] = []
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
                streaming_result = None
                if self._can_use_streaming_pipeline(workspace):
                    yield self._event(
                        "processing",
                        "Streaming ASR → translation → voice workers...",
                        phase="recognize",
                        progress=14,
                        detail="Whisper blocks are bounded at two minutes and kept in timeline order",
                    )
                    streaming_result = self._run_streaming_pipeline(workspace)
                    segments, translated_segments, chunks = streaming_result
                    # OCR fallback/hybrid modes remain authoritative. If the
                    # streamed ASR is too sparse, discard its provisional
                    # downstream work and use the established OCR path.
                    if self._should_use_ocr_fallback(segments):
                        logger.info(
                            "streaming.asr_sparse_fallback request_id=%s segments=%s",
                            workspace.request_id,
                            len(segments),
                        )
                        segments = self._source_timeline(self._run_ocr(workspace))
                        translated_segments = self._canonical_timeline(
                            self._translate_segments(segments, workspace=workspace)
                        )
                        chunks = yield from self._run_tts(translated_segments, workspace)
                else:
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

                if streaming_result is None:
                    yield self._event(
                        "processing",
                        "Translating script...",
                        phase="translate",
                        progress=46,
                        stats=source_stats,
                    )
                    translated_segments = self._canonical_timeline(self._translate_segments(segments, workspace=workspace))
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
                else:
                    yield self._event(
                        "processing",
                        "Streaming translation completed",
                        phase="translate",
                        progress=62,
                        stats=self._segment_stats(translated_segments),
                    )
                yield self._event(
                    "processing",
                    "Creating grounded caption suggestions...",
                    phase="translate",
                    progress=65,
                    detail="Using the source script and translated script as caption context",
                )
                caption_suggestions = self._caption_suggestions(
                    segments,
                    translated_segments,
                    workspace=workspace,
                )
                yield self._event(
                    "processing",
                    "Voice tracks generated",
                    phase="voice",
                    progress=84,
                    stats={"chunks": len(chunks), **self._segment_stats(translated_segments)},
                )
                if self.config.flash_text_enabled:
                    yield self._event(
                        "processing",
                        "Detecting large transient on-screen text...",
                        phase="polish",
                        progress=87,
                        detail="Scanning upper/central frame; subtitle band excluded",
                    )
                    self._run_flash_text_detection(workspace)
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
            original_vocal_path = self._separated_vocal_path(workspace) if self.config.vocal_separation else None
            self._render_video(
                video_path=workspace.input_video,
                subtitle_path=subtitle_path,
                tts_mix_path=tts_mix_path,
                output_path=output_path,
                video_duration=video_duration,
                accompaniment_path=accompaniment_path,
                original_vocal_path=original_vocal_path,
                flash_text_tracks=self.last_flash_text_tracks,
            )
            self._write_srt(translated_segments, output_subtitle_path, video_duration=video_duration)

            yield self._event(
                "success",
                "Completed",
                phase="complete",
                progress=100,
                video_url=f"/media/{output_path.name}",
                subtitle_url=f"/media/{output_subtitle_path.name}",
                caption_suggestions=caption_suggestions,
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
                if self._can_use_streaming_pipeline(workspace):
                    segments, translated, _chunks = self._run_streaming_pipeline(workspace, include_tts=False)
                    if self._should_use_ocr_fallback(segments):
                        segments = self._source_timeline(self._run_ocr(workspace))
                        translated = self._canonical_timeline(self._translate_segments(segments, workspace=workspace))
                else:
                    segments = self._source_timeline(self._extract_source_segments(workspace))
                    translated = self._canonical_timeline(self._translate_segments(segments, workspace=workspace))
                self.last_caption_suggestions = self._caption_suggestions(
                    segments,
                    translated,
                    workspace=workspace,
                )
                VRAMManager.cleanup()
                analyzed = [
                    DubbingScriptSegment(
                        id=segment.id,
                        start=segment.start,
                        end=segment.end,
                        original_text=segments[index].text if index < len(segments) else segment.text,
                        translated_text=segment.text,
                        voice_model=self.config.voice_model,
                        source_language=segment.language,
                        language_probability=segment.language_probability,
                    )
                    for index, segment in enumerate(translated)
                ]
                self._run_flash_text_detection(workspace)
                return analyzed
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
                streaming_result = None
                if self._can_use_streaming_pipeline(workspace):
                    streaming_result = self._run_streaming_pipeline(workspace, include_tts=False)
                    segments, translated, _chunks = streaming_result
                    if self._should_use_ocr_fallback(segments):
                        segments = self._source_timeline(self._run_ocr(workspace))
                        translated = self._canonical_timeline(self._translate_segments(segments, workspace=workspace))
                else:
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
                if streaming_result is None:
                    translated = self._canonical_timeline(self._translate_segments(segments, workspace=workspace))
                yield self._event(
                    "processing",
                    "Script translated",
                    phase="translate",
                    progress=92,
                    stats=self._segment_stats(translated),
                )
                yield self._event(
                    "processing",
                    "Creating grounded caption suggestions...",
                    phase="translate",
                    progress=95,
                    detail="Using the source script and translated script as caption context",
                )
                self.last_caption_suggestions = self._caption_suggestions(
                    segments,
                    translated,
                    workspace=workspace,
                )
                analyzed = [
                    DubbingScriptSegment(
                        id=segment.id,
                        start=segment.start,
                        end=segment.end,
                        original_text=segments[index].text if index < len(segments) else segment.text,
                        translated_text=segment.text,
                        voice_model=self.config.voice_model,
                        source_language=segment.language,
                        language_probability=segment.language_probability,
                    )
                    for index, segment in enumerate(translated)
                ]
                if self.config.flash_text_enabled:
                    yield self._event(
                        "processing",
                        "Detecting large transient on-screen text...",
                        request_id=workspace.request_id,
                        phase="polish",
                        progress=96,
                        detail="Only large overlay text above the subtitle band",
                    )
                self._run_flash_text_detection(workspace)
            finally:
                self._gpu_lock.release()

            yield self._event(
                "success",
                "Analyze completed",
                phase="complete",
                progress=100,
                source_video_path=f"/media/{workspace.request_id}_source.mp4",
                segments=[segment.model_dump() for segment in analyzed],
                flash_text_tracks=[track.model_dump(mode="json") for track in self.last_flash_text_tracks],
                caption_suggestions=self.last_caption_suggestions,
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
        flash_text_tracks: list[FlashTextTrack] | None = None,
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
            source_size = source_video_path.stat().st_size
            if render_source_path.is_file() and render_source_path.stat().st_size == source_size:
                logger.info(
                    "script_render.source_checkpoint_hit request_id=%s staged=%s bytes=%s",
                    workspace.request_id,
                    render_source_path,
                    source_size,
                )
            else:
                self._atomic_copy_file(source_video_path, render_source_path)
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
                checkpoint = self._checkpoint_store(workspace)
                voice_setup_payload = self._voice_setup_checkpoint_payload(
                    workspace,
                    script_segments,
                    timeline_segments,
                )
                voice_setup = checkpoint.load(
                    "voice_setup",
                    voice_setup_payload,
                )
                if isinstance(voice_setup, dict):
                    logger.info(
                        "checkpoint.hit stage=voice_setup request_id=%s mode=%s default_voice=%s segments=%s",
                        workspace.request_id,
                        self.config.voice_mode,
                        self.config.voice_model,
                        len(script_segments),
                    )
                else:
                    checkpoint_path = checkpoint.save(
                        "voice_setup",
                        voice_setup_payload,
                        {
                            "voice_mode": self.config.voice_mode,
                            "default_voice": self.config.voice_model.strip(),
                            "segment_voices": [
                                item["voice_model"] for item in voice_setup_payload["segments"]
                            ],
                        },
                    )
                    if checkpoint_path is not None:
                        logger.info(
                            "checkpoint.saved stage=voice_setup request_id=%s mode=%s default_voice=%s segments=%s",
                            workspace.request_id,
                            self.config.voice_mode,
                            self.config.voice_model,
                            len(script_segments),
                        )
                yield self._event(
                    "processing",
                    "Preparing selected voice and generating AI voice from edited script...",
                    phase="voice",
                    progress=18,
                    stats={
                        **self._script_stats(script_segments),
                        "voice_mode": self.config.voice_mode,
                        "voice_model": self.config.voice_model,
                    },
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
            original_vocal_path = self._separated_vocal_path(workspace) if self.config.vocal_separation else None
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
                original_vocal_path=original_vocal_path,
                flash_text_tracks=flash_text_tracks or [],
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
        if self.config.source_mode == "voice":
            return segments
        if self.config.source_mode == "subtitle":
            ocr_segments = self._run_ocr(workspace)
            return ocr_segments or segments
        if self.config.source_mode == "hybrid":
            ocr_segments = self._run_ocr(workspace)
            return self._merge_asr_ocr_segments(segments, ocr_segments)
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

    def _can_use_streaming_pipeline(self, workspace: Workspace) -> bool:
        """Return whether the safe block pipeline can own this request.

        OCR-only/hybrid and remote/Paraformer requests still use their
        established whole-input path. Whisper is the only local engine that
        can be windowed without changing the selected recognition model.
        """
        if not _env_truthy("AUTODUB_STREAMING_PIPELINE", True):
            return False
        if self.config.ocr_force or self.config.source_mode in {"subtitle", "hybrid"}:
            return False
        if not self._has_audio_stream(workspace.input_video):
            return False
        selected_engine = self.config.asr_engine
        if selected_engine == "auto":
            if (self.config.source_language or "").lower().startswith("zh") and _env_truthy("AUTODUB_AUTO_PARAFORMER"):
                return False
            selected_engine = "whisper"
        return selected_engine == "whisper" and not remote_stt_enabled(self.config.asr_model)

    def _run_streaming_pipeline(
        self,
        workspace: Workspace,
        *,
        include_tts: bool = True,
    ) -> tuple[list[TranscriptSegment], list[TranscriptSegment], list[AudioChunk]]:
        """Run bounded ASR/translation/TTS workers and restore source order."""
        raw_segments: list[TranscriptSegment] = []
        gpu_stage_lock = threading.Lock()
        emitted_tail: list[TranscriptSegment] = []
        pending_block: list[TranscriptSegment] | None = None

        def asr_blocks() -> Iterable[list[TranscriptSegment]]:
            nonlocal emitted_tail, pending_block
            for block in self._iter_whisper_asr_blocks(workspace, gpu_stage_lock):
                current = self._deduplicate_stream_segments(block)
                filtered: list[TranscriptSegment] = []
                for segment in current:
                    duplicate = any(
                        min(previous.end, segment.end) - max(previous.start, segment.start) > 0.15
                        and SequenceMatcher(
                            None,
                            previous.text.strip().lower(),
                            segment.text.strip().lower(),
                        ).ratio() >= 0.78
                        for previous in [*(emitted_tail[-8:]), *((pending_block or [])[-8:])]
                    )
                    if not duplicate:
                        filtered.append(segment)
                if not filtered:
                    continue
                if pending_block is None:
                    pending_block = filtered
                    continue

                # One-block look-ahead lets the speech-gap repair inspect the
                # PCM at the boundary before the previous block reaches TTS.
                repaired = repair_continuous_speech_gaps(
                    workspace.root / "source_audio.wav",
                    [*pending_block, *filtered],
                    max_gap_s=self.config.speech_gap_max_s,
                )
                completed = repaired[:-len(filtered)] if len(filtered) else repaired
                pending_block = repaired[-len(filtered):] if filtered else pending_block
                if completed:
                    raw_segments.extend(completed)
                    emitted_tail.extend(completed)
                    emitted_tail = emitted_tail[-8:]
                    yield completed

            if pending_block:
                raw_segments.extend(pending_block)
                yield pending_block
                pending_block = None

        def translate_block(block: list[TranscriptSegment]) -> list[TranscriptSegment]:
            self._raise_if_cancelled(workspace)
            return TranslationService(self.config, self.cancel_event).translate(block)

        def tts_block(block: list[TranscriptSegment]) -> list[AudioChunk]:
            self._raise_if_cancelled(workspace)
            if not include_tts:
                return []
            # The shared CUDA stage lock prevents model eviction/movement in
            # the middle of an ASR or TTS inference. Translation remains free
            # to overlap with either GPU stage.
            with gpu_stage_lock:
                iterator = self._run_tts(block, workspace)
                while True:
                    try:
                        next(iterator)
                    except StopIteration as stop:
                        return list(stop.value or [])

        result = StreamingPipeline(
            asr_blocks=asr_blocks,
            translate_block=translate_block,
            tts_block=tts_block,
            cancel_event=self.cancel_event,
            queue_size=max(1, min(8, int(self._env_float("AUTODUB_STREAM_QUEUE_SIZE", 2)))),
            translation_group_size=max(1, min(8, round(self._env_float("AUTODUB_STREAM_TRANSLATION_BLOCKS", 3.0)))),
        ).run()

        ordered_raw = self._deduplicate_stream_segments(raw_segments)
        if self.config.fill_speech_gaps:
            ordered_raw = repair_continuous_speech_gaps(
                workspace.root / "source_audio.wav",
                ordered_raw,
                max_gap_s=self.config.speech_gap_max_s,
            )
        ordered_raw = self._annotate_segment_languages(ordered_raw)
        self._validate_stream_timeline(ordered_raw, workspace.request_id)

        checkpoint = self._checkpoint_store(workspace)
        if ordered_raw:
            checkpoint.save(
                "asr",
                self._checkpoint_config("asr"),
                [item.model_dump(mode="json") for item in ordered_raw],
            )
            logger.info(
                "streaming.checkpoint.saved stage=asr request_id=%s segments=%s",
                workspace.request_id,
                len(ordered_raw),
            )

        translated = self._canonical_timeline(result.translated)
        logger.info(
            "streaming.pipeline.done request_id=%s asr_segments=%s translated_segments=%s chunks=%s",
            workspace.request_id,
            len(ordered_raw),
            len(translated),
            len(result.audio_chunks),
        )
        # Keep a strict 1:1 source/translation mapping for streamed blocks.
        # Semantic stitching is intentionally deferred here: stitching after
        # translation could produce different boundaries and mis-pair the
        # original text shown by the analyze endpoint.
        streamed_source = TimelineService().from_transcript(ordered_raw, merge_semantic=False)
        return streamed_source, translated, result.audio_chunks

    def _iter_whisper_asr_blocks(
        self,
        workspace: Workspace,
        gpu_stage_lock: threading.Lock,
    ) -> Iterable[list[TranscriptSegment]]:
        """Transcribe overlapping windows while assigning each cue once.

        Windows overlap at both edges, but ownership is based on the cue's
        original start time. Therefore a cue crossing a two-minute boundary is
        emitted whole by one block; it is never clipped or emitted twice.
        """
        checkpoint = self._checkpoint_store(workspace)
        cached = checkpoint.load("asr", self._checkpoint_config("asr"))
        if isinstance(cached, list):
            try:
                restored = [TranscriptSegment.model_validate(item) for item in cached]
                if restored:
                    self.last_source_engine = "ASR (checkpoint)"
                    logger.info(
                        "streaming.checkpoint.hit request_id=%s segments=%s",
                        workspace.request_id,
                        len(restored),
                    )
                    yield restored
                    return
            except Exception:
                logger.warning(
                    "streaming.checkpoint.invalid request_id=%s",
                    workspace.request_id,
                    exc_info=True,
                )

        original_audio_path = workspace.root / "source_audio.wav"
        ffmpeg = self._ffmpeg()
        (
            ffmpeg.input(str(workspace.input_video))
            .output(str(original_audio_path), ac=1, ar="16000", vn=None, format="wav")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        audio_path = original_audio_path
        if self.config.vocal_separation:
            self._ensure_accompaniment_audio(workspace, workspace.input_video)
            vocals_path = workspace.root / "separated" / "vocals.wav"
            if vocals_path.is_file():
                audio_path = vocals_path
        if (
            audio_path != original_audio_path
            and (self.config.source_language or "").lower().replace("_", "-").startswith("zh")
        ):
            logger.info(
                "streaming.whisper.using_original_audio request_id=%s separated=%s",
                workspace.request_id,
                audio_path,
            )
            audio_path = original_audio_path

        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError("WhisperX is required for GPU ASR. Install whisperx, then restart the backend.") from exc

        self.dependencies.require_cuda()
        audio = whisperx.load_audio(str(audio_path))
        sample_rate = 16000
        audio_frames = int(getattr(audio, "shape", [len(audio)])[0])
        audio_duration = max(0.01, audio_frames / sample_rate)
        block_seconds = max(30.0, min(300.0, self._env_float("AUTODUB_ASR_BLOCK_SECONDS", 120.0)))
        overlap_seconds = max(0.25, min(3.0, self._env_float("AUTODUB_ASR_BLOCK_OVERLAP_SECONDS", 1.5)))
        block_count = max(1, math.ceil(audio_duration / block_seconds))
        whisper_arch = self.config.asr_model if self.config.whisper_model == "auto" else self.config.whisper_model
        self.last_source_engine = f"WhisperX {whisper_arch} on cuda (streaming)"
        logger.info(
            "streaming.asr.plan request_id=%s duration=%.3f block_seconds=%.3f overlap_seconds=%.3f blocks=%s batch_size=%s beam_size=%s vad=%s",
            workspace.request_id,
            audio_duration,
            block_seconds,
            overlap_seconds,
            block_count,
            self.config.whisper_batch_size,
            self.config.whisper_beam_size,
            self.config.whisper_vad_filter,
        )

        for block_index in range(block_count):
            nominal_start = block_index * block_seconds
            nominal_end = min(audio_duration, nominal_start + block_seconds)
            decode_start = max(0.0, nominal_start - overlap_seconds)
            decode_end = min(audio_duration, nominal_end + overlap_seconds)
            first_frame = max(0, round(decode_start * sample_rate))
            last_frame = min(audio_frames, round(decode_end * sample_rate))
            block_audio = audio[first_frame:last_frame]
            if len(block_audio) == 0:
                continue

            with gpu_stage_lock:
                with model_registry.acquire_whisperx_asr(
                    whisperx,
                    whisper_arch=whisper_arch,
                    device="cuda",
                    compute_type=self.config.compute_type,
                    language=self.config.source_language,
                    beam_size=self.config.whisper_beam_size,
                ) as model:
                    result = self._transcribe_whisper(model, block_audio)
                    current_segments = result.get("segments", [])
                    if self._asr_needs_retry(current_segments):
                        retry_result = self._transcribe_whisper(model, block_audio, cautious=True)
                        if self._asr_suspicion_score(retry_result.get("segments", [])) < self._asr_suspicion_score(current_segments):
                            result = retry_result
                        else:
                            logger.warning(
                                "streaming.asr.block_rejected request_id=%s block=%s reason=repetition_loop",
                                workspace.request_id,
                                block_index + 1,
                            )
                            result = {"segments": [], "language": result.get("language")}

            offset_segments = self._offset_whisper_segments(result.get("segments", []), decode_start)
            owned_segments = [
                raw
                for raw in offset_segments
                if nominal_start - 0.001 <= float(raw.get("start", 0.0) or 0.0) < nominal_end - 0.000001
            ]
            normalized = self._normalize_segments(owned_segments)
            if normalized:
                logger.info(
                    "streaming.asr.block_done request_id=%s block=%s/%s start=%.3f end=%.3f segments=%s",
                    workspace.request_id,
                    block_index + 1,
                    block_count,
                    nominal_start,
                    nominal_end,
                    len(normalized),
                )
                yield normalized

        VRAMManager.cleanup()

    def _offset_whisper_segments(
        self,
        raw_segments: list[dict[str, Any]],
        offset_seconds: float,
    ) -> list[dict[str, Any]]:
        offset = max(0.0, float(offset_seconds))
        offset_segments: list[dict[str, Any]] = []
        for raw in raw_segments or []:
            copied = dict(raw)
            start = max(0.0, float(raw.get("start", 0.0) or 0.0) + offset)
            end = max(start, float(raw.get("end", start) or start) + offset)
            copied["start"] = start
            copied["end"] = end
            copied["words"] = [
                {
                    **word,
                    "start": max(0.0, float(word.get("start", raw.get("start", 0.0)) or 0.0) + offset),
                    "end": max(0.0, float(word.get("end", raw.get("end", 0.0)) or 0.0) + offset),
                }
                for word in (raw.get("words") or [])
            ]
            offset_segments.append(copied)
        return offset_segments

    def _deduplicate_stream_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        ordered = sorted(segments, key=lambda item: (item.start, item.end, item.id))
        deduplicated: list[TranscriptSegment] = []
        for segment in ordered:
            if not deduplicated:
                deduplicated.append(segment)
                continue
            previous = deduplicated[-1]
            overlap = min(previous.end, segment.end) - max(previous.start, segment.start)
            previous_duration = max(0.01, previous.end - previous.start)
            segment_duration = max(0.01, segment.end - segment.start)
            text_similarity = SequenceMatcher(None, previous.text.strip().lower(), segment.text.strip().lower()).ratio()
            same_boundary_cue = overlap > 0.15 and text_similarity >= 0.78
            if same_boundary_cue:
                chosen = previous if len(previous.text.strip()) >= len(segment.text.strip()) else segment
                merged = chosen.model_copy(
                    update={
                        "start": min(previous.start, segment.start),
                        "end": max(previous.end, segment.end),
                        "words": chosen.words or previous.words or segment.words,
                    }
                )
                deduplicated[-1] = merged
                continue
            if segment.start < previous.end and overlap / min(previous_duration, segment_duration) > 0.6:
                logger.warning(
                    "streaming.timeline.overlap_kept previous=(%.3f,%.3f) current=(%.3f,%.3f)",
                    previous.start,
                    previous.end,
                    segment.start,
                    segment.end,
                )
            deduplicated.append(segment)
        return [item.model_copy(update={"id": index}) for index, item in enumerate(deduplicated)]

    def _validate_stream_timeline(self, segments: list[TranscriptSegment], request_id: str) -> None:
        previous_end = 0.0
        for index, segment in enumerate(sorted(segments, key=lambda item: (item.start, item.end))):
            if segment.start < previous_end - 0.001:
                logger.warning(
                    "streaming.timeline.overlap request_id=%s index=%s start=%.3f previous_end=%.3f",
                    request_id,
                    index,
                    segment.start,
                    previous_end,
                )
            gap = segment.start - previous_end
            if gap > self.config.speech_gap_max_s:
                logger.info(
                    "streaming.timeline.silence request_id=%s index=%s gap=%.3f max_configured=%.3f",
                    request_id,
                    index,
                    gap,
                    self.config.speech_gap_max_s,
                )
            previous_end = max(previous_end, segment.end)

    def _should_use_ocr_fallback(self, segments: list[TranscriptSegment]) -> bool:
        if not self.config.ocr_fallback:
            return False
        text_chars = sum(len(segment.text.strip()) for segment in segments)
        speech_duration = sum(max(0.0, segment.end - segment.start) for segment in segments)
        return not segments or text_chars < 8 or speech_duration < 0.5

    def _run_ocr(self, workspace: Workspace) -> list[TranscriptSegment]:
        self._raise_if_cancelled(workspace)
        checkpoint = self._checkpoint_store(workspace)
        checkpoint_payload = self._checkpoint_config("ocr")
        cached = checkpoint.load("ocr", checkpoint_payload)
        if isinstance(cached, list):
            try:
                restored = [TranscriptSegment.model_validate(item) for item in cached]
                if restored:
                    self.last_source_engine = "OCR (checkpoint)"
                    logger.info("checkpoint.hit stage=ocr request_id=%s segments=%s", workspace.request_id, len(restored))
                    return restored
            except Exception:
                logger.warning("checkpoint.invalid stage=ocr request_id=%s", workspace.request_id, exc_info=True)
        self.last_source_engine = "OCR"
        ocr_segments = extract_video_ocr_segments(
            workspace.input_video,
            source_language=self.config.source_language,
            model=self.config.ocr_model or self.config.translation_model,
            interval_seconds=self.config.ocr_interval_seconds,
            crop_bottom_ratio=self.config.ocr_crop_bottom_ratio,
            max_frames=self.config.ocr_max_frames,
            adaptive=self.config.ocr_adaptive,
            scene_threshold=self.config.ocr_scene_threshold,
            cancel_event=self.cancel_event,
        )
        normalized = [
            TranscriptSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=[],
                language=detect_language(segment.text, fallback=self.config.source_language)[0],
                language_probability=detect_language(segment.text, fallback=self.config.source_language)[1],
            )
            for segment in ocr_segments
        ]
        if normalized:
            checkpoint.save("ocr", checkpoint_payload, [item.model_dump(mode="json") for item in normalized])
            logger.info("checkpoint.saved stage=ocr request_id=%s segments=%s", workspace.request_id, len(normalized))
        return normalized

    def _run_flash_text_detection(self, workspace: Workspace) -> list[FlashTextTrack]:
        """Detect large transient overlay text without touching subtitle OCR."""
        self.last_flash_text_tracks = []
        if not self.config.flash_text_enabled:
            return []

        self._raise_if_cancelled(workspace)
        checkpoint = self._checkpoint_store(workspace)
        checkpoint_payload = self._checkpoint_config("flash_text")
        cached = checkpoint.load("flash_text", checkpoint_payload)
        if isinstance(cached, list):
            try:
                restored = [FlashTextTrack.model_validate(item) for item in cached]
                self.last_flash_text_tracks = restored
                logger.info(
                    "checkpoint.hit stage=flash_text request_id=%s tracks=%s",
                    workspace.request_id,
                    len(restored),
                )
                return restored
            except Exception:
                logger.warning("checkpoint.invalid stage=flash_text request_id=%s", workspace.request_id, exc_info=True)

        tracks = detect_flash_text_tracks(
            workspace.input_video,
            mode=self.config.flash_text_mode,
            min_confidence=self.config.flash_text_min_confidence,
            max_duration_s=self.config.flash_text_max_duration_s,
            top_ratio=0.04,
            bottom_exclusion_ratio=0.28,
            cancel_event=self.cancel_event,
        )
        self.last_flash_text_tracks = tracks
        checkpoint.save(
            "flash_text",
            checkpoint_payload,
            [track.model_dump(mode="json") for track in tracks],
        )
        logger.info(
            "flash_text.detected request_id=%s tracks=%s mode=%s",
            workspace.request_id,
            len(tracks),
            self.config.flash_text_mode,
        )
        return tracks

    def _merge_asr_ocr_segments(
        self,
        asr_segments: list[TranscriptSegment],
        ocr_segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """Combine speech and hard-subtitle timelines while removing duplicates."""
        if not asr_segments:
            return ocr_segments
        if not ocr_segments:
            return asr_segments

        merged = list(asr_segments)
        for ocr in ocr_segments:
            overlapping = [
                item
                for item in merged
                if min(item.end, ocr.end) - max(item.start, ocr.start) >= -0.15
            ]
            if not overlapping:
                merged.append(ocr)
                continue

            # If the subtitle is the same utterance, ASR supplies better timing.
            # If it is materially different, retain it as a separate cue so a
            # hard-sub-only line is not silently discarded.
            if any(self._text_similarity(item.text, ocr.text) >= 0.78 for item in overlapping):
                continue
            if all(len(item.text.strip()) < 8 for item in overlapping):
                merged.append(ocr)

        return TimelineService().from_transcript(merged)

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        normalize = lambda value: re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", "", value.lower())
        left_key = normalize(left)
        right_key = normalize(right)
        if not left_key or not right_key:
            return 0.0
        return SequenceMatcher(None, left_key, right_key).ratio()

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
                self._materialize_background_stem(cache_acc_path, acc_path)
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
                        # Demucs must always receive the original-quality mix.
                        # HQ_BACKGROUND controls the final mix only; allowing it
                        # to downsample source separation permanently harms the
                        # reusable stem cache and vocal-removal quality.
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
            requested_device = os.environ.get("AUTODUB_DEMUCS_DEVICE", "auto").strip().lower()
            if requested_device not in {"auto", "cpu", "cuda"}:
                logger.warning("vocal_separation.invalid_device value=%s fallback=auto", requested_device)
                requested_device = "auto"
            cuda_available = bool(self.dependencies.cuda_status().get("available"))
            if requested_device == "auto":
                device = "cuda" if cuda_available else "cpu"
            elif requested_device == "cuda" and not cuda_available:
                logger.warning(
                    "vocal_separation.cuda_unavailable fallback=cpu "
                    "hint=install_a_cuda_enabled_pytorch_wheel"
                )
                device = "cpu"
            else:
                device = requested_device
            logger.info("vocal_separation.ensure_start device=%s path=%s", device, source_audio)
            sep_result = VocalSeparationService().separate(
                audio_path=source_audio,
                output_dir=sep_dir,
                device=device,
                cancel_event=self.cancel_event,
            )
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
            self._materialize_background_stem(sep_result.accompaniment_path, acc_path)
            return acc_path
        except PipelineCancelledError:
            raise
        except Exception as exc:
            logger.exception("vocal_separation.ensure_failed error=%s", exc)
            raise RuntimeError(
                "Vocal separation is enabled but failed; render stopped to prevent original voice bleed. "
                f"{exc}"
            ) from exc

    def _separated_vocal_path(self, workspace: Workspace) -> Path | None:
        """Return the workspace vocal stem when it is available for final mixing."""
        vocals_path = workspace.root / "separated" / "vocals.wav"
        if vocals_path.is_file() and vocals_path.stat().st_size > 0:
            return vocals_path
        return None

    def _materialize_background_stem(self, source_path: Path, destination_path: Path) -> None:
        """Keep cached Demucs stems lossless while honoring final-mix quality."""
        if self.config.hq_background:
            shutil.copy2(source_path, destination_path)
            return
        (
            self._ffmpeg()
            .input(str(source_path))
            .output(
                str(destination_path),
                ac=1,
                ar="16000",
                acodec="pcm_s16le",
                format="wav",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

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
        checkpoint = self._checkpoint_store(workspace)
        checkpoint_payload = self._checkpoint_config("asr")
        cached = checkpoint.load("asr", checkpoint_payload)
        if isinstance(cached, list):
            try:
                restored = [TranscriptSegment.model_validate(item) for item in cached]
                if restored:
                    self.last_source_engine = "ASR (checkpoint)"
                    logger.info("checkpoint.hit stage=asr request_id=%s segments=%s", workspace.request_id, len(restored))
                    return restored
            except Exception:
                logger.warning("checkpoint.invalid stage=asr request_id=%s", workspace.request_id, exc_info=True)

        def finish(raw_segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
            normalized = self._normalize_segments(raw_segments) if raw_segments and isinstance(raw_segments[0], dict) else raw_segments
            if self.config.fill_speech_gaps:
                normalized = repair_continuous_speech_gaps(
                    workspace.root / "source_audio.wav",
                    normalized,
                    max_gap_s=self.config.speech_gap_max_s,
                )
            normalized = self._annotate_segment_languages(normalized)
            if normalized:
                checkpoint.save("asr", checkpoint_payload, [item.model_dump(mode="json") for item in normalized])
                logger.info("checkpoint.saved stage=asr request_id=%s segments=%s", workspace.request_id, len(normalized))
            return normalized

        original_audio_path = workspace.root / "source_audio.wav"
        audio_path = original_audio_path
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
            selected_engine = self.config.asr_engine
            if selected_engine == "auto":
                selected_engine = "paraformer" if (self.config.source_language or "").lower().startswith("zh") and _env_truthy("AUTODUB_AUTO_PARAFORMER") else "whisper"
            if selected_engine == "paraformer" and audio_path != original_audio_path:
                # Paraformer is more accurate on the original mixed track;
                # short Demucs stems can distort Chinese phonemes. Keep the
                # separated stem for render/background mixing, not ASR.
                logger.info(
                    "asr.paraformer.using_original_audio request_id=%s separated=%s",
                    workspace.request_id,
                    audio_path,
                )
                audio_path = original_audio_path
            elif (
                selected_engine == "whisper"
                and (self.config.source_language or "").lower().startswith("zh")
                and audio_path != original_audio_path
            ):
                # The separated stem is intentionally optimized for mixing;
                # it can remove consonant energy that Whisper needs for Chinese
                # phonemes. Use the untouched track for recognition as well.
                logger.info(
                    "asr.whisper.using_original_audio request_id=%s separated=%s",
                    workspace.request_id,
                    audio_path,
                )
                audio_path = original_audio_path
            if selected_engine == "paraformer":
                try:
                    from utils.paraformer import transcribe as paraformer_transcribe

                    self.last_source_engine = "Paraformer"
                    return finish(self._normalize_segments(paraformer_transcribe(
                        audio_path,
                        language=self.config.source_language,
                        model_id=_env_value("AUTODUB_PARAFORMER_MODEL", "paraformer-zh"),
                    )))
                except Exception:
                    if self.config.asr_engine == "paraformer":
                        raise
                    logger.warning("paraformer.auto_fallback_to_whisper request_id=%s", workspace.request_id, exc_info=True)

            if remote_stt_enabled(self.config.asr_model):
                self.last_source_engine = f"remote STT ({self.config.asr_model})"
                logger.info(
                    "legacy_pipeline.asr.remote.start request_id=%s model=%s language=%s audio=%s",
                    workspace.request_id,
                    self.config.asr_model,
                    self.config.source_language or "auto",
                    audio_path,
                )
                return finish(self._normalize_segments(
                    transcribe_audio_remote(
                        audio_path,
                        source_language=self.config.source_language,
                        model=self.config.asr_model,
                    )
                ))

            whisper_arch = self.config.asr_model if self.config.whisper_model == "auto" else self.config.whisper_model
            try:
                import whisperx
            except ImportError as exc:
                raise RuntimeError("WhisperX is required for GPU ASR. Install whisperx, then restart the backend.") from exc

            self.dependencies.require_cuda()
            device = "cuda"
            self.last_source_engine = f"WhisperX {whisper_arch} on {device}"
            logger.info(
                "legacy_pipeline.asr.model.load request_id=%s model=%s compute_type=%s device=%s language=%s cache=%s",
                workspace.request_id,
                whisper_arch,
                self.config.compute_type,
                device,
                self.config.source_language or "auto",
                MODEL_CACHE_PATHS.whisperx_asr_cache,
            )
            audio = whisperx.load_audio(str(audio_path))
            force_paraformer_fallback = False
            with model_registry.acquire_whisperx_asr(
                whisperx,
                whisper_arch=whisper_arch,
                device=device,
                compute_type=self.config.compute_type,
                language=self.config.source_language,
                beam_size=self.config.whisper_beam_size,
            ) as model:
                result = self._transcribe_whisper(model, audio)

                # Whisper can enter a repetition loop on quiet/music-heavy
                # clips (for example "phone number ..." or "000,000 ...").
                # Retry without conditioning on previous text before the bad
                # transcript is allowed into translation.
                current_segments = result.get("segments", [])
                if self._asr_needs_retry(current_segments):
                    retry_audio_path = original_audio_path
                    retry_audio = whisperx.load_audio(str(retry_audio_path))
                    retry_result = self._transcribe_whisper(model, retry_audio, cautious=True)
                    current_score = self._asr_suspicion_score(current_segments)
                    retry_score = self._asr_suspicion_score(retry_result.get("segments", []))
                    logger.warning(
                        "asr.suspect_transcript request_id=%s source_language=%s separated=%s current_score=%s retry_score=%s",
                        workspace.request_id,
                        self.config.source_language or "auto",
                        audio_path != original_audio_path,
                        current_score,
                        retry_score,
                    )
                    if retry_score < current_score:
                        result = retry_result
                        audio = retry_audio
                        audio_path = original_audio_path
                        force_paraformer_fallback = retry_score >= 2
                    else:
                        logger.warning(
                            "asr.suspect_transcript.kept request_id=%s reason=retry_not_better",
                            workspace.request_id,
                        )

                # If Whisper still produces a clearly non-Chinese/repeated
                # transcript, use the installed Chinese Paraformer as a
                # quality fallback. This path is restricted to `asr_engine=auto`
                # so an explicit Whisper choice remains respected.
                if (
                    self.config.asr_engine == "auto"
                    and (audio_path != original_audio_path or force_paraformer_fallback or self._asr_needs_retry(result.get("segments", [])))
                    and (
                        (self.config.source_language or "").lower().replace("_", "-").startswith("zh")
                        or (self.config.default_source_language or "").lower().replace("_", "-").startswith("zh")
                        or str(result.get("language") or "").lower().startswith("zh")
                    )
                    and self._asr_needs_retry(result.get("segments", []))
                ):
                    try:
                        from utils.paraformer import transcribe as paraformer_transcribe

                        paraformer_segments = paraformer_transcribe(
                            original_audio_path,
                            language=self.config.source_language,
                            model_id=_env_value("AUTODUB_PARAFORMER_MODEL", "paraformer-zh"),
                        )
                        paraformer_score = self._asr_suspicion_score(paraformer_segments)
                        whisper_score = self._asr_suspicion_score(result.get("segments", []))
                        logger.info(
                            "asr.paraformer_fallback request_id=%s whisper_score=%s paraformer_score=%s",
                            workspace.request_id,
                            whisper_score,
                            paraformer_score,
                        )
                        if paraformer_segments and paraformer_score < whisper_score:
                            result = {"segments": paraformer_segments, "language": self.config.source_language or "zh"}
                            audio = whisperx.load_audio(str(original_audio_path))
                            audio_path = original_audio_path
                            self.last_source_engine = "Paraformer fallback"
                    except Exception as exc:
                        logger.warning(
                            "asr.paraformer_fallback_failed request_id=%s error=%s",
                            workspace.request_id,
                            exc,
                            exc_info=True,
                        )

                # Never translate a known repetition loop. An empty ASR
                # result is safer than publishing fabricated subtitles; the
                # normal OCR fallback may still provide text when enabled.
                if self._asr_needs_retry(result.get("segments", [])):
                    logger.error(
                        "asr.suspect_transcript.rejected request_id=%s reason=repetition_loop",
                        workspace.request_id,
                    )
                    result = {"segments": [], "language": self.config.source_language or result.get("language")}

            VRAMManager.cleanup()

            if not self.config.word_timestamps:
                return finish(self._normalize_segments(result.get("segments", [])))

            language_code = result.get("language") or self.config.source_language or "en"
            logger.info(
                "asr.language_resolved request_id=%s requested=%s model_detected=%s segments=%s",
                workspace.request_id,
                self.config.source_language or "auto",
                language_code,
                len(result.get("segments", [])),
            )
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
                    return finish(self._normalize_segments(aligned.get("segments", [])))
                except Exception as exc:
                    logger.warning(
                        "legacy_pipeline.asr.align.failed request_id=%s language=%s error=%s, falling back to ASR segments",
                        workspace.request_id,
                        language_code,
                        exc,
                    )

            return finish(self._normalize_segments(raw_segs))
        finally:
            VRAMManager.cleanup()

    def _translate_segments(self, segments: list[TranscriptSegment], *, workspace: Workspace | None = None) -> list[TranscriptSegment]:
        self._raise_if_cancelled()
        if workspace is not None:
            checkpoint = self._checkpoint_store(workspace)
            payload = {
                **self._checkpoint_config("translation"),
                "segments": [segment.model_dump(mode="json") for segment in segments],
            }
            cached = checkpoint.load("translation", payload)
            if isinstance(cached, list):
                try:
                    restored = [TranscriptSegment.model_validate(item) for item in cached]
                    if len(restored) == len(segments):
                        logger.info("checkpoint.hit stage=translation request_id=%s segments=%s", workspace.request_id, len(restored))
                        return restored
                except Exception:
                    logger.warning("checkpoint.invalid stage=translation request_id=%s", workspace.request_id, exc_info=True)
            translated = TranslationService(self.config, self.cancel_event).translate(segments)
            checkpoint.save("translation", payload, [item.model_dump(mode="json") for item in translated])
            logger.info("checkpoint.saved stage=translation request_id=%s segments=%s", workspace.request_id, len(translated))
            return translated
        return TranslationService(self.config, self.cancel_event).translate(segments)

    def _caption_suggestions(
        self,
        source_segments: list[TranscriptSegment],
        translated_segments: list[TranscriptSegment],
        *,
        workspace: Workspace | None = None,
    ) -> list[str]:
        """Generate/cached grounded captions from the completed translation."""
        script = [
            {
                "source": source.text,
                "translated": translated_segments[index].text
                if index < len(translated_segments)
                else source.text,
            }
            for index, source in enumerate(source_segments)
            if source.text.strip()
        ]
        if not script:
            return []

        payload = {
            **self._checkpoint_config("captions"),
            "script": script,
        }
        checkpoint = self._checkpoint_store(workspace) if workspace is not None else None
        if checkpoint is not None:
            cached = checkpoint.load("captions", payload)
            if isinstance(cached, list):
                restored = [str(item).strip() for item in cached if str(item).strip()]
                if restored:
                    logger.info(
                        "checkpoint.hit stage=captions request_id=%s suggestions=%s",
                        workspace.request_id,
                        len(restored),
                    )
                    self.last_caption_suggestions = restored[:3]
                    return self.last_caption_suggestions

        self._raise_if_cancelled(workspace)
        suggestions = generate_caption_suggestions(
            script,
            target_language=self.config.target_language,
            source_language=self.config.source_language,
            provider=self.config.translation_provider,
            model=self.config.translation_model,
            cancel_event=self.cancel_event,
            max_items=3,
        )
        self.last_caption_suggestions = suggestions[:3]
        if checkpoint is not None and self.last_caption_suggestions:
            checkpoint.save("captions", payload, self.last_caption_suggestions)
            logger.info(
                "checkpoint.saved stage=captions request_id=%s suggestions=%s",
                workspace.request_id,
                len(self.last_caption_suggestions),
            )
        return self.last_caption_suggestions

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
                clone_voice_source: ClonedVieneuVoice | None = None
                system_voice_source: str | None = None
                if self.config.voice_mode == "clone":
                    clone_voice_source = encode_cloned_vieneu_voice(
                        model,
                        self.config.clone_reference_audio_path or "",
                    )
                    logger.info(
                        "legacy_pipeline.tts.voice.clone request_id=%s reference=%s",
                        workspace.request_id,
                        self.config.clone_reference_audio_path,
                    )
                else:
                    system_voice_source = resolve_vieneu_voice(model, self.config.voice_model)
                    logger.info(
                        "legacy_pipeline.tts.voice.system request_id=%s requested=%s resolved=%s",
                        workspace.request_id,
                        self.config.voice_model,
                        system_voice_source,
                    )

                total_segments = max(1, len(segments))
                batch_size = self._vieneu_batch_size()
                for batch_start in range(0, len(segments), batch_size):
                    batch_segments = segments[batch_start : batch_start + batch_size]
                    self._raise_if_cancelled(workspace)
                    texts = [segment.text for segment in batch_segments]
                    if self.config.voice_mode == "clone":
                        if clone_voice_source is None:
                            raise RuntimeError("Clone voice reference was not prepared.")
                        audio_values = infer_stable_cloned_vieneu_audio_batch(model, texts, clone_voice_source)
                    else:
                        if system_voice_source is None:
                            raise RuntimeError("System voice was not resolved.")
                        audio_values = infer_stable_vieneu_audio_batch(model, texts, system_voice_source)
                    if len(audio_values) != len(batch_segments):
                        raise RuntimeError(
                            f"VieNeu returned {len(audio_values)} audio values for {len(batch_segments)} segments."
                        )

                    for offset, (segment, audio) in enumerate(zip(batch_segments, audio_values)):
                        index = batch_start + offset
                        self._raise_if_cancelled(workspace)
                        raw_path = workspace.chunks_dir / f"{segment.id:04d}_raw.wav"
                        final_path = workspace.chunks_dir / f"{segment.id:04d}.wav"
                        next_start = segments[index + 1].start if index + 1 < len(segments) else None
                        duration = self._timing_fit_duration(segment.start, segment.end, next_start)
                        logger.info(
                            "legacy_pipeline.tts.segment.done request_id=%s segment_id=%s index=%s total=%s mode=%s chars=%s duration=%.3f batch_size=%s",
                            workspace.request_id,
                            segment.id,
                            index + 1,
                            total_segments,
                            self.config.voice_mode,
                            len(segment.text),
                            duration,
                            len(batch_segments),
                        )
                        model.save(audio, str(raw_path))
                        self._fit_audio_duration(raw_path, final_path, duration)
                        chunks.append(AudioChunk(segment.id, final_path, segment.start, segment.start + duration))
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
                                "batch_size": len(batch_segments),
                            },
                        )

            return chunks
        finally:
            VRAMManager.cleanup()

    def _vieneu_batch_size(self) -> int:
        try:
            value = int(_env_value("AUTODUB_VIENEU_BATCH_SIZE", "16"))
        except ValueError:
            value = 16
        return max(1, min(32, value))

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
                "fit_version": 4,
                "policy_version": TTS_POLICY_VERSION,
                "text": text.strip(),
                "voice_mode": voice_mode,
                "voice_model": voice_model,
                "clone_ref": self._clone_reference_cache_id(clone_ref),
                "duration": round(duration, 3),
                "voice_speed": round(self.config.voice_speed, 3),
                "soft_timing_fit": bool(self.config.soft_timing_fit),
                "timing_max_drift_s": round(self.config.timing_max_drift_s, 3),
                "timing_min_gap_s": round(self.config.timing_min_gap_s, 3),
                "timing_max_atempo": round(self.config.timing_max_atempo, 3),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _timing_fit_duration(
        self,
        start: float,
        end: float,
        next_start: float | None = None,
    ) -> float:
        """Return the safe speech window for one cue.

        A translated sentence may need a little more room than the source
        cue.  When soft fitting is enabled, consume only the silence before
        the next cue (and never more than ``timing_max_drift_s``), leaving
        ``timing_min_gap_s`` untouched.  This lets TTS keep a natural rate
        instead of aggressively compressing every long translation.
        """
        natural = max(float(end) - float(start), 0.1)
        if not self.config.soft_timing_fit or self.config.timing_max_drift_s <= 0 or next_start is None:
            return natural
        timeline_gap = float(next_start) - float(end)
        # A rounded/quantized timestamp can leave a few milliseconds between
        # cues that are semantically continuous.  Do not reserve the user
        # configured minimum gap for that boundary; otherwise the padded tail
        # of the previous TTS file is heard as an artificial pause.
        reserved_gap = (
            0.0
            if timeline_gap <= TIMELINE_CONTIGUOUS_TOLERANCE_S
            else self.config.timing_min_gap_s
        )
        available_gap = max(0.0, timeline_gap - reserved_gap)
        extension = min(float(self.config.timing_max_drift_s), available_gap)
        return natural + extension

    def _voice_setup_checkpoint_payload(
        self,
        workspace: Workspace,
        script_segments: list[DubbingScriptSegment],
        timeline_segments: list[TranscriptSegment],
    ) -> dict[str, object]:
        """Build the durable identity of the UI-selected voice setup.

        The render job already persists the full request.  This additional
        stage makes the voice choice explicit and content-addressed, so a
        resumed worker cannot silently reuse a setup from another voice,
        clone reference, timing policy, or edited script.
        """
        ordered_script = [
            item
            for item in sorted(script_segments, key=lambda item: (item.start, item.end, item.id))
            if (item.translated_text or item.original_text).strip()
        ]
        ordered_timeline = sorted(timeline_segments, key=lambda item: (item.start, item.end, item.id))
        segment_payload: list[dict[str, object]] = []
        for index, timeline in enumerate(ordered_timeline):
            script = ordered_script[index] if index < len(ordered_script) else None
            segment_payload.append(
                {
                    "id": timeline.id,
                    "start": round(timeline.start, 6),
                    "end": round(timeline.end, 6),
                    "text": timeline.text,
                    "voice_model": (
                        script.voice_model.strip()
                        if script is not None and script.voice_model.strip()
                        else self.config.voice_model.strip()
                    ),
                }
            )
        return {
            **self._checkpoint_config("tts"),
            "request_id": workspace.request_id,
            "segments": segment_payload,
        }

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
        next_start = None
        if index + 1 < len(voice_segments):
            next_start = voice_segments[index + 1].start
        duration = self._timing_fit_duration(segment.start, segment.end, next_start)
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

    def _tts_group_cache_key(self, group: _TTSGroup, next_start: float | None = None) -> str:
        """Cache key for a merged TTS group."""
        duration = self._timing_fit_duration(group.start, group.end, next_start)
        return self._tts_cache_key(
            group.text,
            self.config.voice_mode,
            group.voice_model,
            self.config.clone_reference_audio_path,
            duration,
        )

    def _tts_group_cache_lookup(self, group: _TTSGroup, next_start: float | None = None) -> Path | None:
        """Return the cache path if a fitted WAV already exists for this group."""
        cache_key = self._tts_group_cache_key(group, next_start)
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
            merged_text = (
                "".join(cur_texts)
                if any(self._contains_cjk(text) for text in cur_texts)
                else " ".join(cur_texts)
            )
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

            same_voice = _voice_key(i) == _voice_key(cur_indices[0])
            unfinished_sentence = not self._ends_with_terminal_punctuation(prev_seg.text)
            regular_merge = (
                seg_dur <= TTS_MERGE_MAX_SEG_DURATION
                and gap <= TTS_MERGE_MAX_GAP
                and merged_dur <= TTS_MERGE_MAX_GROUP_DURATION
                and merged_text_len <= TTS_MERGE_MAX_CHARS
                and unfinished_sentence
                # The segments already in the group must also be short.
                and all(
                    timeline_segments[j].end - timeline_segments[j].start <= TTS_MERGE_MAX_SEG_DURATION
                    for j in cur_indices
                )
            )
            # Long edited cues can still be one continuous narration unit.
            # Pair only truly touching cues and keep the model input bounded;
            # this removes the artificial seam without merging separate shots.
            contiguous_merge = (
                len(cur_indices) == 1
                and abs(gap) <= TIMELINE_CONTIGUOUS_TOLERANCE_S
                and unfinished_sentence
                and merged_dur <= TTS_CONTIGUOUS_MAX_GROUP_DURATION
                and merged_text_len <= TTS_CONTIGUOUS_MAX_CHARS
            )
            can_merge = same_voice and (regular_merge or contiguous_merge)

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
        # Keep the voice list aligned with TimelineService.from_script(),
        # which drops empty cues.  Otherwise one blank edited row shifts every
        # following segment's selected voice by one position.
        voice_segments = [
            item
            for item in sorted(script_segments, key=lambda item: (item.start, item.end, item.id))
            if (item.translated_text or item.original_text).strip()
        ]

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
            self._tts_group_cache_lookup(
                group,
                tts_groups[index + 1].start if index + 1 < len(tts_groups) else None,
            ) is not None
            for index, group in enumerate(tts_groups)
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
                next_start = tts_groups[gi + 1].start if gi + 1 < len(tts_groups) else None
                duration = self._timing_fit_duration(group.start, group.end, next_start)
                cached = self._tts_group_cache_lookup(group, next_start)
                shutil.copy2(str(cached), str(final_path))
                chunks.append(AudioChunk(group.first_segment_id, final_path, group.start, group.start + duration))
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
                clone_voice_reference: ClonedVieneuVoice | None = None
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

                character_voice_map: dict[str, str] = {}

                cache_hits = 0
                pending: list[tuple[int, _TTSGroup, Path, Path, float, str | None]] = []
                pending_voice: str | None = None

                def flush_pending() -> Generator[str, None, None]:
                    nonlocal pending, pending_voice
                    if not pending:
                        return
                    self._raise_if_cancelled(workspace)
                    texts = [item[1].text for item in pending]
                    if self.config.voice_mode == "clone":
                        if clone_voice_reference is None:
                            raise RuntimeError("Clone voice reference was not prepared.")
                        audio_values = infer_stable_cloned_vieneu_audio_batch(
                            model,
                            texts,
                            clone_voice_reference,
                        )
                    else:
                        if pending_voice is None:
                            raise RuntimeError("System voice batch was not resolved.")
                        audio_values = infer_stable_vieneu_audio_batch(model, texts, pending_voice)
                    if len(audio_values) != len(pending):
                        raise RuntimeError(
                            f"VieNeu returned {len(audio_values)} audio values for {len(pending)} texts."
                        )

                    batch = pending
                    pending = []
                    pending_voice = None
                    for (gi, group, final_path, raw_path, duration, _voice), audio in zip(batch, audio_values):
                        self._raise_if_cancelled(workspace)
                        model.save(audio, str(raw_path))
                        self._fit_audio_duration(raw_path, final_path, duration)
                        cache_key = self._tts_group_cache_key(
                            group,
                            tts_groups[gi + 1].start if gi + 1 < total_groups else None,
                        )
                        try:
                            cache_dest = self._tts_cache_dir() / f"{cache_key}.wav"
                            self._atomic_copy_file(final_path, cache_dest)
                        except Exception:
                            logger.warning(
                                "tts_cache.save_failed segment_id=%s cache_key=%s",
                                group.first_segment_id,
                                cache_key,
                            )
                        chunks.append(AudioChunk(group.first_segment_id, final_path, group.start, group.start + duration))
                        progress = min(72, 18 + round(((gi + 1) / max(1, total_groups)) * 54))
                        segs_label = f"({group.segment_count} segs)" if group.segment_count > 1 else ""
                        yield self._event(
                            "processing",
                            f"Đã tạo voice {gi + 1}/{total_groups} {segs_label}".strip(),
                            phase="voice",
                            progress=progress,
                            stats={
                                "chunks": len(chunks),
                                "segments": total_original,
                                "groups": total_groups,
                                "segment_id": group.first_segment_id,
                                "batch_size": len(batch),
                            },
                        )

                for gi, group in enumerate(tts_groups):
                    self._raise_if_cancelled(workspace)
                    final_path = workspace.chunks_dir / f"{group.first_segment_id:04d}.wav"
                    next_start = tts_groups[gi + 1].start if gi + 1 < len(tts_groups) else None
                    duration = self._timing_fit_duration(group.start, group.end, next_start)

                    # ── Cache hit: copy and skip inference ──
                    cached = self._tts_group_cache_lookup(group, next_start)
                    if cached is not None:
                        yield from flush_pending()
                        shutil.copy2(str(cached), str(final_path))
                        chunks.append(AudioChunk(group.first_segment_id, final_path, group.start, group.start + duration))
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

                    # Keep one voice per VieNeu batch. Adjacent groups usually
                    # share a voice, so this reduces GPU forward passes without
                    # mixing speaker embeddings or changing timeline order.
                    if self.config.voice_mode == "clone":
                        current_voice = None
                    else:
                        character_key = group.voice_model
                        if not character_key:
                            raise ValueError(f"System voice is missing for segment {group.first_segment_id}.")
                        current_voice = character_voice_map.setdefault(
                            character_key,
                            resolve_vieneu_voice(model, character_key),
                        )
                    if pending and (
                        pending_voice != current_voice
                        or len(pending) >= self._vieneu_batch_size()
                    ):
                        yield from flush_pending()

                    # ── Cache miss: queue for one batched VieNeu inference ──
                    raw_path = workspace.chunks_dir / f"{group.first_segment_id:04d}_raw.wav"
                    logger.info(
                        "legacy_pipeline.tts_script.group.queued request_id=%s group=%s/%s segment_id=%s segs=%s mode=%s chars=%s duration=%.3f",
                        workspace.request_id,
                        gi + 1,
                        total_groups,
                        group.first_segment_id,
                        group.segment_count,
                        self.config.voice_mode,
                        len(group.text),
                        duration,
                    )
                    pending.append((gi, group, final_path, raw_path, duration, current_voice))

                yield from flush_pending()

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

    @staticmethod
    def _atomic_copy_file(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _fit_audio_duration(self, source: Path, destination: Path, target_duration: float) -> None:
        if self.config.voice_speed == 1.0 and self._fit_audio_duration_with_numpy(source, destination, target_duration):
            return

        ffmpeg = self._ffmpeg()
        current_duration = self._probe_duration(source)
        if current_duration <= 0:
            self._write_silent_wav(destination, target_duration)
            return

        try:
            natural_target = max(target_duration, 0.1)
            ratio = max(0.1, current_duration / natural_target)
            # A small shortfall is normally the model's natural tail/trailing
            # silence.  Slow the complete utterance slightly to fill the cue
            # instead of padding its end with silence.  Very short output is
            # left untouched; stretching it would sound unnatural.
            tempo = (
                ratio
                if current_duration > natural_target
                or ratio >= TTS_MIN_NATURAL_STRETCH_RATIO
                else 1.0
            )
            tempo *= self.config.voice_speed
            if self.config.soft_timing_fit and tempo > self.config.timing_max_atempo and ratio <= self.config.timing_max_atempo:
                tempo = self.config.timing_max_atempo
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

            if current_duration < natural_target:
                ratio = current_duration / natural_target
                # Let the FFmpeg path apply a bounded natural slow-down.  The
                # old fast path padded every short file, which produced an
                # audible tail at the exact boundary of adjacent cues.
                if ratio >= TTS_MIN_NATURAL_STRETCH_RATIO:
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
            natural_speed = (
                speed
                if len(audio) > target_ms
                or speed >= TTS_MIN_NATURAL_STRETCH_RATIO
                else 1.0
            )
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
        original_vocal_path: Path | None = None,
        flash_text_tracks: Iterable[FlashTextTrack] | None = None,
    ) -> None:
        ffmpeg = self._ffmpeg()
        video_input = ffmpeg.input(str(video_path))
        tts_input = ffmpeg.input(str(tts_mix_path))
        script_segment_list = list(script_segments or [])
        flash_text_track_list = list(flash_text_tracks or [])
        has_segment_blur = any(
            segment.blur_style.enabled
            and segment.blur_style.width > 0
            and segment.blur_style.height > 0
            and segment.end > segment.start
            for segment in script_segment_list
        )
        has_flash_blur = self.config.flash_text_enabled and any(
            track.enabled and bool(track.boxes) and track.end > track.start
            for track in flash_text_track_list
        )
        has_speed_filter = not math.isclose(self.config.video_speed, 1.0, abs_tol=1e-6)
        has_visual_filters = bool(
            self.config.burn_subtitles
            or has_segment_blur
            or has_flash_blur
            or has_speed_filter
        )

        video_stream = video_input.video
        if has_flash_blur:
            if not video_width or not video_height:
                video_width, video_height = self._video_dimensions(video_path)
            video_stream = self._apply_flash_text_blur(
                ffmpeg,
                video_stream,
                flash_text_track_list,
                video_width=video_width or 0,
                video_height=video_height or 0,
            )
        if has_speed_filter:
            # 0.82 makes the picture slower and leaves roughly 22% more room
            # for Vietnamese speech; 1.0 keeps the original duration.
            video_stream = video_stream.filter("setpts", f"{1.0 / self.config.video_speed:.6f}*PTS")
        if has_segment_blur:
            video_stream = self._apply_blur_boxes(
                ffmpeg,
                video_stream,
                script_segment_list,
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
        has_original_vocal = (
            original_vocal_path
            and original_vocal_path.is_file()
            and original_vocal_path.stat().st_size > 0
        )
        if self.config.vocal_separation and not has_acc:
            raise RuntimeError(
                "Vocal separation is enabled but no accompaniment track is available; "
                "render stopped to prevent original voice bleed."
            )
        if self.config.vocal_separation and self.config.original_vocal_gain > 0 and not has_original_vocal:
            raise RuntimeError(
                "Original vocal level was requested but no separated vocal track is available."
            )
        if has_acc:
            bg_vol = self.config.background_volume if self.config.background_volume > 0 else ACCOMPANIMENT_DEFAULT_VOLUME
            bg_vol = round(bg_vol * self.config.accompaniment_gain, 6)
            logger.info(
                "legacy_pipeline.render.accompaniment_ducked path=%s volume=%.2f vocal_gain=%.2f ratio=12 release_ms=320",
                accompaniment_path,
                bg_vol,
                self.config.original_vocal_gain if self.config.vocal_separation else 0.0,
            )
            bg_audio = ffmpeg.input(str(accompaniment_path)).audio.filter("volume", bg_vol)
            source_audio = bg_audio
            if self.config.vocal_separation and has_original_vocal and self.config.original_vocal_gain > 0:
                original_vocal = ffmpeg.input(str(original_vocal_path)).audio.filter(
                    "volume", self.config.original_vocal_gain
                )
                source_audio = ffmpeg.filter(
                    [bg_audio, original_vocal],
                    "amix",
                    inputs=2,
                    duration="first",
                    dropout_transition=0,
                    normalize=0,
                )
            # Keep the original ambience near full level between speech, then
            # duck it while dubbed speech is active. This masks residual vocal
            # bleed without flattening music and environmental sound globally.
            background_audio = ffmpeg.filter(
                [source_audio, tts_sidechain],
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
        if self.config.voice_postprocess:
            mixed_audio = mixed_audio.filter(
                "loudnorm",
                I=self.config.voice_target_lufs,
                TP=-1.0,
                LRA=7.0,
            )
            logger.info(
                "legacy_pipeline.render.voice_postprocess lufs=%s bg_duck_db=%s hq_background=%s",
                self.config.voice_target_lufs,
                self.config.bg_duck_voice_db,
                self.config.hq_background,
            )
        mixed_audio = mixed_audio.filter("alimiter", limit=0.95)

        x264_preset = os.environ.get("AUTODUB_X264_PRESET", DEFAULT_X264_PRESET).strip().lower()
        if x264_preset not in VALID_X264_PRESETS:
            logger.warning(
                "legacy_pipeline.render.invalid_x264_preset value=%s fallback=%s",
                x264_preset,
                DEFAULT_X264_PRESET,
            )
            x264_preset = DEFAULT_X264_PRESET
        try:
            x264_crf = int(os.environ.get("AUTODUB_X264_CRF", str(DEFAULT_X264_CRF)))
        except ValueError:
            x264_crf = DEFAULT_X264_CRF
        x264_crf = min(51, max(0, x264_crf))
        encoder_plan = select_video_encoder(
            ffmpeg_binary=self.dependencies.resolve_binary("ffmpeg") or "ffmpeg",
            has_visual_filters=has_visual_filters,
            x264_preset=x264_preset,
            x264_crf=x264_crf,
        )
        logger.info(
            "legacy_pipeline.render.encoder codec=%s reason=%s subtitles=%s segment_blur=%s flash_blur=%s speed_filter=%s",
            encoder_plan.codec,
            encoder_plan.reason,
            self.config.burn_subtitles,
            has_segment_blur,
            has_flash_blur,
            has_speed_filter,
        )

        def build_output(plan):
            return ffmpeg.output(
                video_stream,
                mixed_audio,
                str(output_path),
                acodec="aac",
                movflags="+faststart",
                **plan.options,
            ).overwrite_output()

        command = build_output(encoder_plan)
        try:
            self._run_ffmpeg_command(
                command,
                "render",
                expected_duration=video_duration,
                encoder_codec=encoder_plan.codec,
            )
        except PipelineCancelledError:
            raise
        except RuntimeError as exc:
            if (
                encoder_plan.codec in HARDWARE_ENCODERS
                and is_hardware_encoder_runtime_error(exc, encoder_plan.codec)
            ):
                fallback_plan = cpu_encoder_plan(x264_preset=x264_preset, x264_crf=x264_crf)
                logger.warning(
                    "legacy_pipeline.render.encoder_fallback from=%s to=libx264 error=%s",
                    encoder_plan.codec,
                    exc,
                )
                self._run_ffmpeg_command(
                    build_output(fallback_plan),
                    "render",
                    expected_duration=video_duration,
                    encoder_codec=fallback_plan.codec,
                )
            elif encoder_plan.codec == "copy":
                fallback_plan = select_video_encoder(
                    ffmpeg_binary=self.dependencies.resolve_binary("ffmpeg") or "ffmpeg",
                    has_visual_filters=True,
                    x264_preset=x264_preset,
                    x264_crf=x264_crf,
                )
                logger.warning(
                    "legacy_pipeline.render.stream_copy_fallback codec=%s error=%s",
                    fallback_plan.codec,
                    exc,
                )
                self._run_ffmpeg_command(
                    build_output(fallback_plan),
                    "render",
                    expected_duration=video_duration,
                    encoder_codec=fallback_plan.codec,
                )
            else:
                raise

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
                blur_radius = max(0, int(blur_radius))
                chroma_radius = min(blur_radius, 38)

                blurred_crop = crop_source.crop(
                    left,
                    top,
                    box_width,
                    box_height,
                ).filter(
                    "boxblur",
                    luma_radius=blur_radius,
                    luma_power=1,
                    chroma_radius=chroma_radius,
                    chroma_power=1,
                )
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

    def _apply_flash_text_blur(
        self,
        ffmpeg,
        video_stream,
        tracks: Iterable[FlashTextTrack],
        *,
        video_width: int,
        video_height: int,
    ):
        """Blur only detected flash-text tracks for their exact time window."""
        width = max(1, int(video_width))
        height = max(1, int(video_height))
        for track in sorted(tracks, key=lambda item: (item.start, item.id)):
            if not track.enabled or not track.boxes or track.end <= track.start:
                continue

            left, top, box_width, box_height = self._flash_track_bounds(track, width, height)
            if box_width <= 1 or box_height <= 1:
                continue
            blur_radius = max(0, min(int(track.blur), 64, (min(box_width, box_height) // 2) - 1))
            enable = f"between(t,{track.start:.6f},{track.end:.6f})"
            split_streams = video_stream.filter_multi_output("split")
            base_stream = split_streams[0]
            crop_source = split_streams[1]
            blurred_crop = crop_source.crop(left, top, box_width, box_height)
            if blur_radius > 0:
                blurred_crop = blurred_crop.filter(
                    "boxblur",
                    luma_radius=blur_radius,
                    luma_power=1,
                    chroma_radius=min(blur_radius, 38),
                    chroma_power=1,
            )
            video_stream = ffmpeg.overlay(
                base_stream,
                blurred_crop,
                x=left,
                y=top,
                enable=enable,
            )
            logger.info(
                "flash_text.render track=%s start=%.3f end=%.3f box=%s,%s,%s,%s blur=%s",
                track.id,
                track.start,
                track.end,
                left,
                top,
                box_width,
                box_height,
                blur_radius,
            )
        return video_stream

    @staticmethod
    def _flash_track_bounds(track: FlashTextTrack, width: int, height: int) -> tuple[int, int, int, int]:
        """Use a padded union so animated glyphs remain covered without per-frame filters."""
        min_x = min(box.x for box in track.boxes)
        min_y = min(box.y for box in track.boxes)
        max_x = max(box.x + box.width for box in track.boxes)
        max_y = max(box.y + box.height for box in track.boxes)
        padding_x = (max_x - min_x) * max(0.0, min(0.5, track.padding))
        padding_y = (max_y - min_y) * max(0.0, min(0.5, track.padding))
        left = max(0.0, min(100.0, min_x - padding_x))
        top = max(0.0, min(100.0, min_y - padding_y))
        right = max(left, min(100.0, max_x + padding_x))
        bottom = max(top, min(100.0, max_y + padding_y))
        pixel_left = max(0, min(width - 2, round(width * left / 100.0)))
        pixel_top = max(0, min(height - 2, round(height * top / 100.0)))
        pixel_right = max(pixel_left + 2, min(width, round(width * right / 100.0)))
        pixel_bottom = max(pixel_top + 2, min(height, round(height * bottom / 100.0)))
        return pixel_left, pixel_top, pixel_right - pixel_left, pixel_bottom - pixel_top

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

    def _normalize_segments(self, raw_segments: list[dict[str, Any]]) -> list[TranscriptSegment]:
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
                language=raw.get("language"),
                language_probability=(float(raw.get("language_probability")) if raw.get("language_probability") is not None else None),
            )
            for split_segment in self._split_transcript_segment(base_segment):
                segments.append(split_segment.model_copy(update={"id": len(segments)}))
        return segments

    def _annotate_segment_languages(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if not self.config.segment_language_detection:
            return segments
        fallback = self.config.source_language or self.config.default_source_language
        annotated: list[TranscriptSegment] = []
        for segment in segments:
            detected, probability = detect_language(segment.text, fallback=fallback)
            # A user-selected language remains authoritative when the detector
            # only saw a short Latin fragment or an isolated proper name.
            language = detected
            confidence = probability
            if self.config.source_language:
                requested = self.config.source_language.lower().replace("_", "-")
                # An explicit source choice is the decode contract. Keep
                # script-level detection for Japanese/Korean mixed speech,
                # but do not let a short/garbled Latin hallucination relabel a
                # Chinese timeline as English.
                if requested.startswith("zh") and detected not in {"ja", "ko"}:
                    language = self.config.source_language
                elif probability < 0.7:
                    language = self.config.source_language
                confidence = probability
            annotated.append(segment.model_copy(update={"language": language, "language_probability": confidence}))
        return annotated

    def _transcribe_whisper(self, model: Any, audio: Any, *, cautious: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.config.whisper_batch_size,
            "language": self.config.source_language,
            "vad_filter": self.config.whisper_vad_filter,
            "no_speech_threshold": 0.4,
            "condition_on_previous_text": False,
        }
        if cautious:
            kwargs.update(
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.6,
                vad_filter=self.config.whisper_vad_filter,
            )
        try:
            return model.transcribe(audio, **kwargs)
        except TypeError:
            # Older/fake WhisperX wrappers may not expose the guard knobs;
            # retain compatibility while still using the normal decode path.
            for key in ("vad_filter", "no_speech_threshold", "temperature", "condition_on_previous_text"):
                kwargs.pop(key, None)
            return model.transcribe(audio, **kwargs)

    def _asr_needs_retry(self, raw_segments: list[dict[str, Any]]) -> bool:
        """Detect repetition/number flooding before translation.

        This is intentionally language-agnostic: the screenshot failure was
        English/number hallucination while the source was Chinese, but the
        same Whisper loop can occur for any language.
        """
        return self._asr_suspicion_score(raw_segments) >= 2

    @classmethod
    def _asr_suspicion_score(cls, raw_segments: list[dict[str, Any]]) -> int:
        score = 0
        for raw in raw_segments or []:
            text = str(raw.get("text") or "").strip()
            if len(text) < 8:
                continue
            han_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
            latin_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
            if han_count == 0 and len(latin_words) >= 5:
                score += 1
            # Repeated 2/3-word n-grams are a strong hallucination signal and
            # much safer to flag than dropping a legitimate short sentence.
            for width in (2, 3):
                if len(latin_words) < width * 2:
                    continue
                ngrams = [tuple(latin_words[i : i + width]) for i in range(len(latin_words) - width + 1)]
                if len(ngrams) != len(set(ngrams)):
                    score += 2
                    break
            numeric_tokens = re.findall(r"\d+(?:[.,]\d+)*", text)
            all_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*", text)
            if len(all_tokens) >= 8 and len(numeric_tokens) / len(all_tokens) >= 0.6:
                score += 2
            if len(latin_words) >= 8 and len(set(latin_words)) / len(latin_words) <= 0.45:
                score += 1
        return score

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
            language=original.language,
            language_probability=original.language_probability,
        )

    def _split_segment_by_text(self, segment: TranscriptSegment) -> list[TranscriptSegment]:
        duration = max(segment.end - segment.start, MIN_SEGMENT_DURATION)
        total_chars = max(1, len(segment.text))
        text_limit = self._segment_text_limit(segment.text)
        # Paraformer local checkpoints can return one transcript without word
        # timestamps. In that case a 120-second paragraph must still be split
        # into speech-sized units; otherwise every TTS call inherits a giant
        # duration and translation loses sentence-level timing.
        if not segment.words and duration > self._segment_duration_limit(segment.text):
            target_chars = int(total_chars * self._segment_duration_limit(segment.text) / duration)
            text_limit = min(text_limit, max(16, target_chars))
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
                    language=segment.language,
                    language_probability=segment.language_probability,
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
                    language=previous.language if previous.language == segment.language else None,
                    language_probability=previous.language_probability,
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

    def _run_ffmpeg_command(
        self,
        command,
        stage: str,
        *,
        expected_duration: float | None = None,
        encoder_codec: str | None = None,
    ) -> None:
        if stage == "render":
            self._run_monitored_ffmpeg(
                command,
                stage,
                expected_duration=expected_duration,
                encoder_codec=encoder_codec,
            )
            return
        try:
            command.run(capture_stdout=True, capture_stderr=True)
        except Exception as exc:
            stderr = self._ffmpeg_stderr(exc)
            logger.error("legacy_pipeline.ffmpeg.%s.error stderr=%s", stage, stderr or "<empty>")
            message = self._last_log_lines(stderr) or str(exc)
            raise RuntimeError(f"FFmpeg {stage} failed: {message}") from exc

    def _run_monitored_ffmpeg(
        self,
        command,
        stage: str,
        *,
        expected_duration: float | None,
        encoder_codec: str | None,
    ) -> None:
        """Run final FFmpeg rendering with progress, cancellation and watchdogs."""
        config = self._ffmpeg_monitor_config(expected_duration)
        arguments = list(command.compile())
        # These are global FFmpeg options and must precede every input/output.
        # Inserting them after the executable avoids version-dependent parsing
        # when ffmpeg-python appends global_args after the output filename.
        arguments[1:1] = ["-progress", "pipe:1", "-nostats", "-nostdin"]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise RuntimeError(f"FFmpeg {stage} failed to start: {exc}") from exc

        messages: queue.Queue[tuple[str, str | None]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=80)

        def read_stream(name: str, stream) -> None:
            try:
                for raw_line in iter(stream.readline, ""):
                    messages.put((name, raw_line.rstrip("\r\n")))
            except (OSError, ValueError) as exc:
                messages.put(("stderr", f"FFmpeg {name} monitor error: {exc}"))
            finally:
                messages.put((name, None))

        readers = [
            threading.Thread(
                target=read_stream,
                args=("progress", process.stdout),
                name=f"ffmpeg-{stage}-progress",
                daemon=True,
            ),
            threading.Thread(
                target=read_stream,
                args=("stderr", process.stderr),
                name=f"ffmpeg-{stage}-stderr",
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        started_at = time.monotonic()
        last_progress_at = started_at
        last_heartbeat_at = started_at
        media_seconds = 0.0
        last_frame = 0
        progress_values: dict[str, str] = {}

        def consume_message(source: str, line: str | None) -> None:
            nonlocal media_seconds, last_frame, last_progress_at
            if line is None:
                return
            if source == "stderr":
                if line.strip():
                    stderr_tail.append(line.strip())
                return
            if "=" not in line:
                return
            key, value = line.split("=", 1)
            progress_values[key] = value
            if key == "out_time":
                parsed = _parse_ffmpeg_progress_time(value)
                if parsed is not None and parsed > media_seconds + 0.01:
                    media_seconds = parsed
                    last_progress_at = time.monotonic()
            elif key == "frame":
                try:
                    frame = int(value)
                except ValueError:
                    return
                if frame > last_frame:
                    last_frame = frame
                    last_progress_at = time.monotonic()

        logger.info(
            "legacy_pipeline.ffmpeg.%s.started pid=%s encoder=%s expected_duration=%.2f hard_timeout=%.1f stall_timeout=%.1f",
            stage,
            process.pid,
            encoder_codec or "unknown",
            max(0.0, float(expected_duration or 0.0)),
            config.hard_timeout_seconds,
            config.stall_seconds,
        )

        try:
            while True:
                if process.poll() is not None:
                    break
                if self.cancel_event is not None and self.cancel_event.is_set():
                    self._terminate_ffmpeg_process(process)
                    logger.info("legacy_pipeline.ffmpeg.%s.cancelled pid=%s", stage, process.pid)
                    raise PipelineCancelledError("Pipeline cancelled during FFmpeg render")

                now = time.monotonic()
                elapsed = now - started_at
                if elapsed >= config.hard_timeout_seconds:
                    self._terminate_ffmpeg_process(process)
                    raise RuntimeError(
                        f"FFmpeg {stage} exceeded its {config.hard_timeout_seconds:.0f}s hard timeout "
                        f"at media time {media_seconds:.1f}s"
                    )

                near_end = bool(
                    expected_duration
                    and expected_duration > 0
                    and media_seconds >= max(0.0, expected_duration - 1.0)
                )
                stall_limit = config.finalize_seconds if near_end else config.stall_seconds
                if now - last_progress_at >= stall_limit:
                    self._terminate_ffmpeg_process(process)
                    tail = self._last_log_lines("\n".join(stderr_tail), limit=4)
                    detail = f"; last FFmpeg output: {tail}" if tail else ""
                    raise RuntimeError(
                        f"FFmpeg {stage} stalled for {stall_limit:.0f}s "
                        f"at media time {media_seconds:.1f}s{detail}"
                    )

                try:
                    source, line = messages.get(timeout=0.5)
                    consume_message(source, line)
                except queue.Empty:
                    pass

                now = time.monotonic()
                if now - last_heartbeat_at >= config.heartbeat_seconds:
                    percent = (
                        min(100.0, media_seconds * 100.0 / expected_duration)
                        if expected_duration and expected_duration > 0
                        else None
                    )
                    logger.info(
                        "legacy_pipeline.ffmpeg.%s.progress pid=%s elapsed=%.1f media_time=%.1f percent=%s frame=%s fps=%s speed=%s",
                        stage,
                        process.pid,
                        now - started_at,
                        media_seconds,
                        f"{percent:.1f}" if percent is not None else "unknown",
                        progress_values.get("frame", "unknown"),
                        progress_values.get("fps", "unknown"),
                        progress_values.get("speed", "unknown"),
                    )
                    ui_progress = (
                        min(99, 96 + int(percent * 3.0 / 100.0))
                        if percent is not None
                        else 96
                    )
                    self._publish_runtime_event(
                        {
                            "status": "processing",
                            "step": "Encoding final video...",
                            "phase": "render",
                            "progress": ui_progress,
                            "detail": (
                                f"FFmpeg {encoder_codec or 'encoder'} encoded {percent:.1f}% ({media_seconds:.1f}s)"
                                if percent is not None
                                else f"FFmpeg {encoder_codec or 'encoder'} is active ({now - started_at:.0f}s elapsed)"
                            ),
                            "stats": {
                                "render_media_seconds": round(media_seconds, 2),
                                "render_percent": round(percent, 1) if percent is not None else None,
                                "render_elapsed_seconds": round(now - started_at, 1),
                                "video_encoder": encoder_codec or "unknown",
                            },
                        }
                    )
                    last_heartbeat_at = now

            for reader in readers:
                reader.join(timeout=2.0)
            while True:
                try:
                    source, line = messages.get_nowait()
                except queue.Empty:
                    break
                consume_message(source, line)

            return_code = process.wait(timeout=5.0)
            if return_code != 0:
                stderr = "\n".join(stderr_tail)
                logger.error("legacy_pipeline.ffmpeg.%s.error stderr=%s", stage, stderr or "<empty>")
                message = self._last_log_lines(stderr) or f"exit code {return_code}"
                raise RuntimeError(f"FFmpeg {stage} failed: {message}")

            logger.info(
                "legacy_pipeline.ffmpeg.%s.completed pid=%s elapsed=%.1f media_time=%.1f",
                stage,
                process.pid,
                time.monotonic() - started_at,
                media_seconds,
            )
        finally:
            if process.poll() is None:
                self._terminate_ffmpeg_process(process)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            for reader in readers:
                if reader.is_alive():
                    reader.join(timeout=1.0)

    def _ffmpeg_monitor_config(self, expected_duration: float | None) -> _FFmpegMonitorConfig:
        heartbeat = max(
            2.0,
            min(
                60.0,
                self._env_float(
                    "AUTODUB_FFMPEG_PROGRESS_INTERVAL",
                    DEFAULT_FFMPEG_PROGRESS_INTERVAL_SECONDS,
                ),
            ),
        )
        stall = max(
            30.0,
            min(
                1800.0,
                self._env_float(
                    "AUTODUB_FFMPEG_RENDER_STALL_TIMEOUT",
                    DEFAULT_FFMPEG_RENDER_STALL_TIMEOUT_SECONDS,
                ),
            ),
        )
        finalize = max(
            stall,
            min(
                1800.0,
                self._env_float(
                    "AUTODUB_FFMPEG_RENDER_FINALIZE_TIMEOUT",
                    DEFAULT_FFMPEG_RENDER_FINALIZE_TIMEOUT_SECONDS,
                ),
            ),
        )
        timeout_factor = max(
            1.0,
            min(
                10.0,
                self._env_float(
                    "AUTODUB_FFMPEG_RENDER_TIMEOUT_FACTOR",
                    DEFAULT_FFMPEG_RENDER_TIMEOUT_FACTOR,
                ),
            ),
        )
        minimum_timeout = max(
            60.0,
            min(
                7200.0,
                self._env_float(
                    "AUTODUB_FFMPEG_RENDER_MIN_TIMEOUT",
                    DEFAULT_FFMPEG_RENDER_MIN_TIMEOUT_SECONDS,
                ),
            ),
        )
        if expected_duration and expected_duration > 0:
            hard_timeout = max(minimum_timeout, expected_duration * timeout_factor)
        else:
            hard_timeout = max(
                minimum_timeout,
                min(
                    7200.0,
                    self._env_float(
                        "AUTODUB_FFMPEG_RENDER_UNKNOWN_TIMEOUT",
                        DEFAULT_FFMPEG_RENDER_UNKNOWN_TIMEOUT_SECONDS,
                    ),
                ),
            )
        return _FFmpegMonitorConfig(
            heartbeat_seconds=heartbeat,
            stall_seconds=stall,
            finalize_seconds=finalize,
            hard_timeout_seconds=hard_timeout,
        )

    def _terminate_ffmpeg_process(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

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
