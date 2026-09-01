from __future__ import annotations

import logging
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import PipelineConfig, TranscriptSegment
from app.services.dependency_service import DependencyService
from app.services.timeline_service import TimelineService
from app.utils.media_probe import probe_duration
from app.utils.memory import VRAMManager
from utils.model_cache import configure_model_cache
from utils.model_registry import model_registry
from utils.tts_voice import (
    encode_cloned_vieneu_voice,
    infer_stable_cloned_vieneu_audio_batch,
    infer_stable_cloned_vieneu_audio,
    infer_stable_vieneu_audio_batch,
    infer_stable_vieneu_audio,
    resolve_vieneu_voice,
)

logger = logging.getLogger(__name__)
TTS_MIN_NATURAL_STRETCH_RATIO = 0.82
TIMELINE_CONTIGUOUS_TOLERANCE_S = 0.04


def _ffmpeg():
    DependencyService().require_ffmpeg()
    try:
        import ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg-python is required. Install it with `pip install ffmpeg-python`.") from exc
    return ffmpeg


@dataclass(frozen=True)
class TTSAudioTrack:
    segment_id: int
    path: Path
    start: float
    end: float


class TTSService:
    """Generate segment-level TTS with the shared offloaded model registry."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.model = None
        self.voice: str | None = None
        self.clone_voice_reference = None

    def synthesize(
        self,
        segments: list[TranscriptSegment],
        work_dir: Path,
    ) -> list[TTSAudioTrack]:
        segments = TimelineService().from_transcript(segments)
        tts_dir = work_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        tracks: list[TTSAudioTrack] = []

        try:
            if self.config.mock_tts:
                raise RuntimeError("Mock TTS is disabled because GPU TTS is required.")
            if self.config.tts_device != "cuda":
                raise RuntimeError("CPU TTS is disabled. Use CUDA TTS only.")
            self._log_model_cache()
            DependencyService().require_cuda()

            with model_registry.acquire_vieneu(device="cuda", backend="pytorch") as model:
                self.model = model
                if self.config.voice_mode == "clone":
                    self.clone_voice_reference = encode_cloned_vieneu_voice(
                        model,
                        self.config.clone_reference_audio_path or "",
                    )
                    logger.info(
                        "tts_service.voice.clone reference=%s",
                        self.config.clone_reference_audio_path,
                    )
                else:
                    self.voice = resolve_vieneu_voice(model, self.config.voice_model)
                    logger.info(
                        "tts_service.voice.system requested=%s resolved=%s",
                        self.config.voice_model,
                        self.voice,
                    )
                batch_size = self._vieneu_batch_size()
                for batch_start in range(0, len(segments), batch_size):
                    batch_segments = segments[batch_start : batch_start + batch_size]
                    texts = [segment.text for segment in batch_segments]
                    if self.config.voice_mode == "clone":
                        if self.clone_voice_reference is None:
                            raise RuntimeError("Cloned voice was selected but its reference was not encoded.")
                        audio_values = infer_stable_cloned_vieneu_audio_batch(
                            model,
                            texts,
                            self.clone_voice_reference,
                        )
                    else:
                        if self.voice is None:
                            raise RuntimeError("System voice was selected but was not resolved.")
                        audio_values = infer_stable_vieneu_audio_batch(model, texts, self.voice)
                    if len(audio_values) != len(batch_segments):
                        raise RuntimeError(
                            f"VieNeu returned {len(audio_values)} audio values for {len(batch_segments)} segments."
                        )
                    for offset, (segment, audio) in enumerate(zip(batch_segments, audio_values)):
                        index = batch_start + offset
                        raw_path = tts_dir / f"segment_{segment.id:04d}_raw.wav"
                        final_path = tts_dir / f"segment_{segment.id:04d}.wav"
                        next_start = segments[index + 1].start if index + 1 < len(segments) else None
                        duration = self._timing_fit_duration(segment, next_start)
                        model.save(audio, str(raw_path))
                        self._fit_duration(raw_path, final_path, duration)
                        tracks.append(TTSAudioTrack(segment.id, final_path, segment.start, segment.start + duration))

            return tracks
        except Exception as exc:
            if VRAMManager.is_cuda_error(exc):
                VRAMManager.reset_after_cuda_error()
                logger.exception("CUDA failure during TTS synthesis")
                raise RuntimeError(
                    "CUDA failed during TTS synthesis. Model cache was reset and VRAM cache was cleared."
                ) from exc
            raise
        finally:
            self.model = None
            self.voice = None
            self.clone_voice_reference = None
            VRAMManager.cleanup()

    def unload(self) -> None:
        self.model = None
        self.voice = None
        self.clone_voice_reference = None
        VRAMManager.cleanup()

    def _log_model_cache(self) -> None:
        cache_paths = configure_model_cache()
        logger.info(
            "tts_service.model_cache device=%s hf_home=%s hf_hub_cache=%s",
            self.config.tts_device,
            cache_paths.hf_home,
            cache_paths.hf_hub_cache,
        )

    def _synthesize_with_model(self, text: str, destination: Path) -> None:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")

        if self.config.voice_mode == "clone":
            if self.clone_voice_reference is None:
                raise RuntimeError("Cloned voice was selected but its reference was not encoded.")
            audio = infer_stable_cloned_vieneu_audio(self.model, text, self.clone_voice_reference)
        else:
            if self.voice is None:
                raise RuntimeError("System voice was selected but was not resolved.")
            audio = infer_stable_vieneu_audio(self.model, text, self.voice)
        self.model.save(audio, str(destination))

    def _vieneu_batch_size(self) -> int:
        try:
            value = int(os.environ.get("AUTODUB_VIENEU_BATCH_SIZE", "16"))
        except ValueError:
            value = 16
        return max(1, min(32, value))

    def _timing_fit_duration(self, segment: TranscriptSegment, next_start: float | None) -> float:
        natural = max(float(segment.end) - float(segment.start), 0.1)
        if not self.config.soft_timing_fit or self.config.timing_max_drift_s <= 0 or next_start is None:
            return natural
        timeline_gap = float(next_start) - float(segment.end)
        reserved_gap = 0.0 if timeline_gap <= TIMELINE_CONTIGUOUS_TOLERANCE_S else self.config.timing_min_gap_s
        available_gap = max(0.0, timeline_gap - reserved_gap)
        return natural + min(float(self.config.timing_max_drift_s), available_gap)

    def _fit_duration(self, source: Path, destination: Path, target_duration: float) -> None:
        ffmpeg = _ffmpeg()
        current_duration = self._probe_duration(source)
        if current_duration <= 0:
            self._write_silent_wav(destination, target_duration)
            return

        natural_target = max(target_duration, 0.1)
        ratio = max(0.1, current_duration / natural_target)
        # Fill a small duration shortfall with a bounded slow-down. Padding
        # the tail is audible as a pause when adjacent cues touch exactly.
        tempo = (
            ratio
            if current_duration > natural_target
            or ratio >= TTS_MIN_NATURAL_STRETCH_RATIO
            else 1.0
        )
        tempo *= self.config.voice_speed
        if (
            self.config.soft_timing_fit
            and tempo > self.config.timing_max_atempo
            and ratio <= self.config.timing_max_atempo
        ):
            tempo = self.config.timing_max_atempo
        if tempo > 1.35:
            logger.warning(
                "tts_service.audio_fit.high_speed source=%s raw_duration=%.3f target_duration=%.3f tempo=%.3f",
                source,
                current_duration,
                natural_target,
                tempo,
            )
        stream = ffmpeg.input(str(source)).audio
        for value in self._atempo_filters(tempo):
            stream = stream.filter("atempo", value)

        stream = stream.filter("apad").filter("atrim", duration=natural_target)
        (
            ffmpeg.output(stream, str(destination), ac=1, ar="24000", format="wav")
            .overwrite_output()
            .run(quiet=True)
        )

    def _probe_duration(self, path: Path) -> float:
        return probe_duration(path, ffmpeg_module=_ffmpeg(), logger=logger)

    def _atempo_filters(self, ratio: float) -> list[float]:
        ratio = max(ratio, 0.1)
        filters: list[float] = []
        while ratio > 2.0:
            filters.append(2.0)
            ratio /= 2.0
        while ratio < 0.5:
            filters.append(0.5)
            ratio /= 0.5
        filters.append(round(ratio, 4))
        return filters

    def _write_silent_wav(self, path: Path, duration: float, sample_rate: int = 24000) -> None:
        frames = max(1, math.ceil(duration * sample_rate))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x00" * frames)

