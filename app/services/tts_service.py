from __future__ import annotations

import logging
import math
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
from utils.tts_voice import infer_stable_vieneu_audio, resolve_vieneu_voice

logger = logging.getLogger(__name__)


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
                for segment in segments:
                    raw_path = tts_dir / f"segment_{segment.id:04d}_raw.wav"
                    final_path = tts_dir / f"segment_{segment.id:04d}.wav"
                    duration = max(segment.end - segment.start, 0.1)
                    self._write_silent_wav(raw_path, duration)
                    self._fit_duration(raw_path, final_path, duration)
                    tracks.append(TTSAudioTrack(segment.id, final_path, segment.start, segment.end))
                return tracks

            self._log_model_cache()
            if self.config.tts_device == "cuda":
                DependencyService().require_cuda()

            backend = "pytorch" if self.config.tts_device == "cuda" else "onnx"
            with model_registry.acquire_vieneu(device=self.config.tts_device, backend=backend) as model:
                self.model = model
                self.voice = resolve_vieneu_voice(model, self.config.voice_model)
                logger.info(
                    "tts_service.voice.resolved requested=%s resolved=%s",
                    self.config.voice_model,
                    self.voice,
                )
                for segment in segments:
                    raw_path = tts_dir / f"segment_{segment.id:04d}_raw.wav"
                    final_path = tts_dir / f"segment_{segment.id:04d}.wav"
                    duration = max(segment.end - segment.start, 0.1)

                    self._synthesize_with_model(segment.text, raw_path)
                    self._fit_duration(raw_path, final_path, duration)
                    tracks.append(TTSAudioTrack(segment.id, final_path, segment.start, segment.end))

            return tracks
        finally:
            self.model = None
            self.voice = None
            VRAMManager.cleanup()

    def unload(self) -> None:
        self.model = None
        self.voice = None
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

        voice = getattr(self, "voice", None) or resolve_vieneu_voice(self.model, self.config.voice_model)
        audio = infer_stable_vieneu_audio(self.model, text, voice)
        self.model.save(audio, str(destination))

    def _fit_duration(self, source: Path, destination: Path, target_duration: float) -> None:
        ffmpeg = _ffmpeg()
        current_duration = self._probe_duration(source)
        if current_duration <= 0:
            self._write_silent_wav(destination, target_duration)
            return

        ratio = max(0.1, current_duration / target_duration)
        stream = ffmpeg.input(str(source)).audio
        for value in self._atempo_filters(ratio):
            stream = stream.filter("atempo", value)

        stream = stream.filter("apad").filter("atrim", duration=max(target_duration, 0.1))
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

