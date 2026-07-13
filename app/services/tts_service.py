from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import PipelineConfig, TranscriptSegment
from app.services.dependency_service import DependencyService
from app.utils.memory import VRAMManager

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
    """Generate segment-level TTS while never retaining the model afterward."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.model = None

    def synthesize(
        self,
        segments: list[TranscriptSegment],
        work_dir: Path,
    ) -> list[TTSAudioTrack]:
        tts_dir = work_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)

        try:
            if not self.config.mock_tts:
                self._load_model()

            tracks: list[TTSAudioTrack] = []
            for segment in segments:
                raw_path = tts_dir / f"segment_{segment.id:04d}_raw.wav"
                final_path = tts_dir / f"segment_{segment.id:04d}.wav"
                duration = max(segment.end - segment.start, 0.1)

                if self.config.mock_tts:
                    self._write_silent_wav(raw_path, duration)
                else:
                    self._synthesize_with_model(segment.text, raw_path)

                self._fit_duration(raw_path, final_path, duration)
                tracks.append(
                    TTSAudioTrack(
                        segment_id=segment.id,
                        path=final_path,
                        start=segment.start,
                        end=segment.end,
                    )
                )
            return tracks
        finally:
            self.unload()

    def unload(self) -> None:
        if self.model is not None:
            model = self.model
            self.model = None
            VRAMManager.release_model(model)
        VRAMManager.cleanup()

    def _load_model(self) -> None:
        try:
            from vieneu import Vieneu
        except ImportError as exc:
            raise RuntimeError("VieNeu-TTS is not installed. Install it with `pip install vieneu`.") from exc

        if self.config.tts_device == "cuda":
            DependencyService().require_cuda()
        self.model = Vieneu(
            mode="v3turbo",
            device=self.config.tts_device,
            backend="pytorch" if self.config.tts_device == "cuda" else "onnx",
        )

    def _synthesize_with_model(self, text: str, destination: Path) -> None:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")

        audio = self.model.infer(text=text, voice=self.config.voice_model)
        self.model.save(audio, str(destination))

    def _fit_duration(self, source: Path, destination: Path, target_duration: float) -> None:
        ffmpeg = _ffmpeg()
        current_duration = self._probe_duration(source)
        if current_duration <= 0:
            self._write_silent_wav(destination, target_duration)
            return

        ratio = current_duration / target_duration
        filters = self._atempo_filters(ratio)
        stream = ffmpeg.input(str(source)).audio
        for value in filters:
            stream = stream.filter("atempo", value)

        (
            ffmpeg.output(stream, str(destination), ac=1, ar="24000", format="wav")
            .overwrite_output()
            .run(quiet=True)
        )

    def _probe_duration(self, path: Path) -> float:
        ffmpeg = _ffmpeg()
        probe = ffmpeg.probe(str(path))
        return float(probe["format"].get("duration", 0.0))

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
