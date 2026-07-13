from __future__ import annotations

import logging
from pathlib import Path

from app.models.schemas import PipelineConfig, TranscriptSegment, WordTimestamp
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


class ASRService:
    """WhisperX transcription with strict model unload after inference."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.model = None
        self.align_model = None
        self.align_metadata = None

    def transcribe(self, video_path: Path, work_dir: Path) -> list[TranscriptSegment]:
        audio_path = work_dir / "source_audio.wav"
        self._extract_audio(video_path, audio_path)

        try:
            return self._run_whisperx(audio_path)
        finally:
            self.unload()

    def unload(self) -> None:
        if self.align_model is not None:
            model = self.align_model
            self.align_model = None
            VRAMManager.release_model(model)
        if self.model is not None:
            model = self.model
            self.model = None
            VRAMManager.release_model(model)
        VRAMManager.cleanup()

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        ffmpeg = _ffmpeg()
        (
            ffmpeg.input(str(video_path))
            .output(str(audio_path), ac=1, ar="16000", vn=None, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )

    def _run_whisperx(self, audio_path: Path) -> list[TranscriptSegment]:
        try:
            import torch
            import whisperx
        except ImportError as exc:
            logger.warning("WhisperX unavailable, returning mock transcript: %s", exc)
            return self._mock_transcript()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 4 if device == "cuda" else 1

        self.model = whisperx.load_model(
            self.config.asr_model,
            device=device,
            compute_type=self.config.compute_type,
            language=self.config.source_language,
        )
        audio = whisperx.load_audio(str(audio_path))
        result = self.model.transcribe(audio, batch_size=batch_size)

        model = self.model
        self.model = None
        VRAMManager.release_model(model)

        language_code = result.get("language") or self.config.source_language or "en"
        try:
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=device,
            )
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
        finally:
            align_model = self.align_model
            self.align_model = None
            self.align_metadata = None
            VRAMManager.release_model(align_model)

        return self._normalize_segments(result.get("segments", []))

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
            segments.append(
                TranscriptSegment(
                    id=index,
                    start=float(raw.get("start", 0.0) or 0.0),
                    end=float(raw.get("end", 0.0) or 0.0),
                    text=str(raw.get("text", "")).strip(),
                    words=words,
                )
            )
        return segments

    def _mock_transcript(self) -> list[TranscriptSegment]:
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
