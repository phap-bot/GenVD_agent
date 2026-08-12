from __future__ import annotations

import logging
import os
from pathlib import Path

from app.models.schemas import PipelineConfig, TranscriptSegment, WordTimestamp
from app.services.dependency_service import DependencyService
from app.utils.memory import VRAMManager
from utils.model_cache import configure_model_cache
from utils.model_registry import model_registry
from utils.stt import remote_stt_enabled, transcribe_audio_remote
from utils.language import detect_language

logger = logging.getLogger(__name__)
MODEL_CACHE_PATHS = configure_model_cache()


def _env_value(name: str, default: str = "") -> str:
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


def _ffmpeg():
    DependencyService().require_ffmpeg()
    try:
        import ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg-python is required. Install it with `pip install ffmpeg-python`.") from exc
    return ffmpeg


class ASRService:
    """WhisperX transcription through the shared offloaded model registry."""

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
        selected_engine = self.config.asr_engine
        if (
            selected_engine == "auto"
            and (self.config.source_language or "").lower().startswith("zh")
            and _env_truthy("AUTODUB_AUTO_PARAFORMER")
        ):
            selected_engine = "paraformer"
        if selected_engine == "paraformer":
            try:
                from utils.paraformer import transcribe as paraformer_transcribe

                return self._normalize_segments(paraformer_transcribe(
                    audio_path,
                    language=self.config.source_language,
                    model_id=_env_value("AUTODUB_PARAFORMER_MODEL", "paraformer-zh"),
                ))
            except Exception:
                if self.config.asr_engine == "paraformer":
                    raise
                logger.warning("asr_service.paraformer.auto_fallback_to_whisper", exc_info=True)
        if remote_stt_enabled(self.config.asr_model):
            logger.info(
                "asr_service.remote.start model=%s language=%s audio=%s",
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

        DependencyService().require_cuda()
        device = "cuda"
        batch_size = 4

        whisper_arch = self.config.asr_model if self.config.whisper_model == "auto" else self.config.whisper_model
        logger.info(
            "asr_service.model.load model=%s compute_type=%s device=%s language=%s cache=%s",
            self.config.asr_model,
            self.config.compute_type,
            device,
            self.config.source_language or "auto",
            MODEL_CACHE_PATHS.whisperx_asr_cache,
        )
        audio = whisperx.load_audio(str(audio_path))
        with model_registry.acquire_whisperx_asr(
            whisperx,
            whisper_arch=whisper_arch,
            device=device,
            compute_type=self.config.compute_type,
            language=self.config.source_language,
            beam_size=self.config.whisper_beam_size,
        ) as model:
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                language=self.config.source_language,
            )

        VRAMManager.cleanup()

        language_code = result.get("language") or self.config.source_language or "en"
        logger.info(
            "asr_service.align.load language=%s device=%s cache=%s",
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
                ) as (align_model, align_metadata):
                    aligned = whisperx.align(
                        valid_segs,
                        align_model,
                        align_metadata,
                        audio,
                        device,
                        return_char_alignments=False,
                    )
                return self._normalize_segments(aligned.get("segments", []))
            except Exception as exc:
                logger.warning(
                    "asr_service.align.failed language=%s error=%s, falling back to ASR segments",
                    language_code,
                    exc,
                )

        return self._normalize_segments(raw_segs)

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
                    language=(raw.get("language") or detect_language(str(raw.get("text", "")), fallback=self.config.source_language)[0]),
                    language_probability=(float(raw.get("language_probability")) if raw.get("language_probability") is not None else detect_language(str(raw.get("text", "")), fallback=self.config.source_language)[1]),
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
