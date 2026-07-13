from __future__ import annotations

import json
import logging
import math
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

from app.models.schemas import DubbingScriptSegment, PipelineConfig, TranscriptSegment, WordTimestamp
from app.services.dependency_service import DependencyService
from app.utils.vram import VRAMManager
from app.utils.workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioChunk:
    segment_id: int
    path: Path
    start: float
    end: float


class AutoDubbingPipeline:
    """SSE-producing orchestrator for the auto-dubbing pipeline.

    All model stages are guarded by a process-wide lock so two requests cannot
    load GPU models at the same time on a 4GB VRAM machine.
    """

    _gpu_lock = threading.RLock()

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dependencies = DependencyService()

    def run(self, workspace: Workspace) -> Generator[str, None, Path]:
        output_path = workspace.output_dir / f"{workspace.request_id}_dubbed.mp4"

        try:
            yield self._event("processing", "Initializing workspace...", request_id=workspace.request_id)
            self.dependencies.require_ffmpeg()

            with self._gpu_lock:
                yield self._event("processing", "Extracting audio and transcribing...")
                segments = self._run_asr(workspace)
                VRAMManager.cleanup()

                yield self._event("processing", "Translating script...")
                translated_segments = self._translate_segments(segments)
                VRAMManager.cleanup()

                yield self._event("processing", "Generating AI voice...")
                chunks = self._run_tts(translated_segments, workspace)
                VRAMManager.cleanup()

            yield self._event("processing", "Rendering final video...")
            subtitle_path = workspace.root / "translated.srt"
            self._write_srt(translated_segments, subtitle_path)
            tts_mix_path = workspace.root / "tts_mix.wav"
            self._combine_audio_chunks(chunks, tts_mix_path)
            self._render_video(
                video_path=workspace.input_video,
                subtitle_path=subtitle_path,
                tts_mix_path=tts_mix_path,
                output_path=output_path,
            )

            yield self._event("success", "Completed", video_url=f"/media/{output_path.name}")
            return output_path
        except Exception as exc:
            VRAMManager.cleanup()
            if VRAMManager.is_cuda_oom(exc):
                logger.exception("CUDA OOM during dubbing pipeline")
                yield self._event(
                    "error",
                    "CUDA OOM prevented. Model was unloaded and VRAM cache was cleared.",
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
        with self._gpu_lock:
            try:
                self.dependencies.require_ffmpeg()
                segments = self._run_asr(workspace)
                VRAMManager.cleanup()
                translated = self._translate_segments(segments)
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
        try:
            yield self._event("processing", "Initializing analyze workspace...", request_id=workspace.request_id)
            self.dependencies.require_ffmpeg()
            with self._gpu_lock:
                yield self._event("processing", "Extracting audio and transcribing...")
                segments = self._run_asr(workspace)
                VRAMManager.cleanup()

                yield self._event("processing", "Translating script...")
                translated = self._translate_segments(segments)
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

            yield self._event(
                "success",
                "Analyze completed",
                source_video_path=f"/media/{workspace.request_id}_source.mp4",
                segments=[segment.model_dump() for segment in analyzed],
            )
            return analyzed
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
        subtitle_segments = [
            TranscriptSegment(
                id=index,
                start=segment.start,
                end=segment.end,
                text=segment.translated_text,
                words=[],
            )
            for index, segment in enumerate(script_segments)
        ]

        try:
            yield self._event("processing", "Initializing render workspace...", request_id=workspace.request_id)
            self.dependencies.require_ffmpeg()
            if not source_video_path.exists():
                raise FileNotFoundError(f"Source video not found: {source_video_path}")

            with self._gpu_lock:
                yield self._event("processing", "Generating AI voice from edited script...")
                chunks = self._run_tts_from_script(script_segments, workspace)
                VRAMManager.cleanup()

            yield self._event("processing", "Rendering final video...")
            subtitle_path = workspace.root / "edited_script.srt"
            self._write_srt(subtitle_segments, subtitle_path)
            tts_mix_path = workspace.root / "tts_mix.wav"
            self._combine_audio_chunks(chunks, tts_mix_path)
            self._render_video(
                video_path=source_video_path,
                subtitle_path=subtitle_path,
                tts_mix_path=tts_mix_path,
                output_path=output_path,
            )

            yield self._event("success", "Completed", video_url=f"/media/{output_path.name}")
            return output_path
        except Exception as exc:
            VRAMManager.cleanup()
            if VRAMManager.is_cuda_oom(exc):
                logger.exception("CUDA OOM during script render")
                yield self._event(
                    "error",
                    "CUDA OOM prevented. Model was unloaded and VRAM cache was cleared.",
                    error=str(exc),
                )
                return output_path

            logger.exception("Script render failed")
            yield self._event("error", "Script render failed", error=str(exc))
            return output_path
        finally:
            VRAMManager.cleanup()

    def _run_asr(self, workspace: Workspace) -> list[TranscriptSegment]:
        audio_path = workspace.root / "source_audio.wav"
        ffmpeg = self._ffmpeg()
        (
            ffmpeg.input(str(workspace.input_video))
            .output(str(audio_path), ac=1, ar="16000", vn=None, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )

        model = None
        align_model = None
        try:
            try:
                import torch
                import whisperx
            except ImportError as exc:
                logger.warning("WhisperX unavailable, using mock transcript: %s", exc)
                return self._mock_segments()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisperx.load_model(
                self.config.asr_model,
                device=device,
                compute_type=self.config.compute_type,
                language=self.config.source_language,
            )
            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(audio, batch_size=4 if device == "cuda" else 1)

            del model
            model = None
            VRAMManager.cleanup()

            if not self.config.word_timestamps:
                return self._normalize_segments(result.get("segments", []))

            language_code = result.get("language") or self.config.source_language or "en"
            align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            aligned = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            return self._normalize_segments(aligned.get("segments", []))
        finally:
            if align_model is not None:
                del align_model
            if model is not None:
                del model
            VRAMManager.cleanup()

    def _translate_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        translated: list[TranscriptSegment] = []
        for segment in segments:
            translated.append(
                segment.model_copy(
                    update={"text": f"[{self.config.target_language}] {segment.text}"}
                )
            )
        return translated

    def _run_tts(self, segments: list[TranscriptSegment], workspace: Workspace) -> list[AudioChunk]:
        model = None
        chunks: list[AudioChunk] = []

        try:
            if not self.config.mock_tts:
                try:
                    from vieneu import Vieneu
                except ImportError as exc:
                    raise RuntimeError("VieNeu-TTS is not installed. Install it with `pip install vieneu`.") from exc
                if self.config.tts_device == "cuda":
                    self.dependencies.require_cuda()
                model = Vieneu(
                    mode="v3turbo",
                    device=self.config.tts_device,
                    backend="pytorch" if self.config.tts_device == "cuda" else "onnx",
                )

            for segment in segments:
                raw_path = workspace.chunks_dir / f"{segment.id:04d}_raw.wav"
                final_path = workspace.chunks_dir / f"{segment.id:04d}.wav"
                duration = max(segment.end - segment.start, 0.1)

                if model is None:
                    self._write_silent_wav(raw_path, duration)
                else:
                    audio = model.infer(text=segment.text, voice=self.config.voice_model)
                    model.save(audio, str(raw_path))

                self._fit_audio_duration(raw_path, final_path, duration)
                chunks.append(AudioChunk(segment.id, final_path, segment.start, segment.end))

            return chunks
        finally:
            if model is not None:
                del model
            VRAMManager.cleanup()

    def _run_tts_from_script(
        self,
        segments: list[DubbingScriptSegment],
        workspace: Workspace,
    ) -> list[AudioChunk]:
        model = None
        chunks: list[AudioChunk] = []

        try:
            if not self.config.mock_tts:
                try:
                    from vieneu import Vieneu
                except ImportError as exc:
                    raise RuntimeError("VieNeu-TTS is not installed. Install it with `pip install vieneu`.") from exc
                if self.config.tts_device == "cuda":
                    self.dependencies.require_cuda()
                model = Vieneu(
                    mode="v3turbo",
                    device=self.config.tts_device,
                    backend="pytorch" if self.config.tts_device == "cuda" else "onnx",
                )

            for index, segment in enumerate(segments):
                raw_path = workspace.chunks_dir / f"{index:04d}_raw.wav"
                final_path = workspace.chunks_dir / f"{index:04d}.wav"
                duration = max(segment.end - segment.start, 0.1)
                voice = segment.voice_model or self.config.voice_model

                if model is None:
                    self._write_silent_wav(raw_path, duration)
                else:
                    audio = model.infer(text=segment.translated_text, voice=voice)
                    model.save(audio, str(raw_path))

                self._fit_audio_duration(raw_path, final_path, duration)
                chunks.append(AudioChunk(index, final_path, segment.start, segment.end))

            return chunks
        finally:
            if model is not None:
                del model
            VRAMManager.cleanup()

    def _fit_audio_duration(self, source: Path, destination: Path, target_duration: float) -> None:
        if self._fit_audio_duration_with_pydub(source, destination, target_duration):
            return

        ffmpeg = self._ffmpeg()
        current_duration = self._probe_duration(source)
        if current_duration <= 0:
            self._write_silent_wav(destination, target_duration)
            return

        tempo = max(0.1, current_duration / target_duration)
        filters = self._atempo_filters(tempo)
        stream = ffmpeg.input(str(source)).audio
        for value in filters:
            stream = stream.filter("atempo", value)

        (
            ffmpeg.output(stream, str(destination), ac=1, ar="24000", format="wav")
            .overwrite_output()
            .run(quiet=True)
        )

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
            stretched = audio._spawn(
                audio.raw_data,
                overrides={"frame_rate": max(1, round(audio.frame_rate * speed))},
            ).set_frame_rate(audio.frame_rate)

            if len(stretched) > target_ms:
                stretched = stretched[:target_ms]
            elif len(stretched) < target_ms:
                stretched += AudioSegment.silent(duration=target_ms - len(stretched), frame_rate=audio.frame_rate)

            stretched.export(destination, format="wav")
            return True
        except Exception:
            logger.exception("pydub duration fitting failed, falling back to ffmpeg atempo")
            return False

    def _combine_audio_chunks(self, chunks: list[AudioChunk], destination: Path) -> None:
        if not chunks:
            self._write_silent_wav(destination, 0.1, sample_rate=44100)
            return

        ffmpeg = self._ffmpeg()
        delayed_streams = []
        for chunk in chunks:
            delay_ms = max(0, round(chunk.start * 1000))
            delayed_streams.append(
                ffmpeg.input(str(chunk.path))
                .audio
                .filter("adelay", delays=f"{delay_ms}|{delay_ms}")
            )

        mixed = ffmpeg.filter(
            delayed_streams,
            "amix",
            inputs=len(delayed_streams),
            duration="longest",
            normalize=0,
        )
        (
            ffmpeg.output(mixed, str(destination), ac=2, ar="44100", format="wav")
            .overwrite_output()
            .run(quiet=True)
        )

    def _render_video(
        self,
        video_path: Path,
        subtitle_path: Path,
        tts_mix_path: Path,
        output_path: Path,
    ) -> None:
        ffmpeg = self._ffmpeg()
        video_input = ffmpeg.input(str(video_path))
        tts_input = ffmpeg.input(str(tts_mix_path))

        video_stream = video_input.video
        if self.config.burn_subtitles:
            video_stream = video_stream.filter("subtitles", self._ffmpeg_filter_path(subtitle_path))

        original_audio = video_input.audio.filter("volume", self.config.background_volume)
        tts_audio = tts_input.audio.filter("volume", self.config.tts_volume)
        mixed_audio = ffmpeg.filter(
            [original_audio, tts_audio],
            "amix",
            inputs=2,
            duration="first",
            dropout_transition=0,
            normalize=0,
        )

        (
            ffmpeg.output(video_stream, mixed_audio, str(output_path), vcodec="libx264", acodec="aac", shortest=None)
            .overwrite_output()
            .run(quiet=True)
        )

    def _write_srt(self, segments: Iterable[TranscriptSegment], destination: Path) -> None:
        lines: list[str] = []
        for index, segment in enumerate(segments, start=1):
            lines.extend(
                [
                    str(index),
                    f"{self._srt_time(segment.start)} --> {self._srt_time(segment.end)}",
                    segment.text,
                    "",
                ]
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
        ffmpeg = self._ffmpeg()
        probe = ffmpeg.probe(str(path))
        return float(probe["format"].get("duration", 0.0))

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

    def _ffmpeg_filter_path(self, path: Path) -> str:
        return str(path).replace("\\", "/").replace(":", "\\:")

    def _ffmpeg(self):
        self.dependencies.require_ffmpeg()
        try:
            import ffmpeg
        except ImportError as exc:
            raise RuntimeError("ffmpeg-python is required. Install it with `pip install ffmpeg-python`.") from exc
        return ffmpeg

    def _event(self, status: str, step: str, **extra: object) -> str:
        payload = {"status": status, "step": step, **extra}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
