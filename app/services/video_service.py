from __future__ import annotations

import logging
from pathlib import Path

from app.models.schemas import PipelineConfig, TranscriptSegment
from app.services.dependency_service import DependencyService
from app.services.timeline_service import TimelineService
from app.services.tts_service import TTSAudioTrack

logger = logging.getLogger(__name__)


def _ffmpeg():
    DependencyService().require_ffmpeg()
    try:
        import ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg-python is required. Install it with `pip install ffmpeg-python`.") from exc
    return ffmpeg


class VideoService:
    """CPU-only ffmpeg composition for subtitles and mixed audio."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def render(
        self,
        video_path: Path,
        segments: list[TranscriptSegment],
        tts_tracks: list[TTSAudioTrack],
        work_dir: Path,
        output_path: Path,
    ) -> tuple[Path, Path]:
        segments = TimelineService().from_transcript(segments)
        subtitle_path = work_dir / "subtitles.srt"
        self.generate_srt(segments, subtitle_path)

        tts_mix_path = work_dir / "tts_mix.wav"
        if tts_tracks:
            self._build_aligned_tts_track(tts_tracks, tts_mix_path)

        self._render_video(video_path, subtitle_path, tts_mix_path if tts_tracks else None, output_path)
        return output_path, subtitle_path

    def generate_srt(self, segments: list[TranscriptSegment], destination: Path) -> Path:
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
        return destination

    def _build_aligned_tts_track(self, tracks: list[TTSAudioTrack], destination: Path) -> None:
        ffmpeg = _ffmpeg()
        delayed_streams = []
        for track in tracks:
            delay_ms = max(0, round(track.start * 1000))
            delayed_streams.append(
                ffmpeg.input(str(track.path))
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
        command = ffmpeg.output(mixed, str(destination), ac=2, ar="44100", format="wav").overwrite_output()
        self._run_ffmpeg_command(command, "audio_mix")

    def _render_video(
        self,
        video_path: Path,
        subtitle_path: Path,
        tts_mix_path: Path | None,
        output_path: Path,
    ) -> None:
        ffmpeg = _ffmpeg()
        video_input = ffmpeg.input(str(video_path))
        video_stream = video_input.video
        if self.config.burn_subtitles:
            subtitle_filter_path = self._ffmpeg_filter_path(subtitle_path)
            logger.info("video_service.render.subtitle_filter path=%s", subtitle_filter_path)
            video_stream = video_stream.filter("subtitles", subtitle_filter_path)

        if tts_mix_path is None:
            command = ffmpeg.output(
                video_stream,
                video_input.audio,
                str(output_path),
                vcodec="libx264",
                acodec="aac",
            ).overwrite_output()
            self._run_ffmpeg_command(command, "render")
            return

        tts_input = ffmpeg.input(str(tts_mix_path))
        tts_audio = tts_input.audio.filter("volume", self.config.tts_volume)
        if self.config.background_volume > 0:
            original_audio = video_input.audio.filter("volume", self.config.background_volume)
            mixed_audio = ffmpeg.filter(
                [original_audio, tts_audio],
                "amix",
                inputs=2,
                duration="first",
                dropout_transition=0,
                normalize=0,
            )
        else:
            logger.info("video_service.render.original_audio.muted video=%s", video_path)
            mixed_audio = tts_audio

        command = ffmpeg.output(
            video_stream,
            mixed_audio,
            str(output_path),
            vcodec="libx264",
            acodec="aac",
            shortest=None,
        ).overwrite_output()
        self._run_ffmpeg_command(command, "render")

    def _srt_time(self, seconds: float) -> str:
        millis = round(seconds * 1000)
        hours, remainder = divmod(millis, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

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
            logger.error("video_service.ffmpeg.%s.error stderr=%s", stage, stderr or "<empty>")
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
