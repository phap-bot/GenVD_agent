from __future__ import annotations

from pathlib import Path

from app.models.schemas import PipelineConfig, TranscriptSegment
from app.services.dependency_service import DependencyService
from app.services.tts_service import TTSAudioTrack


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
        (
            ffmpeg.output(mixed, str(destination), ac=2, ar="44100", format="wav")
            .overwrite_output()
            .run(quiet=True)
        )

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
            video_stream = video_stream.filter("subtitles", self._ffmpeg_filter_path(subtitle_path))

        if tts_mix_path is None:
            (
                ffmpeg.output(video_stream, video_input.audio, str(output_path), vcodec="libx264", acodec="aac")
                .overwrite_output()
                .run(quiet=True)
            )
            return

        tts_input = ffmpeg.input(str(tts_mix_path))
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

    def _srt_time(self, seconds: float) -> str:
        millis = round(seconds * 1000)
        hours, remainder = divmod(millis, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def _ffmpeg_filter_path(self, path: Path) -> str:
        return str(path).replace("\\", "/").replace(":", "\\:")
