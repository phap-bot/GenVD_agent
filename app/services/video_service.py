from __future__ import annotations

import logging
from pathlib import Path

from app.models.schemas import PipelineConfig, TranscriptSegment
from app.services.dependency_service import DependencyService
from app.services.timeline_service import TimelineService
from app.services.tts_service import TTSAudioTrack
from app.utils.video_encoder import (
    HARDWARE_ENCODERS,
    cpu_encoder_plan,
    is_hardware_encoder_runtime_error,
    select_video_encoder,
)

logger = logging.getLogger(__name__)
ACCOMPANIMENT_DEFAULT_VOLUME = 0.92
ORIGINAL_FALLBACK_VOLUME = 0.12


def _ffmpeg():
    DependencyService().require_ffmpeg()
    try:
        import ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg-python is required. Install it with `pip install ffmpeg-python`.") from exc
    return ffmpeg


class VideoService:
    """FFmpeg composition with graph-aware video encoder selection."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def render(
        self,
        video_path: Path,
        segments: list[TranscriptSegment],
        tts_tracks: list[TTSAudioTrack],
        work_dir: Path,
        output_path: Path,
        accompaniment_path: Path | None = None,
        original_vocal_path: Path | None = None,
    ) -> tuple[Path, Path]:
        segments = TimelineService().from_transcript(segments)
        subtitle_path = work_dir / "subtitles.srt"
        self.generate_srt(segments, subtitle_path)

        tts_mix_path = work_dir / "tts_mix.wav"
        if tts_tracks:
            self._build_aligned_tts_track(tts_tracks, tts_mix_path)

        self._render_video(
            video_path,
            subtitle_path,
            tts_mix_path if tts_tracks else None,
            output_path,
            accompaniment_path=accompaniment_path,
            original_vocal_path=original_vocal_path,
        )
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
        accompaniment_path: Path | None = None,
        original_vocal_path: Path | None = None,
    ) -> None:
        ffmpeg = _ffmpeg()
        video_input = ffmpeg.input(str(video_path))
        video_stream = video_input.video
        if self.config.burn_subtitles:
            subtitle_filter_path = self._ffmpeg_filter_path(subtitle_path)
            logger.info("video_service.render.subtitle_filter path=%s", subtitle_filter_path)
            video_stream = video_stream.filter("subtitles", subtitle_filter_path)

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

        if tts_mix_path is None:
            if has_acc:
                if self.config.vocal_separation:
                    bg_vol = self.config.background_volume if self.config.background_volume > 0 else ACCOMPANIMENT_DEFAULT_VOLUME
                    audio_stream = ffmpeg.input(str(accompaniment_path)).audio.filter(
                        "volume", round(bg_vol * self.config.accompaniment_gain, 6)
                    )
                    if has_original_vocal and self.config.original_vocal_gain > 0:
                        vocal_stream = ffmpeg.input(str(original_vocal_path)).audio.filter(
                            "volume", self.config.original_vocal_gain
                        )
                        audio_stream = ffmpeg.filter(
                            [audio_stream, vocal_stream],
                            "amix",
                            inputs=2,
                            duration="first",
                            dropout_transition=0,
                            normalize=0,
                        )
                else:
                    # Preserve the historical no-TTS behavior for callers that
                    # provide an accompaniment path without enabling the
                    # configurable separation mix.
                    audio_stream = ffmpeg.input(str(accompaniment_path)).audio
            else:
                audio_stream = video_input.audio
            self._render_output(ffmpeg, video_stream, audio_stream, output_path)
            return

        tts_input = ffmpeg.input(str(tts_mix_path))
        tts_audio = tts_input.audio.filter("volume", self.config.tts_volume)
        tts_split = tts_audio.filter_multi_output("asplit")
        tts_sidechain = tts_split[0]
        tts_for_mix = tts_split[1]
        if has_acc:
            bg_vol = self.config.background_volume if self.config.background_volume > 0 else ACCOMPANIMENT_DEFAULT_VOLUME
            bg_vol = round(bg_vol * self.config.accompaniment_gain, 6)
            logger.info(
                "video_service.render.accompaniment_ducked path=%s volume=%.2f vocal_gain=%.2f",
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
            logger.info("video_service.render.original_audio_emergency_duck video=%s volume=%.2f", video_path, bg_vol)
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

        self._render_output(ffmpeg, video_stream, mixed_audio, output_path)

    def _render_output(self, ffmpeg, video_stream, audio_stream, output_path: Path) -> None:
        plan = select_video_encoder(
            ffmpeg_binary=DependencyService().resolve_binary("ffmpeg") or "ffmpeg",
            has_visual_filters=self.config.burn_subtitles,
            x264_preset="superfast",
            x264_crf=22,
        )
        logger.info(
            "video_service.render.encoder codec=%s reason=%s subtitles=%s",
            plan.codec,
            plan.reason,
            self.config.burn_subtitles,
        )

        def build_output(selected_plan):
            return ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_path),
                acodec="aac",
                movflags="+faststart",
                **selected_plan.options,
            ).overwrite_output()

        try:
            self._run_ffmpeg_command(build_output(plan), "render")
        except RuntimeError as exc:
            if plan.codec in HARDWARE_ENCODERS and is_hardware_encoder_runtime_error(exc, plan.codec):
                fallback = cpu_encoder_plan(x264_preset="superfast", x264_crf=22)
                logger.warning(
                    "video_service.render.encoder_fallback from=%s to=libx264 error=%s",
                    plan.codec,
                    exc,
                )
                self._run_ffmpeg_command(build_output(fallback), "render")
            elif plan.codec == "copy":
                fallback = select_video_encoder(
                    ffmpeg_binary=DependencyService().resolve_binary("ffmpeg") or "ffmpeg",
                    has_visual_filters=True,
                    x264_preset="superfast",
                    x264_crf=22,
                )
                logger.warning(
                    "video_service.render.stream_copy_fallback codec=%s error=%s",
                    fallback.codec,
                    exc,
                )
                self._run_ffmpeg_command(build_output(fallback), "render")
            else:
                raise

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
