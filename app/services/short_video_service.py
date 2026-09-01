from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from uuid import uuid4

from app.models.schemas import ShortVideoProfile
from app.utils.media_probe import probe_duration, probe_video_dimensions


# Short Video is deliberately a separate, fast-turnaround workflow.  Anything
# longer than two minutes belongs to Clone Video so it cannot accidentally pay
# the long-video OCR/translation/TTS cost.
SHORT_VIDEO_MAX_SECONDS = 120.0
SHORT_VIDEO_MEDIA_ROOT = Path("temp") / "short_video"
_MEDIA_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ShortVideoService:
    """Media storage and backend-owned routing for the additive Short Video flow."""

    def __init__(self, media_root: str | Path = SHORT_VIDEO_MEDIA_ROOT) -> None:
        self.media_root = Path(media_root)
        self.media_root.mkdir(parents=True, exist_ok=True)

    @property
    def max_short_seconds(self) -> float:
        raw = os.environ.get("AUTODUB_SHORT_VIDEO_MAX_SECONDS", str(SHORT_VIDEO_MAX_SECONDS)).strip()
        try:
            return min(SHORT_VIDEO_MAX_SECONDS, max(1.0, float(raw)))
        except ValueError:
            return SHORT_VIDEO_MAX_SECONDS

    def store_upload(self, source_name: str | None, source_path: Path) -> tuple[str, Path]:
        media_id = uuid4().hex
        suffix = Path(source_name or "input.mp4").suffix.lower() or ".mp4"
        destination_dir = self.media_root / media_id
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination = destination_dir / f"input{suffix}"
        shutil.copy2(source_path, destination)
        return media_id, destination

    def resolve(self, media_id: str) -> Path:
        clean_id = (media_id or "").strip().lower()
        if not _MEDIA_ID_PATTERN.fullmatch(clean_id):
            raise ValueError("Invalid Short Video media_id")
        directory = (self.media_root / clean_id).resolve()
        root = self.media_root.resolve()
        if directory.parent != root:
            raise ValueError("Invalid Short Video media path")
        candidates = sorted(directory.glob("input.*"))
        if not candidates or not candidates[0].is_file() or candidates[0].stat().st_size <= 0:
            raise FileNotFoundError("Short Video source media was not found")
        return candidates[0]

    def inspect(self, media_id: str, path: Path) -> dict[str, object]:
        duration = round(max(0.0, probe_duration(path)), 3)
        width: int | None = None
        height: int | None = None
        try:
            width, height = probe_video_dimensions(path)
        except Exception:
            # Duration is enough to route; dimensions are informative metadata.
            pass

        has_audio = self._has_audio(path)
        profile = self.resolve_profile(duration, has_audio=has_audio)
        return {
            "media_id": media_id,
            "filename": path.name,
            "input_url": f"/temp/short_video/{media_id}/{path.name}",
            "duration_seconds": duration,
            "has_audio": has_audio,
            "width": width,
            "height": height,
            "profile": profile,
        }

    def resolve_profile(self, duration: float, *, has_audio: bool = True) -> ShortVideoProfile:
        duration = max(0.0, float(duration))
        max_short = self.max_short_seconds
        if duration <= 60:
            name = "micro"
        elif duration <= 120:
            name = "short"
        elif duration <= 300:
            # Kept as a compatibility label for clients that already know the
            # profile name.  Its route is long whenever the configured Short
            # limit is 120 seconds (the default), so it is never processed by
            # the Short pipeline.
            name = "short_extended"
        else:
            name = "long"

        is_short = duration <= max_short
        if name == "micro":
            asr_model = "base"
        elif is_short:
            asr_model = "small"
        else:
            asr_model = "small"

        return ShortVideoProfile(
            name=name,
            route="short_video" if is_short else "clone_video",
            duration_seconds=duration,
            max_short_seconds=max_short,
            asr_model=asr_model,
            source_mode="subtitle" if not has_audio else "auto",
            vocal_separation_default=False,
        )

    def require_short(self, path: Path) -> ShortVideoProfile:
        duration = probe_duration(path)
        profile = self.resolve_profile(duration, has_audio=self._has_audio(path))
        if profile.route != "short_video":
            raise ValueError(
                f"Video duration {duration:.1f}s exceeds Short Video limit {self.max_short_seconds:.1f}s; "
                "use Clone Video for the long-video pipeline."
            )
        return profile

    def evict_stale(self, max_age_seconds: float = 24 * 3600) -> None:
        now = time.time()
        for directory in self.media_root.iterdir():
            try:
                if directory.is_dir() and now - directory.stat().st_mtime > max_age_seconds:
                    shutil.rmtree(directory, ignore_errors=True)
            except OSError:
                continue

    def _has_audio(self, path: Path) -> bool:
        try:
            import ffmpeg

            probe = ffmpeg.probe(str(path))
            return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))
        except Exception:
            return False
