from __future__ import annotations

import re
from pathlib import Path

from app.models.schemas import TranscriptSegment


class SubtitleService:
    """Parse subtitle files into timestamped segments."""

    _time_range_pattern = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )

    def parse_srt(self, subtitle_path: Path) -> list[TranscriptSegment]:
        content = subtitle_path.read_text(encoding="utf-8-sig")
        blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
        segments: list[TranscriptSegment] = []

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            time_line_index = next(
                (index for index, line in enumerate(lines) if "-->" in line),
                None,
            )
            if time_line_index is None:
                continue

            match = self._time_range_pattern.search(lines[time_line_index])
            if not match:
                continue

            text = " ".join(lines[time_line_index + 1 :]).strip()
            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    id=len(segments),
                    start=self._parse_timestamp(match.group("start")),
                    end=self._parse_timestamp(match.group("end")),
                    text=text,
                    words=[],
                )
            )

        return segments

    def _parse_timestamp(self, value: str) -> float:
        hours, minutes, rest = value.replace(",", ".").split(":")
        seconds, millis = rest.split(".")
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(millis) / 1000
        )

