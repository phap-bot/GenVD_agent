from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path
from typing import Iterable, TypeVar

from app.models.schemas import TranscriptSegment

T = TypeVar("T", bound=TranscriptSegment)


def _rms_windows(path: Path, start: float, end: float) -> list[float]:
    """Read short PCM windows and return RMS values without loading the file."""
    if end <= start:
        return []
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            if rate <= 0 or width != 2 or channels <= 0:
                return []
            first = max(0, min(handle.getnframes(), round(start * rate)))
            last = max(first, min(handle.getnframes(), math.ceil(end * rate)))
            handle.setpos(first)
            raw = handle.readframes(last - first)
    except (OSError, EOFError, wave.Error):
        return []

    values = array("h")
    values.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if not values:
        return []
    if channels > 1:
        values = array("h", (
            sum(values[index : index + channels]) // channels
            for index in range(0, len(values) - channels + 1, channels)
        ))
    frame_window = max(1, round(rate * 0.02))
    result: list[float] = []
    for offset in range(0, len(values), frame_window):
        window = values[offset : offset + frame_window]
        if window:
            result.append(math.sqrt(sum(sample * sample for sample in window) / len(window)))
    return result


def _gap_contains_audio(path: Path, left_end: float, right_start: float) -> bool:
    if right_start - left_end <= 0.08:
        return False
    gap = _rms_windows(path, left_end, right_start)
    if not gap:
        return False
    left_context = _rms_windows(path, max(0.0, left_end - 0.3), left_end)
    right_context = _rms_windows(path, right_start, right_start + 0.3)
    context = [value for value in (*left_context, *right_context) if value > 0]
    reference = min(context) if context else max(gap)
    # The gap is speech-bearing when it has sustained energy comparable to
    # the known voiced edges. A single click/noise spike is not enough.
    threshold = max(160.0, reference * 0.22)
    voiced = sum(value >= threshold for value in gap)
    return voiced >= max(2, math.ceil(len(gap) * 0.08))


def repair_continuous_speech_gaps(
    audio_path: Path,
    segments: Iterable[T],
    *,
    max_gap_s: float = 8.0,
) -> list[T]:
    """Close ASR timestamp gaps when the original PCM contains audio.

    ASR engines often leave multi-second holes while decoding continuous
    speech. We preserve real silence, but when the source audio carries
    sustained energy between two cues, move their boundary to the midpoint so
    subtitles/TTS remain narratively continuous. Text is never invented or
    discarded by this repair.
    """
    repaired = sorted(list(segments), key=lambda item: (item.start, item.end))
    if len(repaired) < 2 or not audio_path.is_file():
        return repaired
    for index in range(len(repaired) - 1):
        left = repaired[index]
        right = repaired[index + 1]
        gap = right.start - left.end
        if gap <= 0.08 or gap > max_gap_s or not _gap_contains_audio(audio_path, left.end, right.start):
            continue
        boundary = round((left.end + right.start) / 2.0, 3)
        repaired[index] = left.model_copy(update={"end": max(left.start, boundary)})
        repaired[index + 1] = right.model_copy(update={"start": min(right.end, boundary)})
    return [segment.model_copy(update={"id": index}) for index, segment in enumerate(repaired)]

