from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from app.models.schemas import TranscriptSegment
from app.services.audio_timing import repair_continuous_speech_gaps


def _write_tone(path: Path, voiced_ranges: list[tuple[float, float]], duration: float = 3.0) -> None:
    rate = 16_000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames: list[bytes] = []
        for index in range(round(duration * rate)):
            timestamp = index / rate
            voiced = any(start <= timestamp < end for start, end in voiced_ranges)
            sample = round(5000 * math.sin(2 * math.pi * 220 * timestamp)) if voiced else 0
            frames.append(int(sample).to_bytes(2, "little", signed=True))
        handle.writeframes(b"".join(frames))


class AudioTimingTests(unittest.TestCase):
    def test_repairs_gap_when_original_audio_is_continuous(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "continuous.wav"
            _write_tone(audio, [(0.0, 3.0)])
            segments = [
                TranscriptSegment(id=0, start=0.0, end=1.0, text="một"),
                TranscriptSegment(id=1, start=2.0, end=3.0, text="hai"),
            ]
            repaired = repair_continuous_speech_gaps(audio, segments)
            self.assertAlmostEqual(repaired[0].end, 1.5, places=2)
            self.assertAlmostEqual(repaired[1].start, 1.5, places=2)

    def test_preserves_real_silence_between_cues(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "silence.wav"
            _write_tone(audio, [(0.0, 1.0), (2.0, 3.0)])
            segments = [
                TranscriptSegment(id=0, start=0.0, end=1.0, text="một"),
                TranscriptSegment(id=1, start=2.0, end=3.0, text="hai"),
            ]
            repaired = repair_continuous_speech_gaps(audio, segments)
            self.assertAlmostEqual(repaired[0].end, 1.0, places=2)
            self.assertAlmostEqual(repaired[1].start, 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
