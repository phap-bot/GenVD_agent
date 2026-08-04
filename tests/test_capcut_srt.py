from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.models.schemas import PipelineConfig, TranscriptSegment
from app.services.pipeline import AutoDubbingPipeline


class CapCutSrtTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = AutoDubbingPipeline(
            PipelineConfig(
                copyright_confirmed=True,
                copyright_source="owned",
            )
        )

    def test_srt_uses_capcut_compatible_encoding_and_exact_timeline(self) -> None:
        segments = [
            TranscriptSegment(id=0, start=0.125, end=1.999, text="Xin chao CapCut"),
            TranscriptSegment(id=1, start=1.5, end=3.0, text="Phu de tieng Viet"),
            TranscriptSegment(id=2, start=5.0, end=6.0, text="Ngoai video"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "captions.srt"
            self.pipeline._write_srt(segments, destination, video_duration=2.25)
            raw = destination.read_bytes()

        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"\r\n", raw)
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))
        self.assertEqual(
            raw.decode("utf-8-sig"),
            "1\r\n"
            "00:00:00,125 --> 00:00:01,500\r\n"
            "Xin chao CapCut\r\n"
            "\r\n"
            "2\r\n"
            "00:00:01,500 --> 00:00:02,250\r\n"
            "Phu de tieng Viet\r\n"
            "\r\n",
        )

    def test_srt_preserves_vietnamese_text(self) -> None:
        segment = TranscriptSegment(
            id=0,
            start=0,
            end=1,
            text="Ph\u1ee5 \u0111\u1ec1 ti\u1ebfng Vi\u1ec7t",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "vietnamese.srt"
            self.pipeline._write_srt([segment], destination)
            content = destination.read_text(encoding="utf-8-sig")

        self.assertIn("Ph\u1ee5 \u0111\u1ec1 ti\u1ebfng Vi\u1ec7t", content)


if __name__ == "__main__":
    unittest.main()
