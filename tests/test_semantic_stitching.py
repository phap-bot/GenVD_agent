from __future__ import annotations

import unittest

from app.models.schemas import TranscriptSegment
from app.services.semantic_stitching_service import SemanticStitchingService


class TestSemanticStitchingService(unittest.TestCase):

    def setUp(self):
        self.service = SemanticStitchingService()

    def test_repair_cjk_severed_words(self):
        segments = [
            TranscriptSegment(id=0, start=3.0, end=7.0, text="著朋友們一起喝酒都趁着賽了停止停止灰姐今年の世", words=[]),
            TranscriptSegment(id=1, start=7.05, end=10.0, text="界杯好像有點不一樣了不僅已經來了全新的比賽機制而", words=[]),
        ]
        repaired = self.service.repair_cjk_severed_words(segments)
        self.assertEqual(len(repaired), 1)
        self.assertTrue(repaired[0].text.startswith("著朋友們"))
        self.assertIn("今年の世界杯", repaired[0].text)
        self.assertEqual(repaired[0].start, 3.0)
        self.assertEqual(repaired[0].end, 10.0)

    def test_stitch_transcript_segments_cjk(self):
        segments = [
            TranscriptSegment(id=0, start=3.0, end=7.0, text="著朋友們一起喝酒都趁着賽了停止停止灰姐今年の世", words=[]),
            TranscriptSegment(id=1, start=7.1, end=10.0, text="界杯好像有點不一樣了不僅已經來了全新的比賽機制而", words=[]),
            TranscriptSegment(id=2, start=10.1, end=14.0, text="且還迎來了史上最好的觀賽時段超百分之七十的賽事", words=[]),
        ]
        stitched = self.service.stitch_transcript_segments(segments, source_language="zh", max_duration=12.0)
        self.assertEqual(len(stitched), 1)
        self.assertIn("世界杯", stitched[0].text)
        self.assertEqual(stitched[0].start, 3.0)
        self.assertEqual(stitched[0].end, 14.0)

    def test_smooth_translated_clauses_dangling_ma(self):
        segments = [
            TranscriptSegment(id=0, start=3.0, end=7.0, text="bạn bè cùng uống rượu xem bóng đá rồi. Khoan đã, World Cup năm nay", words=[]),
            TranscriptSegment(id=1, start=7.1, end=10.0, text="hình như có chút khác biệt rồi. Không chỉ có thể thức thi đấu hoàn toàn mới, mà", words=[]),
            TranscriptSegment(id=2, start=10.1, end=14.0, text="còn có khung giờ phát sóng thân thiện nhất lịch sử. Hơn 70% số trận đấu", words=[]),
        ]
        smoothed = self.service.smooth_translated_clauses(segments)
        self.assertEqual(len(smoothed), 2)
        self.assertTrue("hoàn toàn mới, mà còn có khung giờ" in smoothed[1].text)
        self.assertEqual(smoothed[1].start, 7.1)
        self.assertEqual(smoothed[1].end, 14.0)

    def test_terminal_punctuation_prevents_merge(self):
        segments = [
            TranscriptSegment(id=0, start=0.0, end=2.5, text="Hello world.", words=[]),
            TranscriptSegment(id=1, start=2.6, end=5.0, text="This is the next sentence.", words=[]),
        ]
        stitched = self.service.stitch_transcript_segments(segments, source_language="en")
        self.assertEqual(len(stitched), 2)
        self.assertEqual(stitched[0].text, "Hello world.")
        self.assertEqual(stitched[1].text, "This is the next sentence.")


if __name__ == "__main__":
    unittest.main()
