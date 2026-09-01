from __future__ import annotations

import threading
import time
import unittest

from app.models.schemas import PipelineConfig, TranscriptSegment
from app.services.pipeline import AutoDubbingPipeline
from app.services.streaming_pipeline import StreamingPipeline
from utils.translation import _repair_translation_boundaries


class StreamingQueueTests(unittest.TestCase):
    def test_pipeline_preserves_block_order_when_consumers_finish_out_of_order(self) -> None:
        translated: list[int] = []

        def asr_blocks():
            yield [0]
            yield [1]
            yield [2]

        def translate(block: list[int]) -> list[int]:
            return [value + 10 for value in block]

        def tts(block: list[int]) -> list[str]:
            if block == [11]:
                time.sleep(0.03)
            translated.extend(block)
            return [f"audio-{block[0]}"]

        result = StreamingPipeline(
            asr_blocks=asr_blocks,
            translate_block=translate,
            tts_block=tts,
            cancel_event=threading.Event(),
        ).run()

        self.assertEqual(result.translated, [10, 11, 12])
        self.assertEqual(result.audio_chunks, ["audio-10", "audio-11", "audio-12"])

    def test_translation_groups_adjacent_asr_blocks_before_calling_api(self) -> None:
        windows: list[list[int]] = []

        def translate(block: list[int]) -> list[int]:
            windows.append(block[:])
            return [value + 10 for value in block]

        result = StreamingPipeline(
            asr_blocks=lambda: ([value] for value in [0, 1, 2, 3]),
            translate_block=translate,
            tts_block=lambda block: block,
            translation_group_size=3,
        ).run()

        self.assertEqual(windows, [[0, 1, 2], [3]])
        self.assertEqual(result.translated, [10, 11, 12, 13])
        self.assertEqual(result.audio_chunks, [10, 11, 12, 13])

    def test_pipeline_propagates_stage_errors(self) -> None:
        def asr_blocks():
            yield ["source"]

        def fail(_block):
            raise ValueError("translation failed")

        with self.assertRaisesRegex(RuntimeError, "translation failed"):
            StreamingPipeline(
                asr_blocks=asr_blocks,
                translate_block=fail,
                tts_block=lambda block: [],
            ).run()


class StreamingTimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = AutoDubbingPipeline(
            PipelineConfig(copyright_confirmed=True, copyright_source="owned")
        )

    def test_offset_preserves_word_timestamps(self) -> None:
        offset = self.pipeline._offset_whisper_segments(
            [{
                "start": 0.2,
                "end": 1.1,
                "text": "hello",
                "words": [{"word": "hello", "start": 0.2, "end": 1.1}],
            }],
            119.5,
        )

        self.assertEqual((offset[0]["start"], offset[0]["end"]), (119.7, 120.6))
        self.assertEqual(
            (offset[0]["words"][0]["start"], offset[0]["words"][0]["end"]),
            (119.7, 120.6),
        )

    def test_boundary_duplicate_is_collapsed_without_cutting_interval(self) -> None:
        result = self.pipeline._deduplicate_stream_segments([
            TranscriptSegment(id=0, start=119.7, end=120.6, text="same sentence"),
            TranscriptSegment(id=1, start=119.8, end=120.5, text="same sentence"),
        ])

        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].start, result[0].end), (119.7, 120.6))

    def test_cjk_translation_boundary_overlap_is_removed(self) -> None:
        result = _repair_translation_boundaries(["你好世界", "世界继续"], target="zh")

        self.assertEqual(result, ["你好世界", "继续"])


if __name__ == "__main__":
    unittest.main()
