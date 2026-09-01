from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.models.schemas import DubbingScriptSegment, PipelineConfig, TranscriptSegment
from app.services.pipeline import AutoDubbingPipeline
from app.services.short_video_pipeline import ShortVideoPipeline
from app.services.short_video_service import ShortVideoService


class ShortVideoProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ShortVideoService(Path(tempfile.mkdtemp()))

    def test_profiles_route_short_durations_without_touching_long_pipeline(self) -> None:
        self.assertEqual(self.service.resolve_profile(30).name, "micro")
        self.assertEqual(self.service.resolve_profile(120).name, "short")
        self.assertEqual(self.service.resolve_profile(240).name, "short_extended")
        self.assertEqual(self.service.resolve_profile(301).route, "clone_video")

    def test_short_limit_is_configurable(self) -> None:
        with patch.dict("os.environ", {"AUTODUB_SHORT_VIDEO_MAX_SECONDS": "180"}):
            service = ShortVideoService(Path(tempfile.mkdtemp()))
            self.assertEqual(service.resolve_profile(181).route, "clone_video")

    def test_media_id_is_path_safe(self) -> None:
        with self.assertRaises(ValueError):
            self.service.resolve("../../output")

    def test_hybrid_source_keeps_unmatched_hard_subtitle(self) -> None:
        pipeline = AutoDubbingPipeline(
            PipelineConfig(copyright_confirmed=True, copyright_source="owned", source_mode="hybrid")
        )
        asr = [TranscriptSegment(id=0, start=0, end=2, text="spoken line", words=[])]
        ocr = [TranscriptSegment(id=0, start=4, end=5, text="ON SCREEN TITLE", words=[])]
        merged = pipeline._merge_asr_ocr_segments(asr, ocr)
        self.assertEqual([item.text for item in merged], ["spoken line", "ON SCREEN TITLE"])

    def test_dedicated_pipeline_keeps_short_cues_and_tts_groups_separate(self) -> None:
        pipeline = ShortVideoPipeline(
            PipelineConfig(copyright_confirmed=True, copyright_source="owned")
        )
        source = [
            TranscriptSegment(id=0, start=0, end=0.8, text="先说", words=[]),
            TranscriptSegment(id=1, start=0.9, end=1.7, text="再说", words=[]),
        ]
        timeline = pipeline._source_timeline(source)
        groups = pipeline._build_tts_groups(
            timeline,
            [
                DubbingScriptSegment(id=0, start=0, end=0.8, original_text="先说", translated_text="Nói trước"),
                DubbingScriptSegment(id=1, start=0.9, end=1.7, original_text="再说", translated_text="Nói sau"),
            ],
        )
        self.assertTrue(pipeline.config.short_video)
        self.assertEqual(len(timeline), 2)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(group.segment_count == 1 for group in groups))

    def test_soft_timing_uses_only_bounded_silence_between_short_cues(self) -> None:
        pipeline = ShortVideoPipeline(
            PipelineConfig(
                copyright_confirmed=True,
                copyright_source="owned",
                soft_timing_fit=True,
                timing_max_drift_s=0.35,
                timing_min_gap_s=0.08,
            )
        )
        # 0.50s of silence follows the source cue; only 0.35s may be used,
        # while the configured 0.08s gap remains untouched.
        self.assertAlmostEqual(pipeline._timing_fit_duration(0.0, 1.0, 1.58), 1.35)
        self.assertAlmostEqual(pipeline._timing_fit_duration(0.0, 1.0, 1.20), 1.12)
        self.assertAlmostEqual(pipeline._timing_fit_duration(0.0, 1.0, None), 1.0)


if __name__ == "__main__":
    unittest.main()
