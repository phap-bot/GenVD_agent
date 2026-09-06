from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import patch

from app.models.schemas import FlashTextBox, FlashTextTrack, PipelineConfig, RenderScriptRequest
from app.services.pipeline import AutoDubbingPipeline
from app.services.render_job_service import RenderJobStore
from utils.flash_text import detect_flash_text_tracks, detect_large_text_boxes


class FlashTextDetectionTests(unittest.TestCase):
    def test_large_upper_text_is_detected_but_subtitle_band_is_ignored(self) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "FLASH", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
        cv2.putText(frame, "subtitle", (220, 335), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        boxes = detect_large_text_boxes(frame)

        self.assertTrue(boxes)
        self.assertTrue(all(box.y + box.height <= 72.0 for box in boxes))
        self.assertTrue(any(box.width > 15.0 and box.height > 5.0 for box in boxes))

    def test_balanced_scan_catches_two_frame_flash_and_never_extends_to_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "flash.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (640, 360),
            )
            try:
                for index in range(30):
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "subtitle", (220, 335), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    if index in {5, 6}:
                        cv2.putText(frame, "FLASH", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
                    writer.write(frame)
            finally:
                writer.release()

            tracks = detect_flash_text_tracks(path, mode="balanced")

        self.assertEqual(len(tracks), 1)
        self.assertLess(tracks[0].start, 0.3)
        self.assertLessEqual(tracks[0].end - tracks[0].start, 3.0)
        self.assertTrue(all(box.y + box.height <= 72.0 for box in tracks[0].boxes))

    def test_long_lived_text_is_not_classified_as_flash_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "persistent.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (320, 180),
            )
            try:
                for _ in range(60):
                    frame = np.zeros((180, 320, 3), dtype=np.uint8)
                    cv2.putText(frame, "TITLE", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    writer.write(frame)
            finally:
                writer.release()

            tracks = detect_flash_text_tracks(path, mode="strict", max_duration_s=3.0)

        self.assertEqual(tracks, [])

    def test_separate_flash_windows_do_not_merge_across_a_real_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "separate-flashes.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (640, 360),
            )
            try:
                for index in range(45):
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    if index in {5, 6, 20, 21}:
                        cv2.putText(frame, "FLASH", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
                    writer.write(frame)
            finally:
                writer.release()

            tracks = detect_flash_text_tracks(path, mode="balanced")

        self.assertEqual(len(tracks), 2)
        self.assertLess(tracks[0].end, tracks[1].start)


class FlashTextRenderTests(unittest.TestCase):
    def test_flash_track_bounds_include_padding_and_stay_inside_frame(self) -> None:
        track = FlashTextTrack(
            id=0,
            start=0.2,
            end=0.6,
            boxes=[
                FlashTextBox(timestamp=0.2, x=65, y=12, width=20, height=20, confidence=0.9),
                FlashTextBox(timestamp=0.5, x=66, y=13, width=19, height=19, confidence=0.9),
            ],
        )
        bounds = AutoDubbingPipeline._flash_track_bounds(track, 1920, 1080)

        left, top, width, height = bounds
        self.assertLess(left, round(1920 * 65 / 100))
        self.assertLess(top, round(1080 * 12 / 100))
        self.assertLessEqual(left + width, 1920)
        self.assertLessEqual(top + height, 1080)

    def test_feature_is_opt_in_and_does_not_change_existing_defaults(self) -> None:
        config = PipelineConfig(copyright_confirmed=True, copyright_source="owned")

        self.assertFalse(config.flash_text_enabled)
        self.assertEqual(config.flash_text_mode, "balanced")

    def test_flash_tracks_survive_queued_render_request_round_trip(self) -> None:
        track = FlashTextTrack(
            id=0,
            start=0.2,
            end=0.6,
            confidence=0.9,
            boxes=[FlashTextBox(timestamp=0.2, x=60, y=10, width=25, height=20, confidence=0.9)],
        )
        payload = RenderScriptRequest(
            source_video_path="/media/source.mp4",
            flash_text_enabled=True,
            flash_text_tracks=[track],
            copyright_confirmed=True,
            copyright_source="owned",
            segments=[
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "original_text": "source",
                    "translated_text": "translated",
                }
            ],
        )
        config = PipelineConfig(
            flash_text_enabled=True,
            copyright_confirmed=True,
            copyright_source="owned",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source-video")
            store = RenderJobStore(root / "jobs", root / "output")
            job_id = store.prepare(payload, config, source, None)

            _loaded_config, _loaded_segments, loaded_tracks = store.load_request(job_id)

        self.assertTrue(_loaded_config.flash_text_enabled)
        self.assertEqual(len(loaded_tracks), 1)
        self.assertEqual(loaded_tracks[0].boxes[0].x, 60)

    def test_render_graph_uses_exact_window_and_respects_opt_in(self) -> None:
        import ffmpeg

        track = FlashTextTrack(
            id=0,
            start=0.2,
            end=0.6,
            boxes=[FlashTextBox(timestamp=0.2, x=60, y=10, width=25, height=20, confidence=0.9)],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            output = root / "output.mp4"
            video.write_bytes(b"video")
            tts.write_bytes(b"tts")

            enabled_pipeline = AutoDubbingPipeline(
                PipelineConfig(
                    flash_text_enabled=True,
                    burn_subtitles=False,
                    mock_tts=True,
                    copyright_confirmed=True,
                    copyright_source="owned",
                )
            )
            captured: dict[str, list[str]] = {}
            with patch.object(
                enabled_pipeline,
                "_run_ffmpeg_command",
                side_effect=lambda command, _stage, **_kwargs: captured.update(args=command.compile()),
            ):
                enabled_pipeline._render_video(
                    video,
                    root / "unused.srt",
                    tts,
                    output,
                    video_width=1920,
                    video_height=1080,
                    flash_text_tracks=[track],
                )

            graph = " ".join(captured["args"])
            self.assertIn("boxblur", graph)
            self.assertIn(r"between(t\,0.200000\,0.600000)", graph)

            disabled_pipeline = AutoDubbingPipeline(
                PipelineConfig(
                    flash_text_enabled=False,
                    burn_subtitles=False,
                    mock_tts=True,
                    copyright_confirmed=True,
                    copyright_source="owned",
                )
            )
            disabled_captured: dict[str, list[str]] = {}
            with patch.object(
                disabled_pipeline,
                "_run_ffmpeg_command",
                side_effect=lambda command, _stage, **_kwargs: disabled_captured.update(args=command.compile()),
            ):
                disabled_pipeline._render_video(
                    video,
                    root / "unused.srt",
                    tts,
                    output,
                    video_width=1920,
                    video_height=1080,
                    flash_text_tracks=[track],
                )

            disabled_graph = " ".join(disabled_captured["args"])
            self.assertNotIn("boxblur", disabled_graph)
            self.assertIn("-vcodec copy", disabled_graph)


if __name__ == "__main__":
    unittest.main()
