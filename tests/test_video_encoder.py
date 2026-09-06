from __future__ import annotations

import unittest
from unittest.mock import patch

from app.utils.video_encoder import is_hardware_encoder_runtime_error, select_video_encoder


class VideoEncoderSelectorTests(unittest.TestCase):
    def test_stream_copy_is_used_when_the_video_has_no_visual_filters(self) -> None:
        with (
            patch.dict("os.environ", {"AUTODUB_VIDEO_ENCODER": "auto"}, clear=False),
            patch("app.utils.video_encoder.hardware_encoder_usable") as probe,
        ):
            plan = select_video_encoder(
                ffmpeg_binary="ffmpeg",
                has_visual_filters=False,
                x264_preset="superfast",
                x264_crf=22,
            )

        self.assertEqual(plan.codec, "copy")
        self.assertEqual(plan.options, {"vcodec": "copy"})
        probe.assert_not_called()

    def test_auto_mode_uses_nvenc_when_a_real_probe_succeeds(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "AUTODUB_VIDEO_ENCODER": "auto",
                    "AUTODUB_NVENC_PRESET": "p4",
                    "AUTODUB_NVENC_CQ": "23",
                },
                clear=False,
            ),
            patch(
                "app.utils.video_encoder.hardware_encoder_usable",
                side_effect=lambda _binary, codec: codec == "h264_nvenc",
            ),
        ):
            plan = select_video_encoder(
                ffmpeg_binary="ffmpeg",
                has_visual_filters=True,
                x264_preset="superfast",
                x264_crf=22,
            )

        self.assertEqual(plan.codec, "h264_nvenc")
        self.assertEqual(plan.options["preset"], "p4")
        self.assertEqual(plan.options["cq"], 23)

    def test_auto_mode_falls_back_to_x264_when_nvenc_probe_fails(self) -> None:
        with (
            patch.dict("os.environ", {"AUTODUB_VIDEO_ENCODER": "auto"}, clear=False),
            patch("app.utils.video_encoder.hardware_encoder_usable", return_value=False),
        ):
            plan = select_video_encoder(
                ffmpeg_binary="ffmpeg",
                has_visual_filters=True,
                x264_preset="superfast",
                x264_crf=22,
            )

        self.assertEqual(plan.codec, "libx264")
        self.assertEqual(plan.options["preset"], "superfast")
        self.assertEqual(plan.options["crf"], 22)

    def test_auto_mode_uses_qsv_when_nvenc_is_unavailable(self) -> None:
        with (
            patch.dict("os.environ", {"AUTODUB_VIDEO_ENCODER": "auto"}, clear=False),
            patch(
                "app.utils.video_encoder.hardware_encoder_usable",
                side_effect=lambda _binary, codec: codec == "h264_qsv",
            ),
        ):
            plan = select_video_encoder(
                ffmpeg_binary="ffmpeg",
                has_visual_filters=True,
                x264_preset="superfast",
                x264_crf=22,
            )

        self.assertEqual(plan.codec, "h264_qsv")
        self.assertEqual(plan.options["preset"], "veryfast")
        self.assertEqual(plan.options["global_quality"], 23)

    def test_qsv_error_detection_does_not_match_an_unrelated_file_path(self) -> None:
        self.assertFalse(
            is_hardware_encoder_runtime_error(
                RuntimeError("Unable to open C:/tmpmfx/subtitle.srt"),
                "h264_qsv",
            )
        )
        self.assertTrue(
            is_hardware_encoder_runtime_error(
                RuntimeError("[h264_qsv @ 0001] Error initializing an MFX session"),
                "h264_qsv",
            )
        )


if __name__ == "__main__":
    unittest.main()
