from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from app.models.schemas import PipelineConfig, ShortVideoRenderRequest, TranscriptSegment
from app.services.checkpoint_service import CheckpointStore
from app.services.pipeline import AutoDubbingPipeline
from app.services.translation_service import TranslationService
from utils.language import detect_language
from utils.paraformer import _audio_duration_seconds


class AdvancedPipelineTests(unittest.TestCase):
    def test_segment_language_detection(self) -> None:
        self.assertEqual(detect_language("你好，世界。", fallback="en")[0], "zh")
        self.assertEqual(detect_language("Xin chào các bạn.", fallback="en")[0], "vi")
        self.assertEqual(detect_language("Hello everyone.", fallback="en")[0], "en")

    def test_explicit_chinese_source_is_authoritative_for_segment_labels(self) -> None:
        pipeline = AutoDubbingPipeline(PipelineConfig(source_language="zh"))
        annotated = pipeline._annotate_segment_languages([
            TranscriptSegment(id=0, start=0, end=2, text="That's right, Li Yusha arrived at Zainan.")
        ])
        self.assertEqual(annotated[0].language, "zh")

    def test_repeated_english_transcript_is_flagged_as_whisper_hallucination(self) -> None:
        pipeline = AutoDubbingPipeline(PipelineConfig(source_language="zh"))
        self.assertTrue(pipeline._asr_needs_retry([
            {"text": "of the South of the South of the South of the South"}
        ]))
        self.assertFalse(pipeline._asr_needs_retry([
            {"text": "你好，我们现在开始。"}
        ]))

        self.assertTrue(pipeline._asr_needs_retry([
            {"text": "000, 000, 000, 000, 000, 000, 000, 000"}
        ]))

    def test_paraformer_duration_fallback_reads_wav_length(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / "audio.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
            self.assertAlmostEqual(_audio_duration_seconds(audio), 1.0, places=3)

    def test_checkpoint_is_content_addressed_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "video.mp4"
            media.write_bytes(b"same bytes")
            store = CheckpointStore(media, root=str(Path(root) / "checkpoints"))
            path = store.save("asr", {"model": "base"}, [{"text": "hello"}])
            self.assertIsNotNone(path)
            self.assertEqual(store.load("asr", {"model": "base"}), [{"text": "hello"}])
            copied = Path(root) / "copy.mp4"
            copied.write_bytes(media.read_bytes())
            copied_store = CheckpointStore(copied, root=str(Path(root) / "checkpoints"))
            self.assertEqual(copied_store.load("asr", {"model": "base"}), [{"text": "hello"}])

    def test_translation_groups_mixed_languages_in_mock_mode(self) -> None:
        config = PipelineConfig(
            target_language="vi",
            mock_translation=True,
            translate_analysis=True,
            translate_review=False,
        )
        segments = [
            TranscriptSegment(id=0, start=0, end=2, text="你好", language="zh"),
            TranscriptSegment(id=1, start=2, end=4, text="Hello", language="en"),
        ]
        translated = TranslationService(config).translate(segments)
        self.assertEqual(len(translated), 2)
        self.assertEqual(translated[0].language, "zh")
        self.assertEqual(translated[1].language, "en")

    def test_short_render_requires_clone_reference(self) -> None:
        with self.assertRaises(ValueError):
            ShortVideoRenderRequest(media_id="abc", voice_mode="clone", segments=[])
        payload = ShortVideoRenderRequest(
            media_id="abc",
            voice_mode="clone",
            clone_reference_audio_path="/media/reference.wav",
            segments=[{"id": 0, "start": 0, "end": 1, "original_text": "a", "translated_text": "b"}],
        )
        self.assertEqual(payload.voice_mode, "clone")


if __name__ == "__main__":
    unittest.main()
