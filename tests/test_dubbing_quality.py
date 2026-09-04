from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.models.schemas import DubbingScriptSegment, PipelineConfig, TranscriptSegment, WordTimestamp
from app.services.pipeline import AutoDubbingPipeline
from app.services.translation_service import TranslationService
from app.services.vocal_separation_service import VocalSeparationResult
from app.utils.workspace import WorkspaceManager
from utils.translation import _translation_batches, translate_segments, translate_text


def _pipeline(**overrides) -> AutoDubbingPipeline:
    return AutoDubbingPipeline(
        PipelineConfig(
            copyright_confirmed=True,
            copyright_source="owned",
            **overrides,
        )
    )


class SentenceSegmentationTests(unittest.TestCase):
    def test_wordless_long_chinese_transcript_is_time_chunked(self) -> None:
        pipeline = _pipeline(source_language="zh")
        segments = pipeline._normalize_segments([{
            "start": 0.0,
            "end": 120.0,
            "text": "这是一个没有标点和词级时间戳的中文长段落" * 20,
        }])
        self.assertGreater(len(segments), 10)
        self.assertLessEqual(max(segment.end - segment.start for segment in segments), 8.0)

    def test_word_timestamps_split_on_complete_sentences_not_old_three_second_limit(self) -> None:
        pipeline = _pipeline()
        raw = [
            {
                "start": 0.0,
                "end": 9.0,
                "text": "This sentence stays intact, despite its pause. Next scene starts here.",
                "words": [
                    {"word": "This", "start": 0.0, "end": 0.7},
                    {"word": "sentence", "start": 0.7, "end": 1.5},
                    {"word": "stays", "start": 1.5, "end": 2.2},
                    {"word": "intact,", "start": 2.2, "end": 3.2},
                    {"word": "despite", "start": 3.5, "end": 4.3},
                    {"word": "its", "start": 4.3, "end": 4.7},
                    {"word": "pause.", "start": 4.7, "end": 5.5},
                    {"word": "Next", "start": 5.7, "end": 6.2},
                    {"word": "scene", "start": 6.2, "end": 6.8},
                    {"word": "starts", "start": 6.8, "end": 7.5},
                    {"word": "here.", "start": 7.5, "end": 9.0},
                ],
            }
        ]

        segments = pipeline._normalize_segments(raw)

        self.assertEqual([segment.text for segment in segments], [
            "This sentence stays intact, despite its pause.",
            "Next scene starts here.",
        ])
        self.assertEqual((segments[0].start, segments[0].end), (0.0, 5.5))
        self.assertEqual((segments[1].start, segments[1].end), (5.7, 9.0))

    def test_fragments_are_stitched_until_original_sentence_terminator(self) -> None:
        pipeline = _pipeline(source_language="en")
        fragments = [
            TranscriptSegment(id=0, start=0.0, end=2.0, text="A complete thought can span"),
            TranscriptSegment(id=1, start=2.1, end=4.2, text="several ASR fragments before it ends."),
            TranscriptSegment(id=2, start=4.4, end=6.0, text="A new sentence follows."),
        ]

        stitched = pipeline._source_timeline(fragments)

        self.assertEqual(len(stitched), 2)
        self.assertEqual(stitched[0].text, "A complete thought can span several ASR fragments before it ends.")
        self.assertEqual((stitched[0].start, stitched[0].end), (0.0, 4.2))

    def test_tts_does_not_merge_two_finished_sentences(self) -> None:
        pipeline = _pipeline()
        timeline = [
            TranscriptSegment(id=0, start=0.0, end=1.5, text="First sentence."),
            TranscriptSegment(id=1, start=1.6, end=3.0, text="Second sentence."),
        ]
        script = [
            DubbingScriptSegment(id=0, start=0.0, end=1.5, original_text="A.", translated_text="First sentence."),
            DubbingScriptSegment(id=1, start=1.6, end=3.0, original_text="B.", translated_text="Second sentence."),
        ]

        groups = pipeline._build_tts_groups(timeline, script)

        self.assertEqual(len(groups), 2)
        self.assertEqual([group.text for group in groups], ["First sentence.", "Second sentence."])

    def test_tts_merges_long_touching_continuation_cues(self) -> None:
        pipeline = _pipeline()
        timeline = [
            TranscriptSegment(id=0, start=18.0, end=26.0, text="Trong vali không có hành lý, cần giải quyết"),
            TranscriptSegment(id=1, start=26.0, end=33.0, text="rắc rối này ngay lập tức"),
        ]
        script = [
            DubbingScriptSegment(id=0, start=18.0, end=26.0, original_text="A", translated_text=timeline[0].text),
            DubbingScriptSegment(id=1, start=26.0, end=33.0, original_text="B", translated_text=timeline[1].text),
        ]

        groups = pipeline._build_tts_groups(timeline, script)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].segment_count, 2)
        self.assertEqual(groups[0].start, 18.0)
        self.assertEqual(groups[0].end, 33.0)

    def test_touching_cues_do_not_reserve_an_artificial_minimum_gap(self) -> None:
        pipeline = _pipeline(
            soft_timing_fit=True,
            timing_max_drift_s=1.5,
            timing_min_gap_s=0.12,
        )

        # A few milliseconds can come from timestamp quantization.  It is a
        # continuous cue boundary, not a real scene pause, so it may be used
        # by the preceding voice track.
        self.assertAlmostEqual(pipeline._timing_fit_duration(0.0, 8.0, 8.02), 8.02)
        self.assertAlmostEqual(pipeline._timing_fit_duration(0.0, 8.0, 7.999), 8.0)

    def test_short_tts_fit_uses_ffmpeg_for_natural_slowdown_instead_of_padding(self) -> None:
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "short.wav"
            with wave.open(str(source), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b"\x01\x00" * round(0.9 * 24000))

            # 0.9s -> 1.0s is a safe natural stretch. The numpy path must
            # defer to the atempo path so it does not append a silent tail.
            self.assertFalse(
                pipeline._fit_audio_duration_with_numpy(source, root / "fitted.wav", 1.0)
            )


class TimelineTranslationTests(unittest.TestCase):
    def test_repeated_korean_filler_short_circuits_translation_api(self) -> None:
        repeated_source = "흥," * 12
        with (
            patch("utils.translation._translate_9router_text_with_model_fallback") as router_mock,
            patch("utils.translation._translate_google_gtx_cached") as gtx_mock,
        ):
            translated = translate_text(
                repeated_source,
                source_language="auto",
                target_language="vi",
                provider="9router",
                model="model",
            )

        self.assertEqual(translated, "Hừm")
        router_mock.assert_not_called()
        gtx_mock.assert_not_called()

    def test_repeated_fallback_output_is_collapsed_to_one_spoken_sound(self) -> None:
        from utils.translation import _fallback_if_bad_translation

        repeated_fallback = "Hứ, " * 12
        with patch(
            "utils.translation._translate_google_gtx_cached",
            return_value=repeated_fallback,
        ) as gtx_mock:
            translated = _fallback_if_bad_translation(
                "你好吗?",
                "你好吗?",
                source="auto",
                target="vi",
                timeout=5.0,
            )

        self.assertEqual(translated, "Hứ")
        gtx_mock.assert_called_once()

    def test_source_fragment_does_not_trigger_google_fallback(self) -> None:
        from utils.translation import _fallback_if_bad_translation

        with patch("utils.translation._translate_google_gtx_cached") as gtx_mock:
            translated = _fallback_if_bad_translation(
                "到了晚上阿翔在清洗氣局突然聽到聲響,",
                "Đến tối, A Hương bỗng nghe thấy tiếng động ở góc phòng,",
                source="auto",
                target="vi",
                timeout=5.0,
            )

        self.assertEqual(translated, "Đến tối, A Hương bỗng nghe thấy tiếng động ở góc phòng,")
        gtx_mock.assert_not_called()

    def test_translation_request_retries_connection_reset_at_most_three_times(self) -> None:
        from utils.translation import _with_translation_request_retries

        calls = 0

        def flaky_request() -> str:
            nonlocal calls
            calls += 1
            if calls <= 3:
                raise ConnectionResetError(10054, "connection reset")
            return "ok"

        with (
            patch(
                "utils.translation._config_value",
                side_effect=lambda name: "3"
                if name == "AUTODUB_TRANSLATION_CONNECTION_REFUSED_RETRIES"
                else None,
            ),
            patch("utils.translation._connection_refused_retry_delay", return_value=0.0),
        ):
            result = _with_translation_request_retries(
                flaky_request,
                model="google-gtx",
                source="auto",
                target="vi",
                batch_size=1,
                endpoint="google_gtx",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, 4)

    def test_empty_gateway_response_is_retried_before_error(self) -> None:
        from utils.translation import TranslationResponseError, _parse_chat_completion_body, _with_translation_request_retries

        calls = 0

        def empty_response() -> str:
            nonlocal calls
            calls += 1
            raise TranslationResponseError("empty")

        with (
            patch(
                "utils.translation._config_value",
                side_effect=lambda name: "3"
                if name == "AUTODUB_TRANSLATION_CONNECTION_REFUSED_RETRIES"
                else None,
            ),
            patch("utils.translation._connection_refused_retry_delay", return_value=0.0),
            self.assertRaises(TranslationResponseError),
        ):
            _with_translation_request_retries(
                empty_response,
                model="model",
                source="auto",
                target="vi",
                batch_size=24,
                endpoint="9router",
            )

        self.assertEqual(calls, 4)
        with self.assertRaises(TranslationResponseError):
            _parse_chat_completion_body("", "application/json")

    def test_batch_shorten_retries_empty_gateway_response_before_heuristic_fallback(self) -> None:
        from utils.translation import TranslationResponseError, _shorten_9router_segments_cached

        class EmptyResponse:
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b""

        _shorten_9router_segments_cached.cache_clear()
        with (
            patch("utils.translation.urlopen", return_value=EmptyResponse()) as urlopen_mock,
            patch(
                "utils.translation._config_value",
                side_effect=lambda name: "3"
                if name == "AUTODUB_TRANSLATION_CONNECTION_REFUSED_RETRIES"
                else None,
            ),
            patch("utils.translation._connection_refused_retry_delay", return_value=0.0),
            self.assertRaises(TranslationResponseError),
        ):
            _shorten_9router_segments_cached(
                ("A complete sentence.",),
                ("A complete sentence.",),
                (2.0,),
                (6,),
                "vi",
                "model",
                "http://localhost:20128/v1",
                "",
                5.0,
            )

        self.assertEqual(urlopen_mock.call_count, 4)

    def test_structured_request_falls_back_when_gateway_rejects_json_mode(self) -> None:
        from utils.translation import TranslationResponseFormatUnsupported, _with_structured_json_fallback

        modes: list[bool] = []

        def operation(use_json_response_format: bool) -> str:
            modes.append(use_json_response_format)
            if use_json_response_format:
                raise TranslationResponseFormatUnsupported()
            return "prompt-json-result"

        result = _with_structured_json_fallback(
            operation,
            model="model",
            source="auto",
            target="vi",
            batch_size=2,
        )

        self.assertEqual(result, "prompt-json-result")
        self.assertEqual(modes, [True, False])

    def test_batch_shorten_switches_model_after_non_json_content(self) -> None:
        from utils.translation import TranslationResponseError, _shorten_9router_segments_with_model_fallback

        with (
            patch("utils.translation._translation_model_attempts", return_value=["bad-model", "good-model"]),
            patch(
                "utils.translation._shorten_9router_segments_cached",
                side_effect=[
                    TranslationResponseError("Translation model returned non-JSON content", retryable=False),
                    ["Câu đã rút gọn đầy đủ."],
                ],
            ) as shorten_mock,
        ):
            result = _shorten_9router_segments_with_model_fallback(
                ("Câu đầy đủ cần rút gọn.",),
                source_texts=("Câu đầy đủ cần rút gọn.",),
                target_durations=(2.0,),
                max_words=(6,),
                target="vi",
                selected_model="bad-model",
                base_url="http://localhost:20128/v1",
                api_key="",
                timeout=5.0,
                context="",
            )

        self.assertEqual(result, ["Câu đã rút gọn đầy đủ."])
        self.assertEqual([call.args[5] for call in shorten_mock.call_args_list], ["bad-model", "good-model"])

    def test_translation_batch_filters_known_repetition_before_gateway_request(self) -> None:
        from utils.translation import _translate_9router_segments_with_model_fallback

        repeated_source = "흥," * 12
        with patch(
            "utils.translation._translate_9router_segments_cached",
            return_value=["Bản dịch bình thường."],
        ) as gateway_mock:
            translated = _translate_9router_segments_with_model_fallback(
                (repeated_source, "A normal sentence."),
                target_durations=(1.0, 2.0),
                source_intervals=((0.0, 1.0), (1.0, 3.0)),
                source="auto",
                target="vi",
                selected_model="model",
                base_url="http://localhost:20128/v1",
                api_key="",
                timeout=5.0,
            )

        self.assertEqual(translated, ["Hừm", "Bản dịch bình thường."])
        self.assertEqual(gateway_mock.call_args.args[0], ("A normal sentence.",))

    def test_translation_batches_use_word_and_line_limits(self) -> None:
        with patch(
                "utils.translation._config_value",
                side_effect=lambda name: {
                "AUTODUB_TRANSLATION_BATCH_WORDS": "100",
                "AUTODUB_TRANSLATION_BATCH_CHARS": "10000",
            }.get(name),
        ):
            batches = _translation_batches(
                ["one " * 60, "two " * 50, "three " * 10],
                max_items=100,
            )

        self.assertEqual([len(batch) for batch in batches], [1, 2])

    def test_translation_batches_run_concurrently_but_restore_source_order(self) -> None:
        calls: list[tuple[str, ...]] = []

        def fake_batch(texts, **kwargs):
            calls.append(texts)
            time.sleep(0.03 if texts[0] == "first" else 0.005)
            return [f"vi {text}" for text in texts]

        with (
            patch("utils.translation._translation_concurrency", return_value=2),
            patch("utils.translation._translate_9router_segments_with_model_fallback", side_effect=fake_batch),
        ):
            translated = translate_segments(
                ["first", "second", "third", "fourth"],
                target_durations=[1.0, 1.0, 1.0, 1.0],
                source_intervals=[(0.0, 1.0), (1.1, 2.0), (2.1, 3.0), (3.1, 4.0)],
                source_language="en",
                target_language="vi",
                provider="9router",
                model="model",
                batch_size=2,
            )

        self.assertEqual(translated, ["vi first", "vi second", "vi third", "vi fourth"])
        self.assertEqual(sorted(calls), [("first", "second"), ("third", "fourth")])

    def test_translation_keeps_one_to_one_timeline_and_passes_duration_budget(self) -> None:
        segments = [
            TranscriptSegment(id=0, start=0.0, end=2.0, text="One."),
            TranscriptSegment(id=1, start=2.2, end=5.7, text="Two."),
        ]
        config = PipelineConfig(target_language="vi")

        with patch(
            "app.services.translation_service.translate_segments",
            return_value=["Mot.", "Hai."],
        ) as translate_mock:
            translated = TranslationService(config).translate(segments)

        self.assertEqual(len(translated), 2)
        self.assertEqual([(item.start, item.end) for item in translated], [(0.0, 2.0), (2.2, 5.7)])
        self.assertEqual([item.text for item in translated], ["Mot.", "Hai."])
        self.assertEqual(translate_mock.call_args.kwargs["target_durations"], [2.0, 3.5])

    def test_batch_translation_prompt_contains_per_segment_time_budget(self) -> None:
        from utils.translation import _translate_9router_segments_cached

        class Response:
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                content = json.dumps({"translations": [{"id": 0, "text": "Ban dich."}]})
                return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

        _translate_9router_segments_cached.cache_clear()
        with patch("utils.translation.urlopen", return_value=Response()) as urlopen_mock:
            translated = _translate_9router_segments_cached(
                ("Original sentence.",),
                (2.0,),
                "en",
                "vi",
                "model",
                "http://localhost:20128/v1",
                "",
                5.0,
                source_intervals=((10.0, 12.0),),
            )

        request_payload = json.loads(urlopen_mock.call_args.args[0].data.decode("utf-8"))
        user_payload = json.loads(request_payload["messages"][1]["content"])
        self.assertEqual(translated, ["Ban dich."])
        self.assertEqual(request_payload["response_format"], {"type": "json_object"})
        self.assertEqual(user_payload["segments"][0]["start_seconds"], 10.0)
        self.assertEqual(user_payload["segments"][0]["end_seconds"], 12.0)
        self.assertEqual(user_payload["segments"][0]["target_duration_seconds"], 2.0)
        self.assertEqual(user_payload["segments"][0]["max_spoken_words"], 6)

    def test_coherence_review_uses_ordered_batch_requests_with_edge_context(self) -> None:
        from utils.translation import _review_9router_segments_cached, review_translation_coherence

        class Response:
            headers = {"Content-Type": "application/json"}

            def __init__(self, count: int):
                self.count = count

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                content = json.dumps({
                    "segments": [
                        {"id": index, "text": f"Đoạn {index} đã được nối mạch."}
                        for index in range(self.count)
                    ]
                })
                return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

        _review_9router_segments_cached.cache_clear()
        requests = []

        def fake_urlopen(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            requests.append(payload)
            user_payload = json.loads(payload["messages"][1]["content"])
            return Response(len(user_payload["segments"]))

        with (
            patch("utils.translation.urlopen", side_effect=fake_urlopen),
            patch("utils.translation._translation_concurrency", return_value=1),
        ):
            reviewed = review_translation_coherence(
                ["Bản một.", "Bản hai.", "Bản ba."],
                target_durations=[8.0, 8.0, 8.0],
                source_texts=["Nguồn một.", "Nguồn hai.", "Nguồn ba."],
                target_language="vi",
                provider="9router",
                model="model",
                batch_size=2,
            )

        self.assertEqual(len(requests), 2)
        self.assertEqual(reviewed, [
            "Đoạn 0 đã được nối mạch.",
            "Đoạn 1 đã được nối mạch.",
            "Đoạn 0 đã được nối mạch.",
        ])
        self.assertIn("Following translated context", requests[0]["messages"][0]["content"])

    def test_timing_review_batches_all_over_budget_segments_once(self) -> None:
        segments = [
            TranscriptSegment(id=0, start=0.0, end=1.0, text="Source one."),
            TranscriptSegment(id=1, start=1.0, end=2.0, text="Source two."),
            TranscriptSegment(id=2, start=2.0, end=5.0, text="Source three."),
        ]
        config = PipelineConfig(
            target_language="vi",
            translate_review=True,
            translate_cps_budget=12.5,
            translate_batch_size=40,
        )
        with (
            patch(
                "app.services.translation_service.translate_segments",
                return_value=[
                    "Cau thu nhat dang bi dai qua ngan sach.",
                    "Cau thu hai cung dang bi dai qua ngan sach.",
                    "Cau nay vua du.",
                ],
            ),
            patch(
                "app.services.translation_service.shorten_segments_for_duration",
                return_value=["Cau mot.", "Cau hai."],
            ) as shorten_batch,
        ):
            translated = TranslationService(config).translate(segments)

        shorten_batch.assert_called_once()
        self.assertEqual(shorten_batch.call_args.args[0], [
            "Cau thu nhat dang bi dai qua ngan sach.",
            "Cau thu hai cung dang bi dai qua ngan sach.",
        ])
        self.assertEqual(shorten_batch.call_args.kwargs["batch_size"], 40)
        self.assertEqual([item.text for item in translated], ["Cau mot.", "Cau hai.", "Cau nay vua du."])

    def test_short_rhetorical_question_is_not_compressed_into_a_fragment(self) -> None:
        from utils.translation import _postprocess_translation

        self.assertEqual(
            _postprocess_translation("你不是在做事嗎?", "Cậu chẳng phải đang", "vi"),
            "Chẳng phải cậu đang làm việc sao?",
        )

        segment = TranscriptSegment(id=0, start=0.0, end=1.3, text="你不是在做事嗎?")
        config = PipelineConfig(
            target_language="vi",
            translate_review=True,
            translate_cps_budget=12.5,
        )
        with patch(
            "app.services.translation_service.translate_segments",
            return_value=["Chẳng phải cậu đang làm việc sao?"],
        ), patch(
            "app.services.translation_service.shorten_segments_for_duration",
        ) as shorten_batch:
            translated = TranslationService(config).translate([segment])

        shorten_batch.assert_not_called()
        self.assertEqual(translated[0].text, "Chẳng phải cậu đang làm việc sao?")

    def test_timing_review_rejects_model_fragment_for_complete_question(self) -> None:
        segment = TranscriptSegment(id=0, start=0.0, end=1.3, text="你不是在做事嗎?")
        original = "Chẳng phải cậu đang làm việc rất chăm chỉ sao?"
        config = PipelineConfig(
            target_language="vi",
            translate_review=True,
            translate_cps_budget=12.5,
        )
        with patch(
            "app.services.translation_service.translate_segments",
            return_value=[original],
        ), patch(
            "app.services.translation_service.shorten_segments_for_duration",
            return_value=["Cậu chẳng phải đang"],
        ):
            translated = TranslationService(config).translate([segment])

        self.assertEqual(translated[0].text, original)

    def test_batch_shorten_uses_one_9router_request_for_multiple_lines(self) -> None:
        from utils.translation import _shorten_9router_segments_cached, shorten_segments_for_duration

        class Response:
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                content = json.dumps({
                    "segments": [
                        {"id": 0, "text": "Cau mot gon."},
                        {"id": 1, "text": "Cau hai gon."},
                    ]
                })
                return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

        _shorten_9router_segments_cached.cache_clear()
        with patch("utils.translation.urlopen", return_value=Response()) as urlopen_mock:
            shortened = shorten_segments_for_duration(
                ["Cau mot rat dai can rut gon.", "Cau hai rat dai can rut gon."],
                target_durations=[1.0, 1.0],
                source_texts=["Source one.", "Source two."],
                provider="9router",
                model="model",
                max_words=[3, 3],
                batch_size=40,
            )

        self.assertEqual(urlopen_mock.call_count, 1)
        self.assertEqual(shortened, ["Cau mot gon.", "Cau hai gon."])


class TTSFitTests(unittest.TestCase):
    def test_numpy_fast_path_refuses_to_truncate_long_spoken_audio(self) -> None:
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "long.wav"
            destination = root / "fitted.wav"
            with wave.open(str(source), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(b"\x01\x00" * 48000)

            handled = pipeline._fit_audio_duration_with_numpy(source, destination, 1.0)

            self.assertFalse(handled)
            self.assertFalse(destination.exists())

    def test_cache_key_invalidates_pre_fix_truncated_audio(self) -> None:
        pipeline = _pipeline()
        key = pipeline._tts_cache_key("Read every word.", "system", "voice", None, 1.0)
        legacy_payload = json.dumps(
            {
                "text": "Read every word.",
                "voice_mode": "system",
                "voice_model": "voice",
                "clone_ref": "",
                "duration": 1.0,
            },
            sort_keys=True,
        )
        import hashlib

        legacy_key = hashlib.sha256(legacy_payload.encode()).hexdigest()[:16]
        self.assertNotEqual(key, legacy_key)


class AtomicCacheTests(unittest.TestCase):
    def test_failed_cache_copy_never_replaces_previous_complete_wav(self) -> None:
        pipeline = _pipeline()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            destination = root / "cache.wav"
            source.write_bytes(b"new-complete-audio")
            destination.write_bytes(b"old-complete-audio")

            def interrupted_copy(_source, temporary):
                Path(temporary).write_bytes(b"partial")
                raise OSError("simulated disk interruption")

            with patch("app.services.pipeline.shutil.copy2", side_effect=interrupted_copy):
                with self.assertRaisesRegex(OSError, "simulated disk interruption"):
                    pipeline._atomic_copy_file(source, destination)

            self.assertEqual(destination.read_bytes(), b"old-complete-audio")
            self.assertEqual(list(root.glob("*.tmp")), [])


class AudioGraphTests(unittest.TestCase):
    def test_separation_extracts_full_band_stereo_instead_of_asr_mono(self) -> None:
        pipeline = _pipeline(vocal_separation=True)

        class OutputNode:
            def __init__(self, destination: str) -> None:
                self.destination = Path(destination)

            def overwrite_output(self):
                return self

            def run(self, **kwargs):
                self.destination.write_bytes(b"stereo-audio")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = WorkspaceManager(root / "workspaces", root / "output").create()
            video = root / "video.mp4"
            video.write_bytes(b"source-video")
            captured = {}
            ffmpeg_module = MagicMock()
            input_node = MagicMock()

            def capture_output(destination, **kwargs):
                captured.update(kwargs)
                return OutputNode(destination)

            input_node.output.side_effect = capture_output
            ffmpeg_module.input.return_value = input_node

            def fake_separate(audio_path, output_dir, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                self.assertEqual(kwargs.get("device"), "cpu")
                vocals = output_dir / "vocals.wav"
                accompaniment = output_dir / "accompaniment.wav"
                vocals.write_bytes(b"vocals")
                accompaniment.write_bytes(b"background")
                self.assertEqual(audio_path.name, "separation_source.wav")
                return VocalSeparationResult(vocals, accompaniment)

            cache_dir = root / "stem-cache"
            cache_dir.mkdir()
            with (
                patch.dict("os.environ", {"AUTODUB_DEMUCS_DEVICE": "cuda"}),
                patch.object(pipeline, "_has_audio_stream", return_value=True),
                patch.object(pipeline, "_ffmpeg", return_value=ffmpeg_module),
                patch.object(
                    pipeline,
                    "_stem_cache_paths",
                    return_value=(cache_dir / "acc.wav", cache_dir / "vocals.wav"),
                ),
                patch.object(pipeline, "_evict_stale_stem_cache"),
                patch.object(pipeline.dependencies, "cuda_status", return_value={"available": False}),
                patch(
                    "app.services.vocal_separation_service.VocalSeparationService.separate",
                    side_effect=fake_separate,
                ),
            ):
                accompaniment = pipeline._ensure_accompaniment_audio(workspace, video)

        self.assertEqual(captured["ac"], 2)
        self.assertEqual(captured["ar"], "44100")
        self.assertEqual(captured["acodec"], "pcm_s16le")
        self.assertIsNotNone(accompaniment)

    def test_clean_accompaniment_is_sidechain_ducked_and_limited(self) -> None:
        import ffmpeg

        pipeline = _pipeline(burn_subtitles=False, vocal_separation=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            accompaniment = root / "accompaniment.wav"
            output = root / "output.mp4"
            for path in (video, tts, accompaniment):
                path.write_bytes(b"x")

            captured = {}

            def capture(command, stage):
                captured["args"] = command.compile()

            with patch.object(pipeline, "_run_ffmpeg_command", side_effect=capture):
                pipeline._render_video(video, root / "unused.srt", tts, output, accompaniment_path=accompaniment)

        graph = " ".join(str(item) for item in captured["args"])
        self.assertIn("sidechaincompress", graph)
        self.assertIn("ratio=12", graph)
        self.assertIn("volume=0.92", graph)
        self.assertIn("alimiter", graph)
        self.assertIn("-preset superfast", graph)
        self.assertIn("-crf 22", graph)
        self.assertIn("-movflags +faststart", graph)

    def test_source_mix_respects_original_vocal_and_accompaniment_gains(self) -> None:
        pipeline = _pipeline(
            burn_subtitles=False,
            vocal_separation=True,
            original_vocal_gain=0.3,
            accompaniment_gain=1.2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            accompaniment = root / "accompaniment.wav"
            vocals = root / "vocals.wav"
            output = root / "output.mp4"
            for path in (video, tts, accompaniment, vocals):
                path.write_bytes(b"x")

            captured = {}

            with patch.object(
                pipeline,
                "_run_ffmpeg_command",
                side_effect=lambda command, stage: captured.update(args=command.compile()),
            ):
                pipeline._render_video(
                    video,
                    root / "unused.srt",
                    tts,
                    output,
                    accompaniment_path=accompaniment,
                    original_vocal_path=vocals,
                )

        graph = " ".join(str(item) for item in captured["args"])
        self.assertIn("volume=1.104", graph)
        self.assertIn("volume=0.3", graph)
        self.assertIn("amix=", graph)
        self.assertIn("inputs=2", graph)

    def test_source_mix_rejects_requested_vocal_gain_without_vocal_stem(self) -> None:
        pipeline = _pipeline(
            burn_subtitles=False,
            vocal_separation=True,
            original_vocal_gain=0.5,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            accompaniment = root / "accompaniment.wav"
            output = root / "output.mp4"
            for path in (video, tts, accompaniment):
                path.write_bytes(b"x")

            with self.assertRaisesRegex(RuntimeError, "no separated vocal track"):
                pipeline._render_video(
                    video,
                    root / "unused.srt",
                    tts,
                    output,
                    accompaniment_path=accompaniment,
                )

    def test_video_service_uses_the_same_source_mix_controls(self) -> None:
        from app.services.video_service import VideoService

        config = PipelineConfig(
            copyright_confirmed=True,
            copyright_source="owned",
            burn_subtitles=False,
            vocal_separation=True,
            original_vocal_gain=0.4,
            accompaniment_gain=1.1,
        )
        service = VideoService(config)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            accompaniment = root / "accompaniment.wav"
            vocals = root / "vocals.wav"
            output = root / "output.mp4"
            for path in (video, tts, accompaniment, vocals):
                path.write_bytes(b"x")

            captured = {}
            with patch.object(
                service,
                "_run_ffmpeg_command",
                side_effect=lambda command, stage: captured.update(args=command.compile()),
            ):
                service._render_video(
                    video,
                    root / "unused.srt",
                    tts,
                    output,
                    accompaniment_path=accompaniment,
                    original_vocal_path=vocals,
                )

        graph = " ".join(str(item) for item in captured["args"])
        self.assertIn("volume=1.012", graph)
        self.assertIn("volume=0.4", graph)
        self.assertIn("inputs=2", graph)

    def test_invalid_x264_settings_fall_back_to_safe_defaults(self) -> None:
        pipeline = _pipeline(burn_subtitles=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            output = root / "output.mp4"
            video.write_bytes(b"x")
            tts.write_bytes(b"x")
            captured = {}

            with (
                patch.dict(
                    "os.environ",
                    {"AUTODUB_X264_PRESET": "invalid", "AUTODUB_X264_CRF": "invalid"},
                ),
                patch.object(
                    pipeline,
                    "_run_ffmpeg_command",
                    side_effect=lambda command, stage: captured.update(args=command.compile()),
                ),
            ):
                pipeline._render_video(video, root / "unused.srt", tts, output)

        graph = " ".join(str(item) for item in captured["args"])
        self.assertIn("-preset superfast", graph)
        self.assertIn("-crf 22", graph)

    def test_original_audio_fallback_is_kept_at_emergency_level(self) -> None:
        pipeline = _pipeline(burn_subtitles=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            output = root / "output.mp4"
            video.write_bytes(b"x")
            tts.write_bytes(b"x")
            captured = {}

            with patch.object(
                pipeline,
                "_run_ffmpeg_command",
                side_effect=lambda command, stage: captured.update(args=command.compile()),
            ):
                pipeline._render_video(video, root / "unused.srt", tts, output)

        graph = " ".join(str(item) for item in captured["args"])
        self.assertIn("ratio=20", graph)
        self.assertIn("volume=0.12", graph)

    def test_render_refuses_original_audio_when_separation_is_enabled(self) -> None:
        pipeline = _pipeline(burn_subtitles=False, vocal_separation=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            output = root / "output.mp4"
            video.write_bytes(b"x")
            tts.write_bytes(b"x")

            with self.assertRaisesRegex(RuntimeError, "prevent original voice bleed"):
                pipeline._render_video(video, root / "unused.srt", tts, output)

    def test_separation_failure_is_not_silently_downgraded(self) -> None:
        pipeline = _pipeline(vocal_separation=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = WorkspaceManager(root / "workspaces", root / "output").create()
            video = root / "video.mp4"
            video.write_bytes(b"source-video")
            cache_dir = root / "stem-cache"
            cache_dir.mkdir()

            with (
                patch.object(pipeline, "_has_audio_stream", return_value=True),
                patch.object(
                    pipeline,
                    "_stem_cache_paths",
                    return_value=(cache_dir / "acc.wav", cache_dir / "vocals.wav"),
                ),
                patch.object(pipeline, "_evict_stale_stem_cache"),
                patch.object(pipeline, "_ffmpeg", side_effect=RuntimeError("demucs unavailable")),
            ):
                with self.assertRaisesRegex(RuntimeError, "pipeline stopped|render stopped"):
                    pipeline._ensure_accompaniment_audio(workspace, video)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg executable is required")
    def test_sidechain_graph_executes_with_real_ffmpeg(self) -> None:
        pipeline = _pipeline(burn_subtitles=False, vocal_separation=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            tts = root / "tts.wav"
            accompaniment = root / "accompaniment.wav"
            output = root / "output.mp4"
            quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "check": True}
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=1",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(video),
                ],
                **quiet,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=1", str(tts)],
                **quiet,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=1", str(accompaniment)],
                **quiet,
            )

            pipeline._render_video(video, root / "unused.srt", tts, output, accompaniment_path=accompaniment)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
