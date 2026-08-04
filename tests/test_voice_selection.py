from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.models.schemas import PipelineConfig, RenderScriptRequest
from utils.tts_voice import (
    encode_cloned_vieneu_voice,
    infer_stable_cloned_vieneu_audio,
    infer_stable_vieneu_audio,
    resolve_vieneu_voice,
)


class FakeModel:
    def __init__(self) -> None:
        self._preset_voices = {"Trúc Ly": {}, "Ngọc Linh": {}}
        self.infer_calls: list[dict] = []
        self.encoded_paths: list[str] = []

    def encode_reference(self, path: str):
        self.encoded_paths.append(path)
        return [101, 202, 303]

    def infer(self, **kwargs):
        self.infer_calls.append(kwargs)
        return [0.0]


class FakeV31Model(FakeModel):
    def encode_reference(self, path: str):
        self.encoded_paths.append(path)
        return [0.1, 0.2], [101, 202, 303]


def _render_payload(**overrides):
    payload = {
        "source_video_path": "/media/source.mp4",
        "copyright_confirmed": True,
        "copyright_source": "owned",
        "segments": [
            {
                "id": 0,
                "start": 0,
                "end": 1,
                "original_text": "xin chào",
                "translated_text": "xin chào",
                "voice_model": "Trúc Ly",
            }
        ],
    }
    payload.update(overrides)
    return payload


class VoiceSelectionTests(unittest.TestCase):
    def test_system_voice_resolution_is_strict(self) -> None:
        model = FakeModel()

        self.assertEqual(resolve_vieneu_voice(model, "Trúc Ly"), "Trúc Ly")
        self.assertEqual(resolve_vieneu_voice(model, "VieNeu - Trúc Ly"), "Trúc Ly")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            resolve_vieneu_voice(model, "Không tồn tại")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            resolve_vieneu_voice(model, "Ngọc Lan")

    def test_clone_reference_is_encoded_and_used_without_system_voice(self) -> None:
        model = FakeModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "voice.wav"
            reference.write_bytes(b"RIFF-reference")

            cloned_voice = encode_cloned_vieneu_voice(model, reference)
            infer_stable_cloned_vieneu_audio(model, "Nội dung mới", cloned_voice)

            self.assertEqual(model.encoded_paths, [str(reference.resolve())])
            self.assertIs(model.infer_calls[-1]["voice"]["codes"], cloned_voice.voice_payload["codes"])
            self.assertNotIn("ref_codes", model.infer_calls[-1])
            self.assertNotIn("ref_audio", model.infer_calls[-1])

    def test_vieneu_31_clone_uses_explicit_voice_payload(self) -> None:
        model = FakeV31Model()
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir) / "voice.wav"
            reference.write_bytes(b"RIFF-reference")

            cloned_voice = encode_cloned_vieneu_voice(model, reference)
            infer_stable_cloned_vieneu_audio(model, "Nội dung mới", cloned_voice)

            call = model.infer_calls[-1]
            self.assertEqual(call["voice"], cloned_voice.voice_payload)
            self.assertTrue(call["use_ref_codes"])
            self.assertNotIn("ref_audio", call)

    def test_system_inference_never_passes_clone_reference(self) -> None:
        model = FakeModel()

        infer_stable_vieneu_audio(model, "Nội dung mới", "Trúc Ly")

        self.assertEqual(model.infer_calls[-1]["voice"], "Trúc Ly")
        self.assertNotIn("ref_codes", model.infer_calls[-1])
        self.assertNotIn("ref_audio", model.infer_calls[-1])

    def test_clone_mode_requires_reference_path(self) -> None:
        with self.assertRaisesRegex(ValidationError, "clone_reference_audio_path"):
            PipelineConfig(voice_mode="clone")

        with self.assertRaisesRegex(ValidationError, "clone_reference_audio_path"):
            RenderScriptRequest(**_render_payload(voice_mode="clone"))

    def test_render_request_accepts_explicit_clone_reference(self) -> None:
        request = RenderScriptRequest(
            **_render_payload(
                voice_mode="clone",
                clone_reference_audio_path="/media/reference.wav",
            )
        )

        self.assertEqual(request.voice_mode, "clone")
        self.assertEqual(request.clone_reference_audio_path, "/media/reference.wav")


if __name__ == "__main__":
    unittest.main()
