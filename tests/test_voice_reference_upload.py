from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import stream_router
from app.services.voice_reference_service import (
    VOICE_REFERENCE_FRAMES,
    VOICE_REFERENCE_SAMPLE_RATE,
    VoiceReferenceService,
    VoiceReferenceValidationError,
)


def _wav_bytes(duration_seconds: float, *, sample_rate: int = 48_000, channels: int = 2) -> bytes:
    frame_count = round(duration_seconds * sample_rate)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        # A deterministic non-silent signal is enough to exercise decode,
        # channel mixing and resampling without external fixture files.
        pattern = b"\x00\x10\x00\xf0" if channels == 2 else b"\x00\x10"
        wav_file.writeframes(pattern * frame_count)
    return stream.getvalue()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class VoiceReferenceServiceTests(unittest.TestCase):
    def test_ffmpeg_is_forced_to_local_wav_and_time_bounded(self) -> None:
        service = VoiceReferenceService(ffmpeg_binary="ffmpeg")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "selected.wav"
            destination = root / "canonical.wav"
            source.write_bytes(_wav_bytes(3.0))

            def emulate_ffmpeg(command, **_kwargs):
                Path(command[-1]).write_bytes(
                    _wav_bytes(3.0, sample_rate=VOICE_REFERENCE_SAMPLE_RATE, channels=1)
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch(
                "app.services.voice_reference_service.subprocess.run",
                side_effect=emulate_ffmpeg,
            ) as run:
                service.canonicalize(source, destination)

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-protocol_whitelist") + 1], "file")
        self.assertEqual(command[command.index("-f") + 1], "wav")
        self.assertEqual(run.call_args.kwargs["timeout"], 30.0)

    def test_ffmpeg_timeout_does_not_publish_partial_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "selected.wav"
            destination = root / "canonical.wav"
            source.write_bytes(_wav_bytes(3.0))
            service = VoiceReferenceService(ffmpeg_binary="ffmpeg")
            with (
                patch(
                    "app.services.voice_reference_service.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30),
                ),
                self.assertRaisesRegex(VoiceReferenceValidationError, "Timed out"),
            ):
                service.canonicalize(source, destination)

            self.assertFalse(destination.exists())

    def test_canonicalizes_exact_clip_to_three_second_mono_pcm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "selected.wav"
            destination = root / "canonical.wav"
            source.write_bytes(_wav_bytes(3.0))

            metadata = VoiceReferenceService().canonicalize(source, destination)

            self.assertEqual(metadata.duration_seconds, 3.0)
            self.assertEqual(metadata.sample_rate, VOICE_REFERENCE_SAMPLE_RATE)
            self.assertEqual(metadata.frame_count, VOICE_REFERENCE_FRAMES)
            self.assertEqual(len(metadata.sha256), 64)
            with wave.open(str(destination), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), VOICE_REFERENCE_SAMPLE_RATE)
                self.assertEqual(wav_file.getnframes(), VOICE_REFERENCE_FRAMES)

    def test_rejects_short_or_full_length_source_instead_of_silently_trimming(self) -> None:
        for duration in (2.9, 3.1, 8.0):
            with self.subTest(duration=duration), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "source.wav"
                destination = root / "canonical.wav"
                source.write_bytes(_wav_bytes(duration))

                with self.assertRaisesRegex(VoiceReferenceValidationError, "selected 3-second clip"):
                    VoiceReferenceService().canonicalize(source, destination)

                self.assertFalse(destination.exists())

    def test_rejects_corrupt_file_with_audio_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "fake.wav"
            source.write_bytes(b"not audio")

            with self.assertRaises(VoiceReferenceValidationError):
                VoiceReferenceService().canonicalize(source, root / "canonical.wav")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class VoiceReferenceRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(stream_router)
        self.client = TestClient(app)

    def test_endpoint_saves_only_canonical_clip_and_returns_selection_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.api.routes.VOICE_REFERENCE_DIR", Path(temp_dir)
        ):
            response = self.client.post(
                "/api/voice-reference",
                files={"audio": ("chosen-cut.wav", _wav_bytes(3.0), "audio/wav")},
                data={
                    "clip_duration_seconds": "3",
                    "selection_start_seconds": "4.25",
                    "selection_end_seconds": "7.25",
                },
            )

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["duration_seconds"], 3.0)
            self.assertEqual(payload["sample_rate"], VOICE_REFERENCE_SAMPLE_RATE)
            self.assertEqual(payload["frame_count"], VOICE_REFERENCE_FRAMES)
            self.assertEqual(payload["selection_start_seconds"], 4.25)
            self.assertEqual(payload["selection_end_seconds"], 7.25)
            saved_files = list(Path(temp_dir).glob("*_voice_reference.wav"))
            self.assertEqual(len(saved_files), 1)
            VoiceReferenceService().validate_canonical(saved_files[0])

    def test_endpoint_rejects_original_full_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.api.routes.VOICE_REFERENCE_DIR", Path(temp_dir)
        ):
            response = self.client.post(
                "/api/voice-reference",
                files={"audio": ("full-source.wav", _wav_bytes(8.0), "audio/wav")},
                data={"clip_duration_seconds": "3"},
            )

            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_endpoint_enforces_bounded_upload(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("app.api.routes.VOICE_REFERENCE_DIR", Path(temp_dir)),
            patch.dict("os.environ", {"AUTODUB_VOICE_REFERENCE_MAX_BYTES": "64"}),
        ):
            response = self.client.post(
                "/api/voice-reference",
                files={"audio": ("selected.wav", _wav_bytes(3.0), "audio/wav")},
                data={"clip_duration_seconds": "3"},
            )

            self.assertEqual(response.status_code, 413, response.text)
            self.assertEqual(list(Path(temp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
