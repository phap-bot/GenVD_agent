from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from app.models.schemas import PipelineConfig, RenderScriptRequest
from app.services.dependency_service import DependencyService
from app.services.vocal_separation_service import (
    VocalSeparationResult,
    VocalSeparationService,
    _load_audio,
    _resolve_torch_device,
    _resolve_demucs_chunk_seconds,
    _save_audio,
)
from utils.model_registry import ModelRegistry, _load_cached_demucs_model, model_registry


class VocalSeparationTests(unittest.TestCase):
    def test_cuda_request_falls_back_before_model_or_audio_moves(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False

        self.assertEqual(_resolve_torch_device(fake_torch, "cuda"), "cpu")

    def test_auto_device_uses_cuda_only_when_available(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        self.assertEqual(_resolve_torch_device(fake_torch, "auto"), "cuda")
        fake_torch.cuda.is_available.return_value = False
        self.assertEqual(_resolve_torch_device(fake_torch, "auto"), "cpu")

    def test_demucs_chunk_size_is_bounded(self) -> None:
        self.assertEqual(_resolve_demucs_chunk_seconds("not-a-number"), 60.0)
        self.assertEqual(_resolve_demucs_chunk_seconds(1), 10.0)
        self.assertEqual(_resolve_demucs_chunk_seconds(9_999), 600.0)

    def test_pcm_wave_fallback_loads_and_saves_without_torchaudio_backend(self) -> None:
        import wave
        import torch

        fake_torchaudio = MagicMock()
        fake_torchaudio.load.side_effect = RuntimeError("no audio backend")
        fake_torchaudio.save.side_effect = RuntimeError("no audio backend")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            saved = root / "saved.wav"
            with wave.open(str(source), "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44_100)
                wav_file.writeframes(b"\x00\x10\x00\xf0" * 256)

            waveform, sample_rate = _load_audio(torch, fake_torchaudio, source)
            _save_audio(torch, fake_torchaudio, saved, waveform, sample_rate)

            self.assertEqual(tuple(waveform.shape), (2, 256))
            self.assertEqual(sample_rate, 44_100)
            with wave.open(str(saved), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 2)
                self.assertEqual(wav_file.getframerate(), 44_100)
                self.assertEqual(wav_file.getnframes(), 256)

    def test_pipeline_config_vocal_separation_default_false(self) -> None:
        config = PipelineConfig()
        self.assertFalse(config.vocal_separation)

        config_enabled = PipelineConfig(vocal_separation=True)
        self.assertTrue(config_enabled.vocal_separation)

    def test_render_script_request_vocal_separation_default_false(self) -> None:
        payload = {
            "source_video_path": "/media/source.mp4",
            "copyright_confirmed": True,
            "copyright_source": "owned",
            "segments": [
                {
                    "id": 0,
                    "start": 0,
                    "end": 1,
                    "original_text": "hello",
                    "translated_text": "xin chào",
                }
            ],
        }
        req = RenderScriptRequest(**payload)
        self.assertFalse(req.vocal_separation)

        payload["vocal_separation"] = True
        req_enabled = RenderScriptRequest(**payload)
        self.assertTrue(req_enabled.vocal_separation)

    def test_dependency_service_has_demucs_check(self) -> None:
        service = DependencyService()
        status = service.status()
        self.assertIn("demucs", status)

    def test_demucs_registry_configures_project_cache_before_model_load(self) -> None:
        fake_model = MagicMock()
        registry = ModelRegistry()

        with (
            patch("utils.model_registry.configure_model_cache") as configure_cache,
            patch("utils.model_registry._load_cached_demucs_model", return_value=None),
            patch("demucs.pretrained.get_model", return_value=fake_model) as get_model,
        ):
            with registry.acquire_demucs(device="cpu") as loaded:
                self.assertIs(loaded, fake_model)

        configure_cache.assert_called_once_with()
        get_model.assert_called_once_with("htdemucs")

    def test_cached_demucs_loader_does_not_require_network(self) -> None:
        fake_model = MagicMock()
        fake_bag = MagicMock()
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir)
            (snapshot / "htdemucs.yaml").write_text("models: ['abc123']\n", encoding="utf-8")
            (snapshot / "abc123.safetensors").write_bytes(b"cached")

            with (
                patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download,
                patch("demucs.hf.load_safetensors_model", return_value=fake_model),
                patch("demucs.apply.BagOfModels", return_value=fake_bag),
            ):
                loaded = _load_cached_demucs_model("htdemucs")

        self.assertIs(loaded, fake_bag)
        download.assert_called_once_with("adefossez/HTDemucs", local_files_only=True)

    @patch("torchaudio.load")
    @patch("torchaudio.save")
    def test_vocal_separation_service_flow(self, mock_save, mock_load) -> None:
        import torch

        # Mock torchaudio.load -> returns (tensor, sample_rate)
        left = torch.linspace(-0.5, 0.5, 44100, dtype=torch.float32)
        mock_wav = torch.stack((left, left * 0.5))
        mock_load.return_value = (mock_wav, 44100)

        # Mock Demucs model
        fake_model = MagicMock()
        fake_model.samplerate = 44100
        fake_model.sources = ["drums", "bass", "other", "vocals"]

        mock_apply_model = MagicMock(return_value=torch.zeros((1, 4, 2, 44100), dtype=torch.float32))

        with tempfile.TemporaryDirectory() as temp_dir:
            input_wav = Path(temp_dir) / "source.wav"
            input_wav.write_bytes(b"dummy")
            out_dir = Path(temp_dir) / "output"

            # Create mock demucs module if demucs is not installed in local environment
            mock_demucs = MagicMock()
            mock_demucs.apply.apply_model = mock_apply_model
            
            with patch.dict(sys.modules, {"demucs": mock_demucs, "demucs.apply": mock_demucs.apply}):
                with patch.object(model_registry, "acquire_demucs") as mock_acquire:
                    mock_acquire.return_value.__enter__.return_value = fake_model

                    service = VocalSeparationService()
                    res = service.separate(input_wav, out_dir, device="cpu")

                    self.assertIsInstance(res, VocalSeparationResult)
                    self.assertEqual(res.vocals_path, out_dir / "vocals.wav")
                    self.assertEqual(res.accompaniment_path, out_dir / "accompaniment.wav")
                    self.assertEqual(mock_save.call_count, 2)
                    normalized_input = mock_apply_model.call_args.args[1]
                    self.assertEqual(tuple(normalized_input.shape), (1, 2, 44100))
                    self.assertAlmostEqual(float(normalized_input.mean()), 0.0, places=5)
                    self.assertEqual(mock_apply_model.call_args.kwargs["shifts"], 1)
                    self.assertEqual(mock_apply_model.call_args.kwargs["overlap"], 0.5)
                    saved_accompaniment = mock_save.call_args_list[1].args[1]
                    self.assertTrue(torch.allclose(saved_accompaniment, mock_wav, atol=1e-5))
                    self.assertEqual(saved_accompaniment.data_ptr(), mock_wav.data_ptr())

    @patch("torchaudio.load")
    @patch("torchaudio.save")
    def test_long_input_is_inferred_in_bounded_chunks(self, mock_save, mock_load) -> None:
        import torch

        sample_rate = 1_000
        mock_wav = torch.zeros((2, 2_500), dtype=torch.float32)
        mock_load.return_value = (mock_wav, sample_rate)
        fake_model = MagicMock()
        fake_model.samplerate = sample_rate
        fake_model.sources = ["drums", "bass", "other", "vocals"]

        def apply_chunk(_model, normalized, **_kwargs):
            return torch.zeros((1, 4, 2, normalized.shape[-1]), dtype=torch.float32)

        mock_apply_model = MagicMock(side_effect=apply_chunk)
        mock_demucs = MagicMock()
        mock_demucs.apply.apply_model = mock_apply_model

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            source.write_bytes(b"dummy")
            with (
                patch.dict(sys.modules, {"demucs": mock_demucs, "demucs.apply": mock_demucs.apply}),
                patch.object(model_registry, "acquire_demucs") as acquire,
                patch("app.services.vocal_separation_service._resolve_demucs_chunk_seconds", return_value=1.0),
            ):
                acquire.return_value.__enter__.return_value = fake_model
                result = VocalSeparationService().separate(source, root / "output", device="cpu")

        self.assertEqual(mock_apply_model.call_count, 3)
        self.assertEqual(tuple(mock_save.call_args_list[0].args[1].shape), (2, 2_500))
        self.assertEqual(tuple(mock_save.call_args_list[1].args[1].shape), (2, 2_500))
        self.assertEqual(result.vocals_path.name, "vocals.wav")


if __name__ == "__main__":
    unittest.main()
