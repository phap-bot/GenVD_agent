from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils.model_cache import load_whisperx_model, resolve_cached_whisperx_model
from utils.model_registry import ModelRegistry


PROJECT_TEMP = Path(__file__).resolve().parents[1] / "temp"


def _write_complete_snapshot(path: Path) -> None:
    path.mkdir(parents=True)
    for filename in ("config.json", "model.bin", "tokenizer.json"):
        (path / filename).write_bytes(b"cached")


class _FakeWhisperX:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def load_model(self, model_source: str, **kwargs: object) -> object:
        self.calls.append((model_source, kwargs))
        return object()


class ModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        PROJECT_TEMP.mkdir(parents=True, exist_ok=True)
        self.temp_dir = TemporaryDirectory(prefix="model-cache-test-", dir=PROJECT_TEMP)
        self.cache_root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resolve_cached_whisperx_model_uses_main_snapshot(self) -> None:
        repo_cache = self.cache_root / "models--Systran--faster-whisper-base"
        snapshot = repo_cache / "snapshots" / "abc123"
        _write_complete_snapshot(snapshot)
        (repo_cache / "refs").mkdir()
        (repo_cache / "refs" / "main").write_text("abc123\n", encoding="utf-8")

        self.assertEqual(resolve_cached_whisperx_model(self.cache_root, "base"), snapshot.resolve())

    def test_resolve_cached_whisperx_model_skips_incomplete_current_ref(self) -> None:
        repo_cache = self.cache_root / "models--Systran--faster-whisper-base"
        incomplete = repo_cache / "snapshots" / "new-incomplete"
        incomplete.mkdir(parents=True)
        (incomplete / "config.json").write_text("{}", encoding="utf-8")
        complete = repo_cache / "snapshots" / "older-complete"
        _write_complete_snapshot(complete)
        (repo_cache / "refs").mkdir()
        (repo_cache / "refs" / "main").write_text("new-incomplete", encoding="utf-8")

        self.assertEqual(resolve_cached_whisperx_model(self.cache_root, "base"), complete.resolve())

    def test_load_whisperx_model_passes_cached_snapshot_as_local_directory(self) -> None:
        snapshot = (
            self.cache_root
            / "models--Systran--faster-whisper-base"
            / "snapshots"
            / "abc123"
        )
        _write_complete_snapshot(snapshot)
        fake_whisperx = _FakeWhisperX()
        cache_paths = SimpleNamespace(whisperx_asr_cache=self.cache_root)

        with patch("utils.model_cache.configure_model_cache", return_value=cache_paths):
            loaded = load_whisperx_model(fake_whisperx, "base", device="cuda")

        self.assertIsNotNone(loaded)
        self.assertEqual(
            fake_whisperx.calls,
            [
                (
                    str(snapshot.resolve()),
                    {"device": "cuda", "download_root": str(self.cache_root)},
                )
            ],
        )

    def test_load_whisperx_model_keeps_alias_when_cache_is_missing(self) -> None:
        fake_whisperx = _FakeWhisperX()
        cache_paths = SimpleNamespace(whisperx_asr_cache=self.cache_root)

        with patch("utils.model_cache.configure_model_cache", return_value=cache_paths):
            load_whisperx_model(fake_whisperx, "base", device="cuda")

        self.assertEqual(fake_whisperx.calls[0][0], "base")

    def test_registry_applies_beam_size_at_model_load(self) -> None:
        fake_pipeline = SimpleNamespace(tokenizer=object(), model=None, vad_model=None)
        registry = ModelRegistry()
        with patch(
            "utils.model_registry.load_whisperx_model",
            return_value=fake_pipeline,
        ) as load_model:
            with registry.acquire_whisperx_asr(
                object(),
                whisper_arch="base",
                device="cpu",
                compute_type="int8",
                language=None,
                beam_size=1,
            ):
                pass

        self.assertEqual(load_model.call_args.kwargs["asr_options"], {"beam_size": 1})

    def test_registry_reuses_loaded_handle_for_repeated_requests(self) -> None:
        fake_pipeline = SimpleNamespace(tokenizer=object(), model=None, vad_model=None)
        registry = ModelRegistry()
        registry.cpu_offload = False
        with patch(
            "utils.model_registry.load_whisperx_model",
            return_value=fake_pipeline,
        ) as load_model:
            for _ in range(2):
                with registry.acquire_whisperx_asr(
                    object(),
                    whisper_arch="base",
                    device="cpu",
                    compute_type="int8",
                    language=None,
                    beam_size=5,
                ):
                    pass

        self.assertEqual(load_model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
