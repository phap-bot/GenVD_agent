from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("auto_dubbing.model_cache")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ModelCachePaths:
    root: Path
    hf_home: Path
    hf_hub_cache: Path
    hf_assets_cache: Path
    transformers_cache: Path
    torch_home: Path
    whisperx_asr_cache: Path
    whisperx_align_cache: Path

    def as_log_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "hf_home": str(self.hf_home),
            "hf_hub_cache": str(self.hf_hub_cache),
            "hf_assets_cache": str(self.hf_assets_cache),
            "transformers_cache": str(self.transformers_cache),
            "torch_home": str(self.torch_home),
            "whisperx_asr_cache": str(self.whisperx_asr_cache),
            "whisperx_align_cache": str(self.whisperx_align_cache),
        }


_CONFIGURED_PATHS: ModelCachePaths | None = None


def configure_model_cache() -> ModelCachePaths:
    """Pin model caches to stable repo-local folders.

    This keeps heavyweight model files on disk between requests while still
    allowing each pipeline stage to unload models from VRAM after inference.
    """

    global _CONFIGURED_PATHS
    if _CONFIGURED_PATHS is not None:
        return _CONFIGURED_PATHS

    root = _env_path("AUTODUB_MODEL_ROOT", DEFAULT_MODEL_ROOT)
    hf_home = _env_path("HF_HOME", root / "huggingface")
    hf_hub_cache = _env_path("HF_HUB_CACHE", hf_home / "hub")
    hf_assets_cache = _env_path("HF_ASSETS_CACHE", hf_home / "assets")
    transformers_cache = _env_path("TRANSFORMERS_CACHE", hf_home / "transformers")
    torch_home = _env_path("TORCH_HOME", root / "torch")
    whisperx_asr_cache = _env_path("WHISPERX_ASR_CACHE", root / "whisperx" / "asr")
    whisperx_align_cache = _env_path("WHISPERX_ALIGN_CACHE", root / "whisperx" / "align")

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    _CONFIGURED_PATHS = ModelCachePaths(
        root=root,
        hf_home=hf_home,
        hf_hub_cache=hf_hub_cache,
        hf_assets_cache=hf_assets_cache,
        transformers_cache=transformers_cache,
        torch_home=torch_home,
        whisperx_asr_cache=whisperx_asr_cache,
        whisperx_align_cache=whisperx_align_cache,
    )
    logger.info("model_cache.configured %s", _CONFIGURED_PATHS.as_log_dict())
    return _CONFIGURED_PATHS


def load_whisperx_model(whisperx_module: Any, whisper_arch: str, **kwargs: Any) -> Any:
    cache_paths = configure_model_cache()
    kwargs.setdefault("download_root", str(cache_paths.whisperx_asr_cache))
    logger.info(
        "model_cache.whisperx_asr.load arch=%s download_root=%s",
        whisper_arch,
        kwargs.get("download_root"),
    )
    try:
        return whisperx_module.load_model(whisper_arch, **kwargs)
    except TypeError:
        logger.warning(
            "model_cache.whisperx_asr.download_root_unsupported arch=%s retrying_without_cache_arg",
            whisper_arch,
            exc_info=True,
        )
        kwargs.pop("download_root", None)
        return whisperx_module.load_model(whisper_arch, **kwargs)


def load_whisperx_align_model(
    whisperx_module: Any,
    *,
    language_code: str,
    device: str,
    **kwargs: Any,
) -> Any:
    cache_paths = configure_model_cache()
    kwargs.setdefault("model_dir", str(cache_paths.whisperx_align_cache))
    logger.info(
        "model_cache.whisperx_align.load language=%s device=%s model_dir=%s",
        language_code,
        device,
        kwargs.get("model_dir"),
    )
    try:
        return whisperx_module.load_align_model(language_code=language_code, device=device, **kwargs)
    except TypeError:
        logger.warning(
            "model_cache.whisperx_align.model_dir_unsupported language=%s retrying_without_cache_arg",
            language_code,
            exc_info=True,
        )
        kwargs.pop("model_dir", None)
        return whisperx_module.load_align_model(language_code=language_code, device=device, **kwargs)


def _env_path(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw).expanduser() if raw else fallback
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(resolved)
    return resolved
