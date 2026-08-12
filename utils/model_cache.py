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


# Keep this mapping in sync with faster-whisper's public model aliases.  We
# resolve these aliases before importing/loading the model so an existing Hub
# snapshot can be used as a normal local directory.  Passing just ``base`` (or
# another alias) to faster-whisper always enters its Hub download path; with
# ``local_files_only=False`` that path performs a network request even when the
# requested revision is already cached.
_FASTER_WHISPER_REPOS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

_WHISPER_SNAPSHOT_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")


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
    cached_snapshot = resolve_cached_whisperx_model(cache_paths.whisperx_asr_cache, whisper_arch)
    model_source = str(cached_snapshot) if cached_snapshot is not None else whisper_arch
    logger.info(
        "model_cache.whisperx_asr.load arch=%s source=%s cache_hit=%s download_root=%s",
        whisper_arch,
        model_source,
        cached_snapshot is not None,
        kwargs.get("download_root"),
    )
    try:
        return whisperx_module.load_model(model_source, **kwargs)
    except TypeError:
        logger.warning(
            "model_cache.whisperx_asr.download_root_unsupported arch=%s retrying_without_cache_arg",
            whisper_arch,
            exc_info=True,
        )
        kwargs.pop("download_root", None)
        return whisperx_module.load_model(model_source, **kwargs)


def resolve_cached_whisperx_model(cache_root: Path | str, whisper_arch: str) -> Path | None:
    """Return a complete local CTranslate2 snapshot without touching the Hub.

    ``faster-whisper`` stores named models using the standard Hugging Face cache
    layout under ``download_root``.  Resolve the current ref first, then any
    other complete snapshot (useful after an interrupted cache update).  A
    direct local model directory is also accepted.
    """

    direct_path = Path(whisper_arch).expanduser()
    if direct_path.is_dir() and _is_complete_whisper_snapshot(direct_path):
        return direct_path.resolve()

    repo_id = _FASTER_WHISPER_REPOS.get(whisper_arch, whisper_arch if "/" in whisper_arch else None)
    if not repo_id:
        return None

    repo_cache = Path(cache_root) / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_cache / "snapshots"
    candidates: list[Path] = []

    ref_path = repo_cache / "refs" / "main"
    try:
        revision = ref_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        revision = ""
    if revision:
        candidates.append(snapshots_dir / revision)

    if snapshots_dir.is_dir():
        try:
            other_snapshots = sorted(
                (path for path in snapshots_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            other_snapshots = []
        candidates.extend(path for path in other_snapshots if path not in candidates)

    for candidate in candidates:
        if _is_complete_whisper_snapshot(candidate):
            logger.info(
                "model_cache.whisperx_asr.local_snapshot arch=%s repo=%s snapshot=%s",
                whisper_arch,
                repo_id,
                candidate,
            )
            return candidate.resolve()
    return None


def _is_complete_whisper_snapshot(path: Path) -> bool:
    return all((path / filename).is_file() for filename in _WHISPER_SNAPSHOT_REQUIRED_FILES)


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
