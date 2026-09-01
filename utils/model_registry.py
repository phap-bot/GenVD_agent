from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from utils.model_cache import (
    configure_model_cache,
    load_whisperx_align_model,
    load_whisperx_model,
)

logger = logging.getLogger("auto_dubbing.model_registry")

ModelMover = Callable[[Any, str], None]


def _load_cached_demucs_model(model_name: str) -> Any | None:
    """Load a complete Demucs Hugging Face snapshot without a network probe."""
    try:
        import yaml
        from demucs.apply import BagOfModels
        from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name, load_safetensors_model
        from huggingface_hub import snapshot_download

        repo_id = f"{DEFAULT_NAMESPACE}/{hf_repo_name(model_name)}"
        snapshot = Path(snapshot_download(repo_id, local_files_only=True))
        definition_path = snapshot / f"{model_name}.yaml"
        if not definition_path.is_file():
            return None
        with definition_path.open("r", encoding="utf-8") as handle:
            bag = yaml.safe_load(handle)
        model_paths = [snapshot / f"{signature}.safetensors" for signature in bag["models"]]
        if not all(path.is_file() for path in model_paths):
            return None
        models = [load_safetensors_model(path) for path in model_paths]
        logger.info("model_registry.demucs.local_cache_hit model=%s snapshot=%s", model_name, snapshot)
        return BagOfModels(models, bag.get("weights"), bag.get("segment"))
    except Exception:
        logger.debug("model_registry.demucs.local_cache_miss model=%s", model_name, exc_info=True)
        return None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cleanup_cuda_cache() -> None:
    gc.collect()
    if "torch" not in sys.modules:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        logger.warning("model_registry.cuda_cleanup.skipped error=%s", exc)


def _torch_device(device: str):
    import torch

    if device.startswith("cuda"):
        return torch.device(device if ":" in device else "cuda:0")
    return torch.device("cpu")


def _move_torch_object(obj: Any, device: str) -> bool:
    if obj is None or not hasattr(obj, "to"):
        return False
    try:
        obj.to(_torch_device(device))
        if hasattr(obj, "eval"):
            obj.eval()
        return True
    except Exception:
        logger.warning(
            "model_registry.torch_move.failed type=%s device=%s",
            type(obj).__name__,
            device,
            exc_info=True,
        )
        return False


def _move_vad_model(vad_model: Any, device: str) -> None:
    if vad_model is None:
        return

    vad_pipeline = getattr(vad_model, "vad_pipeline", None)
    if vad_pipeline is not None and _move_torch_object(vad_pipeline, device):
        return

    _move_torch_object(vad_model, device)


def _move_whisperx_asr(model: Any, device: str) -> None:
    ctranslate_model = getattr(getattr(model, "model", None), "model", None)
    if ctranslate_model is not None:
        try:
            if device.startswith("cuda"):
                ctranslate_model.load_model(keep_cache=True)
            elif getattr(ctranslate_model, "device", None) == "cuda":
                ctranslate_model.unload_model(to_cpu=True)
        except Exception:
            logger.warning(
                "model_registry.asr_move.failed device=%s",
                device,
                exc_info=True,
            )

    _move_vad_model(getattr(model, "vad_model", None), device)


def _move_align_payload(payload: tuple[Any, dict[str, Any]], device: str) -> None:
    align_model, _metadata = payload
    _move_torch_object(align_model, device)


def _move_vieneu_model(model: Any, device: str) -> None:
    if getattr(model, "backend", None) == "onnx":
        return

    engine = getattr(model, "engine", None)
    if engine is None:
        _move_torch_object(model, device)
        return

    target = _torch_device(device)
    moved = False
    for attr in ("model", "audio_tokenizer", "backbone", "codec"):
        component = getattr(engine, attr, None)
        if component is not None:
            moved = _move_torch_object(component, str(target)) or moved

    if moved and hasattr(engine, "device"):
        engine.device = target


def _prepare_vieneu_for_cuda(model: Any) -> None:
    """Avoid SDPA CUDA kernel crashes seen in Qwen3-based VieNeu inference."""
    if getattr(model, "backend", None) != "pytorch":
        return

    semantic_backbone = getattr(getattr(getattr(model, "engine", None), "model", None), "semantic_backbone", None)
    config = getattr(semantic_backbone, "config", None)
    if config is not None and getattr(config, "_attn_implementation", None) != "eager":
        config._attn_implementation = "eager"
        logger.info("model_registry.vieneu.attention_backend backend=eager")


@dataclass(frozen=True)
class _HandleConfig:
    key: tuple[Any, ...]
    loader: Callable[[], Any]
    mover: ModelMover
    offload_to_cpu: bool


class _ModelHandle:
    def __init__(self, config: _HandleConfig) -> None:
        self.config = config
        self.model: Any | None = None
        self.lock = threading.RLock()
        self.device: str | None = None

    @contextmanager
    def acquire(
        self,
        *,
        device: str,
        prepare: Callable[[Any], None] | None = None,
    ) -> Iterator[Any]:
        with self.lock:
            if self.model is None:
                logger.info("model_registry.load.start key=%s", self.config.key)
                self.model = self.config.loader()
                self.device = device
                logger.info("model_registry.load.done key=%s", self.config.key)

            if self.device != device:
                self.config.mover(self.model, device)
                self.device = device
            if prepare is not None:
                prepare(self.model)

            try:
                yield self.model
            finally:
                if self.config.offload_to_cpu and device.startswith("cuda"):
                    self.offload()

    def offload(self) -> None:
        """Move an idle resident model to CPU without destroying its handle."""
        with self.lock:
            if self.model is None or not (self.device or "").startswith("cuda"):
                return
            self.config.mover(self.model, "cpu")
            self.device = "cpu"
            _cleanup_cuda_cache()
            logger.info("model_registry.offloaded key=%s idle_device=cpu", self.config.key)

    def close(self) -> None:
        with self.lock:
            self.model = None
            self.device = None


class ModelRegistry:
    """Process-wide model singleton cache with CPU offload after use."""

    def __init__(self) -> None:
        self._handles: dict[tuple[Any, ...], _ModelHandle] = {}
        self._lock = threading.RLock()
        self.cpu_offload = _env_flag("AUTODUB_CPU_OFFLOAD", False)

    def _handle(
        self,
        key: tuple[Any, ...],
        loader: Callable[[], Any],
        mover: ModelMover,
    ) -> _ModelHandle:
        with self._lock:
            handle = self._handles.get(key)
            if handle is None:
                handle = _ModelHandle(
                    _HandleConfig(
                        key=key,
                        loader=loader,
                        mover=mover,
                        offload_to_cpu=self.cpu_offload,
                    )
                )
                self._handles[key] = handle
            return handle

    def _evict_other_resident_models(self, active: _ModelHandle) -> None:
        """Keep at most one model on CUDA while preserving all CPU handles.

        This allows ``AUTODUB_CPU_OFFLOAD=0`` to make repeated renders fast,
        while still protecting low-VRAM machines when the pipeline switches
        from ASR to alignment, TTS or Demucs.
        """
        with self._lock:
            handles = [handle for handle in self._handles.values() if handle is not active]
        for handle in handles:
            handle.offload()

    @contextmanager
    def acquire_whisperx_asr(
        self,
        whisperx_module: Any,
        *,
        whisper_arch: str,
        device: str,
        compute_type: str,
        language: str | None,
        beam_size: int = 1,
    ) -> Iterator[Any]:
        normalized_beam_size = max(1, min(10, int(beam_size)))
        normalized_language = (language or "auto").strip().lower()
        key = ("whisperx-asr", whisper_arch, device, compute_type, normalized_beam_size, normalized_language)

        def loader() -> Any:
            return load_whisperx_model(
                whisperx_module,
                whisper_arch,
                device=device,
                compute_type=compute_type,
                language=None if normalized_language == "auto" else normalized_language,
                asr_options={"beam_size": normalized_beam_size},
            )

        def prepare(model: Any) -> None:
            model.tokenizer = None

        handle = self._handle(key, loader, _move_whisperx_asr)
        self._evict_other_resident_models(handle)
        with handle.acquire(device=device, prepare=prepare) as model:
            yield model

    @contextmanager
    def acquire_whisperx_align(
        self,
        whisperx_module: Any,
        *,
        language_code: str,
        device: str,
    ) -> Iterator[tuple[Any, dict[str, Any]]]:
        key = ("whisperx-align", language_code, device)

        def loader() -> tuple[Any, dict[str, Any]]:
            return load_whisperx_align_model(
                whisperx_module,
                language_code=language_code,
                device=device,
            )

        handle = self._handle(key, loader, _move_align_payload)
        self._evict_other_resident_models(handle)
        with handle.acquire(device=device) as payload:
            yield payload

    @contextmanager
    def acquire_vieneu(
        self,
        *,
        device: str,
        backend: str,
    ) -> Iterator[Any]:
        try:
            configured_batch_size = int(os.environ.get("AUTODUB_VIENEU_BATCH_SIZE", "16"))
        except ValueError:
            configured_batch_size = 16
        max_batch_size = max(1, min(32, configured_batch_size))
        key = ("vieneu", "v3turbo", device, backend, max_batch_size)

        def loader() -> Any:
            try:
                from vieneu import Vieneu
            except ImportError as exc:
                raise RuntimeError("VieNeu-TTS is not installed. Install it with `pip install vieneu`.") from exc

            # Prefer an explicitly downloaded local checkpoint so deployment
            # never silently re-downloads a gated/large model from Hugging Face.
            backbone_repo = (
                os.environ.get("AUTODUB_VIENEU_BACKBONE_REPO")
                or os.environ.get("AUTODUB_VIENEU_MODEL_DIR")
                or "pnnbao-ump/VieNeu-TTS-v3-Turbo"
            )
            kwargs: dict[str, Any] = {
                "device": device,
                "backend": backend,
                "backbone_repo": backbone_repo,
            }
            model_subfolder = os.environ.get("AUTODUB_VIENEU_MODEL_SUBFOLDER")
            if model_subfolder:
                kwargs["model_subfolder"] = model_subfolder
            onnx_dir = os.environ.get("AUTODUB_VIENEU_ONNX_DIR")
            if onnx_dir:
                kwargs["onnx_dir"] = onnx_dir
            kwargs["max_batch_size"] = max_batch_size
            try:
                return Vieneu(mode="v3turbo", **kwargs)
            except TypeError as exc:
                # Older VieNeu builds may not expose the static-batching
                # constructor option; inference helpers still fall back to
                # one-at-a-time generation for those versions.
                if "max_batch_size" not in str(exc):
                    raise
                kwargs.pop("max_batch_size", None)
                return Vieneu(mode="v3turbo", **kwargs)

        handle = self._handle(key, loader, _move_vieneu_model)
        self._evict_other_resident_models(handle)
        with handle.acquire(device=device, prepare=_prepare_vieneu_for_cuda) as model:
            yield model

    @contextmanager
    def acquire_demucs(
        self,
        *,
        model_name: str = "htdemucs",
        device: str = "cpu",
    ) -> Iterator[Any]:
        key = ("demucs", model_name)

        def loader() -> Any:
            # This loader is also used by PipelineManager without importing the
            # legacy pipeline module first. Pin the cache here so Demucs never
            # falls back to the user profile or redownloads model weights.
            configure_model_cache()
            try:
                from demucs.pretrained import get_model
            except ImportError as exc:
                raise RuntimeError("Demucs is required for vocal separation. Install demucs: `pip install demucs`") from exc
            model = _load_cached_demucs_model(model_name)
            if model is None:
                model = get_model(model_name)
            model.eval()
            return model

        def mover(model: Any, target_device: str) -> None:
            _move_torch_object(model, target_device)

        handle = self._handle(key, loader, mover)
        self._evict_other_resident_models(handle)
        with handle.acquire(device=device) as model:
            yield model

    def startup(self) -> None:
        configure_model_cache()
        preload = os.environ.get("AUTODUB_PRELOAD_MODELS", "none").strip().lower()
        if preload in {"", "0", "false", "no", "off", "none"}:
            logger.info("model_registry.preload.skip")
            return

        if preload in {"1", "true", "yes", "on"}:
            preload = "asr"
        preload_targets = {item.strip() for item in preload.replace(",", " ").split() if item.strip()}

        try:
            import torch
            import whisperx

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for model preload")
            device = "cuda"
            if "asr" in preload_targets or "all" in preload_targets:
                arch = os.environ.get("AUTODUB_PRELOAD_ASR_MODEL", "base")
                compute_type = os.environ.get("AUTODUB_PRELOAD_COMPUTE_TYPE", "float16")
                with self.acquire_whisperx_asr(
                    whisperx,
                    whisper_arch=arch,
                    device=device,
                    compute_type=compute_type,
                    language=None,
                ):
                    pass

            align_languages = [
                item.strip()
                for item in os.environ.get("AUTODUB_PRELOAD_ALIGN_LANGS", "").split(",")
                if item.strip()
            ]
            for language_code in align_languages:
                with self.acquire_whisperx_align(
                    whisperx,
                    language_code=language_code,
                    device=device,
                ):
                    pass

            if "tts" in preload_targets or "all" in preload_targets:
                with self.acquire_vieneu(device="cuda", backend="pytorch"):
                    pass
        except Exception:
            logger.warning("model_registry.preload.failed", exc_info=True)

    def shutdown(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for handle in handles:
            handle.close()
        _cleanup_cuda_cache()
        logger.info("model_registry.shutdown.done")

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "cpu_offload": self.cpu_offload,
                "loaded_models": [str(key) for key in self._handles],
            }


model_registry = ModelRegistry()
