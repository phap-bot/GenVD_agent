from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from utils.model_cache import (
    configure_model_cache,
    load_whisperx_align_model,
    load_whisperx_model,
)

logger = logging.getLogger("auto_dubbing.model_registry")

ModelMover = Callable[[Any, str], None]


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
                logger.info("model_registry.load.done key=%s", self.config.key)

            self.config.mover(self.model, device)
            if prepare is not None:
                prepare(self.model)

            try:
                yield self.model
            finally:
                if self.config.offload_to_cpu and device.startswith("cuda"):
                    self.config.mover(self.model, "cpu")
                    _cleanup_cuda_cache()
                    logger.info("model_registry.offloaded key=%s idle_device=cpu", self.config.key)

    def close(self) -> None:
        with self.lock:
            self.model = None


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

    @contextmanager
    def acquire_whisperx_asr(
        self,
        whisperx_module: Any,
        *,
        whisper_arch: str,
        device: str,
        compute_type: str,
        language: str | None,
    ) -> Iterator[Any]:
        key = ("whisperx-asr", whisper_arch, device, compute_type)

        def loader() -> Any:
            return load_whisperx_model(
                whisperx_module,
                whisper_arch,
                device=device,
                compute_type=compute_type,
                language=None,
            )

        def prepare(model: Any) -> None:
            model.tokenizer = None

        handle = self._handle(key, loader, _move_whisperx_asr)
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
        with handle.acquire(device=device) as payload:
            yield payload

    @contextmanager
    def acquire_vieneu(
        self,
        *,
        device: str,
        backend: str,
    ) -> Iterator[Any]:
        key = ("vieneu", "v3turbo", device, backend)

        def loader() -> Any:
            try:
                from vieneu import Vieneu
            except ImportError as exc:
                raise RuntimeError("VieNeu-TTS is not installed. Install it with `pip install vieneu`.") from exc

            return Vieneu(mode="v3turbo", device=device, backend=backend)

        handle = self._handle(key, loader, _move_vieneu_model)
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
                compute_type = os.environ.get("AUTODUB_PRELOAD_COMPUTE_TYPE", "int8")
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
