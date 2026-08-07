from __future__ import annotations

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)


class VRAMManager:
    """Strict CUDA cleanup for 4GB VRAM systems."""

    _CUDA_ERROR_MARKERS = (
        "cuda error",
        "torch.acceleratorerror",
        "cublas_status",
        "cudnn_status",
        "device-side assert",
        "illegal memory access",
        "unspecified launch failure",
    )

    @staticmethod
    def cleanup() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("CUDA cache cleared")
        except Exception as exc:
            logger.warning("CUDA cleanup skipped: %s", exc)

    @classmethod
    def release(cls, *objects: Any) -> None:
        for obj in objects:
            if obj is not None:
                del obj
        cls.cleanup()

    @staticmethod
    def is_cuda_oom(exc: BaseException) -> bool:
        text = str(exc).lower()
        return "cuda out of memory" in text or "cublas_status_alloc_failed" in text

    @classmethod
    def is_cuda_error(cls, exc: BaseException) -> bool:
        text_parts: list[str] = []
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            text_parts.append(current.__class__.__module__.lower())
            text_parts.append(current.__class__.__name__.lower())
            text_parts.append(str(current).lower())
            current = current.__cause__ or current.__context__

        text = " ".join(text_parts)
        return any(marker in text for marker in cls._CUDA_ERROR_MARKERS)

    @classmethod
    def reset_after_cuda_error(cls) -> None:
        try:
            from utils.model_registry import model_registry

            model_registry.shutdown()
        except Exception as exc:
            logger.warning("CUDA model registry reset skipped: %s", exc)
        cls.cleanup()
