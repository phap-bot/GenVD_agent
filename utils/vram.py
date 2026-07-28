from __future__ import annotations

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)


class VRAMManager:
    """Strict GPU memory cleanup for very small NVIDIA cards."""

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
        return (
            "cuda out of memory" in text
            or "cublas_status_alloc_failed" in text
            or "cuda error: out of memory" in text
        )
