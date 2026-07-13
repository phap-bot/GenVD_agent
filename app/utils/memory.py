"""Utilities for aggressive GPU memory cleanup between model stages."""

from __future__ import annotations

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)


class VRAMManager:
    """Centralized cleanup for low-VRAM sequential inference.

    The pipeline runs on 4GB VRAM, so every service must release model
    references and empty CUDA cache before the next model is loaded.
    """

    @staticmethod
    def clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("CUDA cache cleared")
        except Exception as exc:
            logger.warning("CUDA cleanup skipped: %s", exc)

    @staticmethod
    def collect_garbage() -> None:
        gc.collect()

    @classmethod
    def cleanup(cls) -> None:
        cls.collect_garbage()
        cls.clear_cuda_cache()

    @classmethod
    def release_model(cls, model_instance: Any | None) -> None:
        if model_instance is not None:
            del model_instance
        cls.cleanup()

