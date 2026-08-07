from __future__ import annotations

import unittest

from app.utils.vram import VRAMManager
from utils.model_registry import _prepare_vieneu_for_cuda


class _Config:
    _attn_implementation = "sdpa"


class _SemanticBackbone:
    config = _Config()


class _EngineModel:
    semantic_backbone = _SemanticBackbone()


class _Engine:
    model = _EngineModel()


class _VieneuModel:
    backend = "pytorch"
    engine = _Engine()


class CudaRecoveryTests(unittest.TestCase):
    def test_cuda_unknown_error_is_classified_as_cuda_failure(self) -> None:
        exc = RuntimeError("CUDA error: unknown error")

        self.assertTrue(VRAMManager.is_cuda_error(exc))
        self.assertFalse(VRAMManager.is_cuda_oom(exc))

    def test_plain_unknown_error_is_not_classified_as_cuda_failure(self) -> None:
        exc = RuntimeError("unknown error")

        self.assertFalse(VRAMManager.is_cuda_error(exc))

    def test_vieneu_cuda_prepare_uses_eager_attention(self) -> None:
        model = _VieneuModel()
        model.engine.model.semantic_backbone.config._attn_implementation = "sdpa"

        _prepare_vieneu_for_cuda(model)

        self.assertEqual(model.engine.model.semantic_backbone.config._attn_implementation, "eager")


if __name__ == "__main__":
    unittest.main()
