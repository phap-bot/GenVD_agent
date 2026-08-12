from __future__ import annotations

"""Durable, file-backed checkpoints for ASR/OCR/translation stages."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class CheckpointStore:
    """Content-addressed stage cache.

    A checkpoint is keyed by media bytes (small sampled digest), stage and a
    config/input payload. It therefore survives workspace cleanup and backend
    restarts without accidentally reusing a result for a different file or
    model configuration.
    """

    def __init__(self, media_path: Path, *, root: str = "temp/checkpoints", enabled: bool = True) -> None:
        self.media_path = Path(media_path)
        self.root = Path(root)
        self.enabled = enabled
        self._media_digest: str | None = None

    @property
    def media_digest(self) -> str:
        if self._media_digest:
            return self._media_digest
        digest = hashlib.sha256()
        try:
            stat = self.media_path.stat()
            digest.update(str(stat.st_size).encode("ascii"))
            sample_size = 1024 * 1024
            with self.media_path.open("rb") as handle:
                offsets = (0, max(0, stat.st_size // 2 - sample_size // 2), max(0, stat.st_size - sample_size))
                for offset in offsets:
                    handle.seek(offset)
                    digest.update(handle.read(sample_size))
        except OSError:
            digest.update(str(self.media_path).encode("utf-8", errors="replace"))
        self._media_digest = digest.hexdigest()[:32]
        return self._media_digest

    def _path(self, stage: str, payload: Any) -> Path:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        config_digest = hashlib.sha256(encoded).hexdigest()[:24]
        safe_stage = "".join(char if char.isalnum() or char in "-_" else "_" for char in stage)
        return self.root / self.media_digest / f"{safe_stage}-{config_digest}.json"

    def load(self, stage: str, payload: Any) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(stage, payload)
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            if document.get("version") != 1:
                return None
            return document.get("data")
        except (OSError, ValueError, TypeError):
            return None

    def save(self, stage: str, payload: Any, data: Any) -> Path | None:
        if not self.enabled:
            return None
        path = self._path(stage, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"version": 1, "stage": stage, "media_digest": self.media_digest, "data": data}
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            return path
        finally:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
