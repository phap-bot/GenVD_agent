from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


logger = logging.getLogger("auto_dubbing.workspace")


@dataclass(frozen=True)
class Workspace:
    request_id: str
    root: Path
    input_video: Path
    chunks_dir: Path
    output_dir: Path


class WorkspaceManager:
    """UUID-based temp workspace lifecycle."""

    _active_lock = threading.RLock()
    _active_workspaces: set[Path] = set()

    def __init__(
        self,
        temp_root: str | Path = "temp_workspace",
        output_root: str | Path = "output",
    ) -> None:
        self.temp_root = Path(temp_root)
        self.output_root = Path(output_root)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def create(self) -> Workspace:
        request_id = uuid4().hex
        root = self.temp_root / request_id
        chunks_dir = root / "chunks"
        output_dir = self.output_root
        chunks_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._active_lock:
            self._active_workspaces.add(root.resolve())
        return Workspace(
            request_id=request_id,
            root=root,
            input_video=root / "input.mp4",
            chunks_dir=chunks_dir,
            output_dir=output_dir,
        )

    async def save_upload(self, upload: UploadFile, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            while chunk := await upload.read(1024 * 1024):
                buffer.write(chunk)
        return destination

    def cleanup(self, workspace: Workspace) -> None:
        root = workspace.root.resolve()
        with self._active_lock:
            self._active_workspaces.discard(root)
        shutil.rmtree(root, ignore_errors=True)

    def evict_stale(self, max_age_seconds: float = 24 * 3600) -> int:
        """Remove abandoned request workspaces older than the retention window.

        Only direct children of ``temp_root`` are eligible. Workspaces created
        by this process are protected, so a daily sweep cannot interrupt an
        active render; abandoned workspaces from a crashed process are removed
        on the next sweep once they exceed the age limit.
        """
        root = self.temp_root.resolve()
        if not root.is_dir():
            return 0
        cutoff = time.time() - max(1.0, float(max_age_seconds))
        removed = 0
        with self._active_lock:
            active = set(self._active_workspaces)
        try:
            entries = list(root.iterdir())
        except OSError:
            return 0
        for entry in entries:
            try:
                resolved = entry.resolve()
                if resolved.parent != root or resolved in active or not entry.is_dir():
                    continue
                if entry.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(resolved)
                removed += 1
            except (OSError, RuntimeError):
                # A request may finish/remove the directory concurrently.
                continue
        if removed:
            logger.info("workspace.cleanup removed=%s root=%s max_age_s=%s", removed, root, max_age_seconds)
        return removed

    @classmethod
    def evict_stale_all(cls, max_age_seconds: float = 24 * 3600) -> dict[str, int]:
        """Sweep request workspaces and other bounded temporary media roots."""
        removed = {"temp_workspace": cls().evict_stale(max_age_seconds)}
        # Short Video uploads and source previews are durable only for the
        # current processing window; they must not accumulate indefinitely.
        for name, path in (
            ("short_video", Path("temp") / "short_video"),
            ("source_media", Path("temp") / "source_media"),
        ):
            removed[name] = cls._evict_directory_tree(path, max_age_seconds)
        total = sum(removed.values())
        if total:
            logger.info("temp.cleanup.summary removed=%s details=%s", total, removed)
        return removed

    @staticmethod
    def _evict_directory_tree(root_path: Path, max_age_seconds: float) -> int:
        root = root_path.resolve()
        if not root.is_dir():
            return 0
        cutoff = time.time() - max(1.0, float(max_age_seconds))
        removed = 0
        try:
            entries = list(root.iterdir())
        except OSError:
            return 0
        for entry in entries:
            try:
                resolved = entry.resolve()
                if resolved.parent != root or not entry.is_dir() or entry.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(resolved)
                removed += 1
            except (OSError, RuntimeError):
                continue
        return removed
