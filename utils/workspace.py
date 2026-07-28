from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


@dataclass(frozen=True)
class Workspace:
    request_id: str
    root: Path
    input_video: Path
    chunks_dir: Path
    output_dir: Path
    output_video: Path


class WorkspaceManager:
    """UUID-scoped temporary workspace lifecycle."""

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
        chunks_dir.mkdir(parents=True, exist_ok=True)

        return Workspace(
            request_id=request_id,
            root=root,
            input_video=root / "input.mp4",
            chunks_dir=chunks_dir,
            output_dir=self.output_root,
            output_video=self.output_root / f"{request_id}_dubbed.mp4",
        )

    async def save_upload(self, upload: UploadFile, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            while chunk := await upload.read(1024 * 1024):
                buffer.write(chunk)
        return destination

    def cleanup(self, workspace: Workspace) -> None:
        root = workspace.root.resolve()
        temp_root = self.temp_root.resolve()
        if root == temp_root or temp_root not in root.parents:
            raise RuntimeError(f"Refusing to delete unsafe workspace path: {root}")
        shutil.rmtree(root, ignore_errors=True)
