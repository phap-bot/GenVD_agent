"""Filesystem helpers for request-scoped temporary files."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


class UploadSizeLimitError(ValueError):
    """Raised when a streamed upload exceeds its endpoint-specific byte limit."""


def make_request_id() -> str:
    return uuid4().hex


def ensure_output_dir(path: str | Path = "output") -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_filename(filename: str | None, fallback_suffix: str = ".mp4") -> str:
    if not filename:
        return f"upload{fallback_suffix}"
    return os.path.basename(filename).replace(" ", "_")


async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        while chunk := await upload_file.read(1024 * 1024):
            buffer.write(chunk)
    return destination


async def save_upload_file_limited(upload_file: UploadFile, destination: Path, *, max_bytes: int) -> Path:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := await upload_file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadSizeLimitError(f"Upload exceeds the {max_bytes}-byte limit.")
                buffer.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination
