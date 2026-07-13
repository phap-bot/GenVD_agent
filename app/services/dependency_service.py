from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any


class DependencyService:
    """Runtime dependency checks for local media/AI tooling."""

    def check_binary(self, name: str) -> bool:
        return self.resolve_binary(name) is not None

    def resolve_binary(self, name: str) -> str | None:
        found = shutil.which(name)
        if found:
            return found

        self._add_known_ffmpeg_dirs_to_path()
        return shutil.which(name)

    @staticmethod
    def check_python_module(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def status(self) -> dict[str, bool | str | None]:
        cuda = self.cuda_status()
        return {
            "ffmpeg_binary": self.check_binary("ffmpeg"),
            "ffprobe_binary": self.check_binary("ffprobe"),
            "ffmpeg_path": self.resolve_binary("ffmpeg"),
            "ffprobe_path": self.resolve_binary("ffprobe"),
            "ffmpeg_python": self.check_python_module("ffmpeg"),
            "torch": self.check_python_module("torch"),
            "whisperx": self.check_python_module("whisperx"),
            "yt_dlp": self.check_python_module("yt_dlp"),
            "vieneu": self.check_python_module("vieneu"),
            "cuda_available": cuda["available"],
            "cuda_device": cuda["device"],
        }

    def cuda_status(self) -> dict[str, Any]:
        try:
            import torch

            available = bool(torch.cuda.is_available())
            return {
                "available": available,
                "device": torch.cuda.get_device_name(0) if available else None,
            }
        except Exception:
            return {"available": False, "device": None}

    def require_cuda(self) -> None:
        status = self.cuda_status()
        if not status["available"]:
            raise RuntimeError(
                "CUDA is not available in this Python environment. "
                "Install the NVIDIA driver and a CUDA-enabled PyTorch wheel, then restart the backend."
            )

    def require_ffmpeg(self) -> None:
        missing = [
            name
            for name in ("ffmpeg", "ffprobe")
            if not self.check_binary(name)
        ]
        if missing:
            raise RuntimeError(
                "Missing FFmpeg executable(s): "
                + ", ".join(missing)
                + ". Install FFmpeg and add its bin folder to PATH, then restart the backend."
            )
        if not self.check_python_module("ffmpeg"):
            raise RuntimeError("Missing Python package ffmpeg-python. Install it with `pip install ffmpeg-python`.")

    def _add_known_ffmpeg_dirs_to_path(self) -> None:
        for directory in self._known_ffmpeg_dirs():
            if not directory.exists():
                continue
            if not (directory / "ffmpeg.exe").exists():
                continue

            current_path = os.environ.get("PATH", "")
            directory_text = str(directory)
            paths = current_path.split(os.pathsep) if current_path else []
            if directory_text not in paths:
                os.environ["PATH"] = directory_text + os.pathsep + current_path

    def _known_ffmpeg_dirs(self) -> list[Path]:
        candidates: list[Path] = []

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            candidates.extend(winget_packages.glob("Gyan.FFmpeg*/*/bin"))

        candidates.extend(
            [
                Path("C:/ffmpeg/bin"),
                Path("C:/Program Files/ffmpeg/bin"),
                Path("C:/Program Files/Gyan/FFmpeg/bin"),
            ]
        )
        return candidates
