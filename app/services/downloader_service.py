from __future__ import annotations

from pathlib import Path


class DownloaderService:
    """Download Douyin videos through yt-dlp.

    This service is CPU/network only. It does not load GPU models.
    """

    def download_douyin(self, url: str, destination_dir: Path, max_items: int = 10) -> list[Path]:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise RuntimeError("yt-dlp is required for Douyin downloads. Install it with `pip install yt-dlp`.") from exc

        destination_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(destination_dir / "%(uploader|douyin)s_%(id)s.%(ext)s")

        options = {
            "outtmpl": output_template,
            "format": "mp4/bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "noplaylist": max_items == 1,
            "playlistend": max_items,
            "quiet": True,
            "no_warnings": True,
        }

        before = set(destination_dir.glob("*"))
        with YoutubeDL(options) as ydl:
            ydl.download([url])

        after = set(destination_dir.glob("*"))
        downloaded = [
            path
            for path in sorted(after - before)
            if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
        ]
        if not downloaded:
            downloaded = [
                path
                for path in sorted(destination_dir.glob("*"))
                if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
            ]

        return downloaded[:max_items]
