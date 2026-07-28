from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any


def probe_duration(path: Path, *, ffmpeg_module: Any | None = None, logger: logging.Logger | None = None) -> float:
    probe = _probe_with_ffmpeg(path, ffmpeg_module=ffmpeg_module, logger=logger, purpose="duration")
    if probe is not None:
        return float(probe.get("format", {}).get("duration", 0.0) or 0.0)

    if path.suffix.lower() == ".wav":
        return _probe_wav_duration(path, logger=logger)
    return _probe_video_duration_with_cv2(path, logger=logger)


def probe_video_dimensions(
    path: Path,
    *,
    ffmpeg_module: Any | None = None,
    logger: logging.Logger | None = None,
    default: tuple[int, int] = (1920, 1080),
) -> tuple[int, int]:
    probe = _probe_with_ffmpeg(path, ffmpeg_module=ffmpeg_module, logger=logger, purpose="dimensions")
    if probe is not None:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            width = int(stream.get("width") or default[0])
            height = int(stream.get("height") or default[1])
            return width, height

    return _probe_video_dimensions_with_cv2(path, logger=logger, default=default)


def _probe_with_ffmpeg(
    path: Path,
    *,
    ffmpeg_module: Any | None,
    logger: logging.Logger | None,
    purpose: str,
) -> dict[str, Any] | None:
    if ffmpeg_module is None:
        return None

    try:
        return ffmpeg_module.probe(str(path))
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "media_probe.ffprobe_failed path=%s purpose=%s reason=%s",
                path,
                purpose,
                _format_probe_error(exc),
            )
        return None


def _probe_wav_duration(path: Path, *, logger: logging.Logger | None) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        return frame_count / frame_rate if frame_rate > 0 else 0.0
    except Exception as exc:
        if logger is not None:
            logger.warning("media_probe.wav_duration_failed path=%s error=%s", path, exc)
        return 0.0


def _probe_video_duration_with_cv2(path: Path, *, logger: logging.Logger | None) -> float:
    capture = _open_cv2_capture(path, logger=logger)
    if capture is None:
        return 0.0

    try:
        fps = float(capture.get(_cv2().CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(_cv2().CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0 and frame_count > 0:
            return frame_count / fps
        return _scan_duration_with_cv2(capture)
    finally:
        capture.release()


def _probe_video_dimensions_with_cv2(
    path: Path,
    *,
    logger: logging.Logger | None,
    default: tuple[int, int],
) -> tuple[int, int]:
    capture = _open_cv2_capture(path, logger=logger)
    if capture is None:
        return default

    try:
        width = int(capture.get(_cv2().CAP_PROP_FRAME_WIDTH) or default[0])
        height = int(capture.get(_cv2().CAP_PROP_FRAME_HEIGHT) or default[1])
        return width, height
    finally:
        capture.release()


def _open_cv2_capture(path: Path, *, logger: logging.Logger | None):
    try:
        capture = _cv2().VideoCapture(str(path))
    except Exception as exc:
        if logger is not None:
            logger.warning("media_probe.cv2_open_failed path=%s error=%s", path, exc)
        return None

    if capture.isOpened():
        return capture

    capture.release()
    if logger is not None:
        logger.warning("media_probe.cv2_unavailable path=%s", path)
    return None


def _scan_duration_with_cv2(capture) -> float:
    cv2 = _cv2()
    timestamp = 0.0
    last_seen = 0.0
    while True:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, _frame = capture.read()
        if not ok:
            break
        last_seen = timestamp
        timestamp += 0.5
        if timestamp > 60 * 60:
            break
    return last_seen


def _cv2():
    import cv2

    return cv2


def _format_probe_error(exc: Exception) -> str:
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 4551:
        return "blocked by Windows Application Control policy"
    return str(exc) or exc.__class__.__name__
