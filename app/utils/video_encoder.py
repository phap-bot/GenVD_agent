from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger("auto_dubbing.video_encoder")

NVENC_PRESETS = frozenset({"p1", "p2", "p3", "p4", "p5", "p6", "p7"})
QSV_PRESETS = frozenset({"veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"})
HARDWARE_ENCODERS = ("h264_nvenc", "h264_qsv", "h264_amf")


@dataclass(frozen=True)
class VideoEncoderPlan:
    codec: str
    options: dict[str, str | int]
    reason: str


def select_video_encoder(
    *,
    ffmpeg_binary: str,
    has_visual_filters: bool,
    x264_preset: str,
    x264_crf: int,
) -> VideoEncoderPlan:
    """Select stream-copy, NVENC, or CPU encoding from the actual graph."""
    copy_when_possible = os.environ.get("AUTODUB_VIDEO_COPY_WHEN_POSSIBLE", "true").strip().lower()
    if not has_visual_filters and copy_when_possible not in {"0", "false", "no", "off"}:
        return VideoEncoderPlan(codec="copy", options={"vcodec": "copy"}, reason="no_visual_filters")

    requested = os.environ.get("AUTODUB_VIDEO_ENCODER", "auto").strip().lower()
    if requested in {"cpu", "x264"}:
        requested = "libx264"
    elif requested in {"gpu", "nvenc"}:
        requested = "h264_nvenc"
    elif requested in {"qsv", "quicksync", "quick_sync"}:
        requested = "h264_qsv"
    elif requested == "amf":
        requested = "h264_amf"
    if requested not in {"auto", "libx264", *HARDWARE_ENCODERS}:
        logger.warning("video_encoder.invalid requested=%s fallback=auto", requested)
        requested = "auto"

    candidates = HARDWARE_ENCODERS if requested == "auto" else (requested,)
    for codec in candidates:
        if codec == "libx264" or not hardware_encoder_usable(ffmpeg_binary, codec):
            continue
        return hardware_encoder_plan(codec)

    if requested in HARDWARE_ENCODERS:
        logger.warning("video_encoder.hardware_unavailable requested=%s fallback=libx264", requested)
    return cpu_encoder_plan(x264_preset=x264_preset, x264_crf=x264_crf)


def hardware_encoder_plan(codec: str) -> VideoEncoderPlan:
    if codec == "h264_nvenc":
        preset = os.environ.get("AUTODUB_NVENC_PRESET", "p4").strip().lower()
        if preset not in NVENC_PRESETS:
            logger.warning("video_encoder.invalid_nvenc_preset value=%s fallback=p4", preset)
            preset = "p4"
        try:
            cq = int(os.environ.get("AUTODUB_NVENC_CQ", "23"))
        except ValueError:
            cq = 23
        cq = min(51, max(0, cq))
        return VideoEncoderPlan(
            codec="h264_nvenc",
            options={
                "vcodec": "h264_nvenc",
                "preset": preset,
                "tune": "hq",
                "rc": "vbr",
                "cq": cq,
                "b:v": "0",
                "pix_fmt": "yuv420p",
            },
            reason="nvenc_available",
        )
    if codec == "h264_qsv":
        preset = os.environ.get("AUTODUB_QSV_PRESET", "veryfast").strip().lower()
        if preset not in QSV_PRESETS:
            logger.warning("video_encoder.invalid_qsv_preset value=%s fallback=veryfast", preset)
            preset = "veryfast"
        quality = _bounded_env_int("AUTODUB_QSV_QUALITY", 23, minimum=1, maximum=51)
        return VideoEncoderPlan(
            codec="h264_qsv",
            options={
                "vcodec": "h264_qsv",
                "preset": preset,
                "global_quality": quality,
                "pix_fmt": "nv12",
            },
            reason="qsv_available",
        )
    if codec == "h264_amf":
        quality = _bounded_env_int("AUTODUB_AMF_QP", 23, minimum=0, maximum=51)
        return VideoEncoderPlan(
            codec="h264_amf",
            options={
                "vcodec": "h264_amf",
                "quality": "speed",
                "rc": "cqp",
                "qp_i": quality,
                "qp_p": quality,
                "pix_fmt": "yuv420p",
            },
            reason="amf_available",
        )
    raise ValueError(f"Unsupported hardware encoder: {codec}")


def cpu_encoder_plan(*, x264_preset: str, x264_crf: int) -> VideoEncoderPlan:
    return VideoEncoderPlan(
        codec="libx264",
        options={
            "vcodec": "libx264",
            "preset": x264_preset,
            "crf": x264_crf,
            "pix_fmt": "yuv420p",
        },
        reason="cpu_fallback",
    )


@lru_cache(maxsize=16)
def hardware_encoder_usable(ffmpeg_binary: str, codec: str) -> bool:
    """Probe a real one-frame encode; encoder listings alone can be misleading."""
    if codec not in HARDWARE_ENCODERS:
        return False
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:r=1",
        "-frames:v",
        "1",
        "-c:v",
        codec,
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15.0,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("video_encoder.hardware_probe_failed codec=%s error=%s", codec, exc)
        return False
    if completed.returncode == 0:
        logger.info("video_encoder.hardware_available codec=%s ffmpeg=%s", codec, ffmpeg_binary)
        return True
    detail = completed.stderr.decode("utf-8", errors="replace").strip() if completed.stderr else ""
    logger.warning(
        "video_encoder.hardware_probe_rejected codec=%s code=%s detail=%s",
        codec,
        completed.returncode,
        _diagnostic_line(detail) or "<empty>",
    )
    return False


def nvenc_usable(ffmpeg_binary: str) -> bool:
    return hardware_encoder_usable(ffmpeg_binary, "h264_nvenc")


def is_hardware_encoder_runtime_error(exc: BaseException, codec: str) -> bool:
    text = str(exc).casefold()
    codec_markers = {
        "h264_nvenc": ("h264_nvenc", "nvenc @", "nvcuda", "nvencodeapi", "no capable devices"),
        "h264_qsv": ("h264_qsv", "qsv @", "mfx session", "libmfx", "quick sync"),
        "h264_amf": ("h264_amf", "amf @", "amfrt64"),
    }
    return any(
        marker in text
        for marker in (
            *codec_markers.get(codec, ()),
            "unsupported device",
            "initialize encoder session",
            "failed to create hardware device",
        )
    )


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


def _diagnostic_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        lowered = line.casefold()
        if any(
            marker in lowered
            for marker in ("driver", "required", "cannot load", "failed", "unsupported", "no capable")
        ):
            return line
    return lines[-1] if lines else ""
