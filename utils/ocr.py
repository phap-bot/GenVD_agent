from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from threading import Event
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from utils.translation import (
    _api_url,
    _nine_router_api_key,
    _nine_router_base_url,
    _openai_compatible_headers,
    _parse_chat_completion_body,
)

logger = logging.getLogger("auto_dubbing.ocr")

DEFAULT_OCR_MODEL = "gemini/gemini-2.5-flash"


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("OCR cancelled")


@dataclass(frozen=True)
class OcrFrame:
    index: int
    timestamp: float
    image_base64: str


@dataclass(frozen=True)
class OcrTextSegment:
    id: int
    start: float
    end: float
    text: str


def extract_video_ocr_segments(
    video_path: Path,
    *,
    source_language: str | None = None,
    model: str | None = None,
    interval_seconds: float = 0.75,
    crop_bottom_ratio: float = 0.35,
    max_frames: int = 80,
    batch_size: int = 6,
    timeout: float = 45.0,
    cancel_event: Event | None = None,
) -> list[OcrTextSegment]:
    selected_model = (model or os.environ.get("AUTODUB_OCR_MODEL") or DEFAULT_OCR_MODEL).strip()
    interval = min(5.0, max(0.25, interval_seconds))
    crop_ratio = min(0.85, max(0.12, crop_bottom_ratio))
    frames, duration = _capture_subtitle_frames(video_path, interval, crop_ratio, max_frames)
    if not frames:
        logger.warning("ocr.frames.empty video=%s", video_path)
        return []

    logger.info(
        "ocr.start video=%s frames=%s duration=%.3f interval=%.2f crop_bottom_ratio=%.2f model=%s",
        video_path,
        len(frames),
        duration,
        interval,
        crop_ratio,
        selected_model,
    )

    frame_text: dict[int, str] = {}
    for start in range(0, len(frames), batch_size):
        _raise_if_cancelled(cancel_event)
        batch = frames[start : start + batch_size]
        try:
            frame_text.update(
                _ocr_frames_with_9router(
                    batch,
                    source_language=source_language,
                    model=selected_model,
                    timeout=timeout,
                    cancel_event=cancel_event,
                )
            )
        except Exception:
            logger.exception("ocr.batch.failed video=%s batch_start=%s batch_size=%s", video_path, start, len(batch))

    segments = _merge_ocr_frames(frames, frame_text, interval, duration)
    logger.info("ocr.done video=%s frames=%s segments=%s", video_path, len(frames), len(segments))
    return segments


def _capture_subtitle_frames(
    video_path: Path,
    interval_seconds: float,
    crop_bottom_ratio: float,
    max_frames: int,
) -> tuple[list[OcrFrame], float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for OCR fallback. Install opencv-python-headless.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for OCR: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if duration <= 0:
        duration = _probe_duration_with_cv2(capture, interval_seconds)

    timestamps = _sample_timestamps(duration, interval_seconds, max_frames)
    frames: list[OcrFrame] = []
    try:
        for index, timestamp in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            height, width = frame.shape[:2]
            y0 = max(0, min(height - 1, int(height * (1.0 - crop_bottom_ratio))))
            cropped = frame[y0:height, 0:width]
            cropped = cv2.resize(cropped, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
            ok, encoded = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue

            frames.append(
                OcrFrame(
                    index=index,
                    timestamp=timestamp,
                    image_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
                )
            )
    finally:
        capture.release()

    return frames, duration


def _probe_duration_with_cv2(capture, interval_seconds: float) -> float:
    timestamp = 0.0
    last_seen = 0.0
    while True:
        capture.set(0, timestamp * 1000)
        ok, _frame = capture.read()
        if not ok:
            break
        last_seen = timestamp
        timestamp += max(0.5, interval_seconds)
        if timestamp > 60 * 60:
            break
    return last_seen


def _sample_timestamps(duration: float, interval_seconds: float, max_frames: int) -> list[float]:
    if duration <= 0:
        return [0.0]
    count = max(1, int(duration / interval_seconds) + 1)
    if count > max_frames:
        interval_seconds = duration / max_frames
        count = max_frames
    return [min(duration, round(index * interval_seconds, 3)) for index in range(count)]


def _ocr_frames_with_9router(
    frames: list[OcrFrame],
    *,
    source_language: str | None,
    model: str,
    timeout: float,
    cancel_event: Event | None = None,
) -> dict[int, str]:
    if not frames:
        return {}

    _raise_if_cancelled(cancel_event)

    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Read ONLY subtitle/dialogue text visibly overlaid in the lower part of each video frame. "
                "Ignore logos, UI, watermarks, timestamps, usernames, product labels, and background signs. "
                "Do not translate. Do not invent text when no subtitle is visible. "
                "Return strict JSON only in this exact shape: "
                '{"frames":[{"index":0,"text":"..."},{"index":1,"text":""}]}. '
                f"Expected source language: {source_language or 'auto-detect'}."
            ),
        }
    ]
    for frame in frames:
        content.append({"type": "text", "text": f"Frame index={frame.index}, timestamp={frame.timestamp:.3f}s"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.image_base64}"},
            }
        )

    payload = {
        "model": model,
        "temperature": 0.0,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise OCR engine for video subtitles. Return JSON only.",
            },
            {"role": "user", "content": content},
        ],
    }

    request = Request(
        _api_url(_nine_router_base_url(), "chat/completions"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_openai_compatible_headers(_nine_router_api_key() or ""),
        method="POST",
    )
    try:
        _raise_if_cancelled(cancel_event)
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"9Router OCR request failed with HTTP {exc.code}: {detail}") from exc

    text = _parse_chat_completion_body(body, content_type).strip()
    parsed = _parse_ocr_json(text)
    logger.info("ocr.batch.done model=%s frames=%s parsed=%s", model, len(frames), len(parsed))
    return parsed


def _parse_ocr_json(text: str) -> dict[int, str]:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"```$", "", clean).strip()

    payload: object
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", clean, flags=re.DOTALL)
        if not match:
            logger.warning("ocr.parse.no_json text=%s", clean[:300])
            return {}
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("ocr.parse.invalid_json text=%s", clean[:300])
            return {}

    frames = payload.get("frames", []) if isinstance(payload, dict) else payload
    if not isinstance(frames, list):
        return {}

    parsed: dict[int, str] = {}
    for item in frames:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        parsed[index] = _clean_ocr_text(str(item.get("text") or ""))
    return parsed


def _merge_ocr_frames(
    frames: list[OcrFrame],
    frame_text: dict[int, str],
    interval_seconds: float,
    duration: float,
) -> list[OcrTextSegment]:
    segments: list[OcrTextSegment] = []
    active_text = ""
    active_start: float | None = None
    active_end = 0.0

    for frame in frames:
        text = _clean_ocr_text(frame_text.get(frame.index, ""))
        timestamp = frame.timestamp
        if not text:
            if active_text and active_start is not None:
                segments.append(
                    OcrTextSegment(
                        id=len(segments),
                        start=active_start,
                        end=max(active_start + 0.25, active_end),
                        text=active_text,
                    )
                )
            active_text = ""
            active_start = None
            active_end = 0.0
            continue

        if active_text and _texts_match(active_text, text):
            active_end = min(duration or timestamp + interval_seconds, timestamp + interval_seconds)
            if len(text) > len(active_text):
                active_text = text
            continue

        if active_text and active_start is not None:
            segments.append(
                OcrTextSegment(
                    id=len(segments),
                    start=active_start,
                    end=max(active_start + 0.25, active_end),
                    text=active_text,
                )
            )

        active_text = text
        active_start = timestamp
        active_end = min(duration or timestamp + interval_seconds, timestamp + interval_seconds)

    if active_text and active_start is not None:
        segments.append(
            OcrTextSegment(
                id=len(segments),
                start=active_start,
                end=max(active_start + 0.25, active_end),
                text=active_text,
            )
        )

    return [segment for segment in segments if segment.text and segment.end > segment.start]


def _texts_match(left: str, right: str) -> bool:
    left_key = _ocr_compare_key(left)
    right_key = _ocr_compare_key(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key or SequenceMatcher(None, left_key, right_key).ratio() >= 0.92


def _ocr_compare_key(text: str) -> str:
    compact = re.sub(r"\s+", "", text.strip().lower())
    return re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", "", compact)


def _clean_ocr_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    clean = clean.strip("`'\"“”‘’[]{}")
    if clean.lower() in {"none", "null", "no text", "no subtitle", "empty", "n/a"}:
        return ""
    return clean



