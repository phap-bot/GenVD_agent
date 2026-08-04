from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("auto_dubbing.genvideo")
router = APIRouter(prefix="/api", tags=["genvideo"])

GENVIDEO_BASE_URL = os.getenv("AUTODUB_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
REMOTE_OPERATIONS: dict[str, str] = {}
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GenVideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    model: str = Field(default="veo-3.1-generate-preview", min_length=1, max_length=120)
    aspect_ratio: str = Field(default="9:16", pattern=r"^(16:9|9:16)$")
    resolution: str = Field(default="1080p", pattern=r"^(720p|1080p|4k)$")


def _api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("AUTODUB_GEMINI_API_KEY", "").strip()


def _json_request(url: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 60.0) -> dict:
    key = _api_key()
    if not key:
        raise HTTPException(status_code=503, detail="Chưa cấu hình GEMINI_API_KEY ở backend.")

    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.warning("genvideo.google_api_failed status=%s detail=%s", exc.code, detail[:1000])
        raise HTTPException(status_code=502, detail=f"Google Veo API trả lỗi {exc.code}: {detail[:500]}") from exc
    except URLError as exc:
        logger.warning("genvideo.google_api_unreachable reason=%s", exc.reason)
        raise HTTPException(status_code=502, detail=f"Không kết nối được Google Veo API: {exc.reason}") from exc


@router.post("/gen-video")
def start_gen_video(request: GenVideoRequest) -> dict[str, str]:
    payload = {
        "instances": [{"prompt": request.prompt}],
        "parameters": {
            "aspectRatio": request.aspect_ratio,
            "resolution": request.resolution,
        },
    }
    result = _json_request(
        f"{GENVIDEO_BASE_URL}/models/{request.model}:predictLongRunning",
        method="POST",
        payload=payload,
    )
    remote_name = str(result.get("name", "")).strip()
    if not remote_name:
        raise HTTPException(status_code=502, detail="Google Veo không trả operation name.")

    operation_id = uuid4().hex
    REMOTE_OPERATIONS[operation_id] = remote_name
    logger.info("genvideo.operation_started id=%s model=%s remote=%s", operation_id, request.model, remote_name)
    return {"operation_id": operation_id, "status": "running"}


@router.get("/gen-video/{operation_id}")
def poll_gen_video(operation_id: str) -> dict[str, object]:
    remote_name = REMOTE_OPERATIONS.get(operation_id)
    if not remote_name:
        raise HTTPException(status_code=404, detail="Không tìm thấy operation Veo.")

    result = _json_request(f"{GENVIDEO_BASE_URL}/{remote_name}")
    if not result.get("done"):
        return {"operation_id": operation_id, "done": False, "progress": 50, "status": "running"}

    error = result.get("error")
    if isinstance(error, dict):
        raise HTTPException(status_code=502, detail=f"Veo generation failed: {error.get('message', error)}")

    samples = result.get("response", {}).get("generateVideoResponse", {}).get("generatedSamples", [])
    video_uri = samples[0].get("video", {}).get("uri") if samples else None
    if not video_uri:
        raise HTTPException(status_code=502, detail="Operation hoàn tất nhưng không có video URI.")

    video_data = _download_video(video_uri)
    output_path = OUTPUT_DIR / f"genvideo_{operation_id}.mp4"
    output_path.write_bytes(video_data)
    REMOTE_OPERATIONS.pop(operation_id, None)
    logger.info("genvideo.operation_completed id=%s output=%s bytes=%s", operation_id, output_path, len(video_data))
    return {"operation_id": operation_id, "done": True, "progress": 100, "status": "completed", "video_url": f"/media/{output_path.name}"}


def _download_video(video_uri: str) -> bytes:
    key = _api_key()
    request = Request(video_uri, headers={"x-goog-api-key": key})
    try:
        with urlopen(request, timeout=120) as response:
            return response.read()
    except (HTTPError, URLError) as exc:
        logger.warning("genvideo.download_failed uri=%s error=%s", video_uri, exc)
        raise HTTPException(status_code=502, detail="Không tải được file video từ Google Veo.") from exc
