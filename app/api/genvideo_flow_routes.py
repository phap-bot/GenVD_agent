from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.genvideo_routes import GenVideoRequest, _api_key, start_gen_video

logger = logging.getLogger("auto_dubbing.genvideo.flow")
router = APIRouter(prefix="/api/gen-video", tags=["genvideo-flow"])

FLOW_DIR = Path("temp") / "genvideo_flows"
FLOW_DIR.mkdir(parents=True, exist_ok=True)

NodeKind = Literal["brief", "script", "image", "veo", "voice", "subtitles", "review", "export"]
NodeStatus = Literal["idle", "ready", "running", "done", "error"]


class FlowNode(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    kind: NodeKind
    title: str = Field(min_length=1, max_length=120)
    eyebrow: str = Field(default="STEP", max_length=32)
    description: str = Field(default="", max_length=500)
    x: float = Field(ge=0, le=4000)
    y: float = Field(ge=0, le=4000)
    status: NodeStatus = "idle"
    settings: dict[str, str] = Field(default_factory=dict)


class FlowGraph(BaseModel):
    flow_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    version: int = Field(default=1, ge=1)
    nodes: list[FlowNode] = Field(min_length=1, max_length=64)


NODE_CATALOG = [
    {"kind": "brief", "title": "Ý tưởng video", "description": "Chủ đề, đối tượng, thông điệp", "color": "amber", "default_settings": {"prompt": "Nhập ý tưởng video", "style": "minimalist stick figure 2D"}},
    {"kind": "script", "title": "Kịch bản AI", "description": "Chia cảnh và lời dẫn", "color": "blue", "default_settings": {"language": "vi-VN", "duration_seconds": "30", "scene_count": "5"}},
    {"kind": "image", "title": "Tạo ảnh cảnh", "description": "Keyframe phong cách người que", "color": "violet", "default_settings": {"model": "gemini-3.1-flash-image", "aspect_ratio": "9:16", "image_size": "1K"}},
    {"kind": "veo", "title": "Veo 3.1", "description": "Gen chuyển động và âm thanh", "color": "cyan", "default_settings": {"model": "veo-3.1-generate-preview", "resolution": "1080p", "aspect_ratio": "9:16"}},
    {"kind": "voice", "title": "Giọng đọc", "description": "TTS theo từng cảnh", "color": "rose", "default_settings": {"voice": "Trúc Ly", "language": "vi-VN", "pace": "1.0"}},
    {"kind": "subtitles", "title": "Phụ đề & blur", "description": "Style, vị trí, che chữ gốc", "color": "emerald", "default_settings": {"position": "bottom-center", "font_size": "42", "blur_original_text": "true"}},
    {"kind": "review", "title": "Kiểm duyệt", "description": "Kiểm tra timeline và quyền dùng", "color": "orange", "default_settings": {"copyright_confirmed": "false", "safety_review": "required"}},
    {"kind": "export", "title": "Xuất bản", "description": "Render video hoàn chỉnh", "color": "slate", "default_settings": {"format": "mp4", "codec": "h264", "quality": "high"}},
]


def _node(node_id: str, kind: NodeKind, title: str, eyebrow: str, description: str, x: int, y: int, settings: dict[str, str]) -> dict:
    return {
        "id": node_id,
        "kind": kind,
        "title": title,
        "eyebrow": eyebrow,
        "description": description,
        "x": x,
        "y": y,
        "status": "ready" if kind == "brief" else "idle",
        "settings": settings,
    }


DEFAULT_NODES = [
    _node("brief-1", "brief", "Ý tưởng video", "INPUT", "Nhập ý tưởng và phong cách tổng thể", 42, 64, {
        "prompt": "Một câu chuyện hoạt hình người que vui nhộn về học cách làm việc nhóm",
        "style": "minimalist stick figure 2D, clean lines, expressive motion",
    }),
    _node("script-1", "script", "Kịch bản AI", "PLAN", "Chia ý tưởng thành các cảnh có nhịp rõ ràng", 316, 64, {
        "language": "vi-VN", "duration_seconds": "30", "scene_count": "5",
    }),
    _node("image-1", "image", "Tạo ảnh cảnh", "VISUAL", "Tạo keyframe đồng nhất nhân vật và bối cảnh", 590, 64, {
        "model": "gemini-3.1-flash-image", "aspect_ratio": "9:16", "image_size": "1K",
    }),
    _node("veo-1", "veo", "Veo 3.1", "GENERATE", "Animate keyframe thành clip có chuyển động và âm thanh", 178, 278, {
        "model": "veo-3.1-generate-preview", "resolution": "1080p", "aspect_ratio": "9:16",
    }),
    _node("voice-1", "voice", "Giọng đọc", "AUDIO", "Tạo lời dẫn và mix audio theo timeline cảnh", 452, 278, {
        "voice": "Trúc Ly", "language": "vi-VN", "pace": "1.0",
    }),
    _node("subtitles-1", "subtitles", "Phụ đề & blur", "POLISH", "Đồng bộ phụ đề, style box và che nội dung gốc", 726, 278, {
        "position": "bottom-center", "font_size": "42", "blur_original_text": "true",
    }),
    _node("review-1", "review", "Kiểm duyệt", "HITL", "Duyệt nội dung trước khi render file cuối", 316, 492, {
        "copyright_confirmed": "false", "safety_review": "required",
    }),
    _node("export-1", "export", "Xuất bản", "OUTPUT", "Render MP4 và đưa vào thư viện video", 590, 492, {
        "format": "mp4", "codec": "h264", "quality": "high",
    }),
]


def _flow_path(flow_id: str) -> Path:
    if not flow_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in flow_id):
        raise HTTPException(status_code=422, detail="flow_id không hợp lệ")
    return FLOW_DIR / f"{flow_id}.json"


def _default_graph(flow_id: str) -> FlowGraph:
    return FlowGraph(flow_id=flow_id, nodes=[FlowNode.model_validate(node) for node in DEFAULT_NODES])


def _load_graph(flow_id: str) -> FlowGraph:
    path = _flow_path(flow_id)
    if not path.exists():
        return _default_graph(flow_id)
    try:
        return FlowGraph.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("genvideo.flow.load_failed flow_id=%s path=%s", flow_id, path)
        raise HTTPException(status_code=500, detail=f"Không đọc được flow đã lưu: {exc}") from exc


def _save_graph(graph: FlowGraph) -> None:
    path = _flow_path(graph.flow_id)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)


@router.get("/bootstrap")
def get_bootstrap(flow_id: str = "main") -> dict[str, object]:
    graph = _load_graph(flow_id)
    return {
        "catalog": NODE_CATALOG,
        "graph": graph.model_dump(),
        "provider": {
            "name": "Google Gemini API",
            "configured": bool(_api_key()),
            "models": ["veo-3.1-generate-preview", "veo-3.1-fast-generate-preview"],
            "image_models": ["gemini-3.1-flash-image"],
            "aspect_ratios": ["9:16", "16:9"],
            "resolutions": ["720p", "1080p", "4k"],
            "key_source": "GEMINI_API_KEY hoặc AUTODUB_GEMINI_API_KEY",
        },
    }


@router.get("/flows/{flow_id}")
def get_flow(flow_id: str) -> FlowGraph:
    return _load_graph(flow_id)


@router.put("/flows/{flow_id}")
def save_flow(flow_id: str, graph: FlowGraph) -> dict[str, object]:
    if graph.flow_id != flow_id:
        raise HTTPException(status_code=422, detail="flow_id trong URL và payload không khớp")
    graph.version += 1
    _save_graph(graph)
    logger.info("genvideo.flow.saved flow_id=%s version=%s nodes=%s", flow_id, graph.version, len(graph.nodes))
    return {"saved": True, "flow_id": flow_id, "version": graph.version, "nodes": len(graph.nodes)}


@router.post("/flows/{flow_id}/run")
def run_flow(flow_id: str) -> dict[str, str]:
    graph = _load_graph(flow_id)
    by_kind = {node.kind: node for node in graph.nodes}
    missing = [kind for kind in ("brief", "script", "veo", "review", "export") if kind not in by_kind]
    if missing:
        raise HTTPException(status_code=422, detail=f"Flow thiếu node bắt buộc: {', '.join(missing)}")

    brief = by_kind["brief"]
    script = by_kind["script"]
    image = by_kind.get("image")
    veo = by_kind["veo"]
    review = by_kind["review"]
    if review.settings.get("copyright_confirmed", "false").lower() != "true":
        raise HTTPException(status_code=422, detail="Cần xác nhận quyền sử dụng tại node Kiểm duyệt trước khi chạy.")

    prompt_parts = [
        brief.settings.get("prompt", "").strip(),
        f"Visual style: {brief.settings.get('style', 'stick figure 2D')}",
        f"Language: {script.settings.get('language', 'vi-VN')}",
        f"Target scenes: {script.settings.get('scene_count', '1')}",
    ]
    if image:
        prompt_parts.append(f"Keep visual consistency suitable for {image.settings.get('aspect_ratio', '9:16')} frames.")
    prompt = "\n".join(part for part in prompt_parts if part)

    request = GenVideoRequest(
        prompt=prompt,
        model=veo.settings.get("model", "veo-3.1-generate-preview"),
        aspect_ratio=veo.settings.get("aspect_ratio", "9:16"),
        resolution=veo.settings.get("resolution", "1080p"),
    )
    logger.info("genvideo.flow.run flow_id=%s model=%s aspect=%s resolution=%s", flow_id, request.model, request.aspect_ratio, request.resolution)
    return start_gen_video(request)
