from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class WordTimestamp(BaseModel):
    word: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class TranscriptSegment(BaseModel):
    id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)

    @field_validator("end")
    @classmethod
    def end_must_be_after_start(cls, value: float, info):
        start = info.data.get("start")
        if start is not None and value < start:
            raise ValueError("segment end must be greater than or equal to start")
        return value


class TranscribeRequest(BaseModel):
    uuid: str


class PipelineConfig(BaseModel):
    source_language: str | None = Field(default=None, max_length=16)
    target_language: str = Field(default="en", min_length=2, max_length=16)
    translation_provider: Literal["9router", "google", "mock"] = "9router"
    translation_model: str = Field(default="ag/gemini-3-flash-agent", min_length=1, max_length=160)
    asr_model: str = Field(default="base", min_length=1, max_length=160)
    compute_type: Literal["int8", "float16"] = "int8"
    word_timestamps: bool = False
    voice_model: str = Field(default="Trúc Ly", min_length=1, max_length=64)
    voice_mode: Literal["system", "clone"] = "system"
    clone_reference_audio_path: str | None = Field(default=None, max_length=1024)
    tts_device: Literal["cuda"] = "cuda"
    background_volume: float = Field(default=0.0, ge=0, le=1)
    tts_volume: float = Field(default=1.0, ge=0, le=2)
    burn_subtitles: bool = True
    mock_translation: bool = False
    mock_tts: bool = False
    copyright_confirmed: bool = False
    copyright_source: Literal["owned", "licensed", "public_domain", "permission", "platform_library", "unknown"] = "unknown"
    copyright_notes: str = Field(default="", max_length=500)
    ocr_fallback: bool = True
    ocr_force: bool = False
    ocr_model: str = Field(default="gemini/gemini-2.5-flash", min_length=1, max_length=160)
    ocr_interval_seconds: float = Field(default=0.75, ge=0.25, le=5.0)
    ocr_crop_bottom_ratio: float = Field(default=0.35, ge=0.12, le=0.85)

    @model_validator(mode="after")
    def require_selected_voice_source(self):
        if self.voice_mode == "clone" and not (self.clone_reference_audio_path or "").strip():
            raise ValueError("Clone voice mode requires clone_reference_audio_path.")
        return self


    def require_copyright_preflight(self) -> None:
        if not self.copyright_confirmed:
            raise ValueError("Copyright preflight required: confirm you have rights to use this media before processing.")
        if self.copyright_source == "unknown":
            raise ValueError("Copyright preflight required: choose a clear rights source before processing.")

class PipelineResult(BaseModel):
    request_id: str
    output_video_path: Path
    subtitle_path: Path | None = None
    segments: list[TranscriptSegment]


class SubtitleStyle(BaseModel):
    x: float = Field(default=12, ge=0, le=100)
    y: float = Field(default=72, ge=0, le=100)
    width: float = Field(default=76, ge=5, le=100)
    height: float = Field(default=16, ge=4, le=60)
    font_size: int = Field(default=42, ge=12, le=120)
    color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    outline_color: str = Field(default="#000000", pattern=r"^#[0-9A-Fa-f]{6}$")
    outline_width: int = Field(default=3, ge=0, le=12)
    align: Literal["left", "center", "right"] = "center"


class BlurStyle(BaseModel):
    enabled: bool = False
    x: float = Field(default=18, ge=0, le=100)
    y: float = Field(default=10, ge=0, le=100)
    width: float = Field(default=44, ge=5, le=100)
    height: float = Field(default=18, ge=4, le=60)
    blur: int = Field(default=16, ge=0, le=48)
    opacity: float = Field(default=0.26, ge=0, le=0.95)


class DubbingScriptSegment(BaseModel):
    id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    original_text: str
    translated_text: str
    subtitle_style: SubtitleStyle = Field(default_factory=SubtitleStyle)
    blur_style: BlurStyle = Field(default_factory=BlurStyle)
    voice_model: str = "Trúc Ly"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class AnalyzeResponse(BaseModel):
    request_id: str
    status: Literal["completed"]
    source_video_path: str
    segments: list[DubbingScriptSegment]


class RenderScriptRequest(BaseModel):
    source_video_path: str = Field(min_length=1)
    target_language: str = Field(default="vi", min_length=2, max_length=16)
    translation_provider: Literal["9router", "google", "mock"] = "9router"
    translation_model: str = Field(default="ag/gemini-3-flash-agent", min_length=1, max_length=160)
    voice_model: str = Field(default="Trúc Ly", min_length=1, max_length=64)
    voice_mode: Literal["system", "clone"] = "system"
    clone_reference_audio_path: str | None = Field(default=None, max_length=1024)
    tts_device: Literal["cuda"] = "cuda"
    background_volume: float = Field(default=0.0, ge=0, le=1)
    tts_volume: float = Field(default=1.0, ge=0, le=2)
    burn_subtitles: bool = True
    mock_tts: bool = False
    copyright_confirmed: bool = False
    copyright_source: Literal["owned", "licensed", "public_domain", "permission", "platform_library", "unknown"] = "unknown"
    copyright_notes: str = Field(default="", max_length=500)
    segments: list[DubbingScriptSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def require_selected_voice_source(self):
        if self.voice_mode == "clone" and not (self.clone_reference_audio_path or "").strip():
            raise ValueError("Clone voice mode requires clone_reference_audio_path.")
        return self


class ShortenTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    target_duration: float = Field(gt=0, le=120)
    target_language: str = Field(default="vi", min_length=2, max_length=16)
    translation_provider: Literal["9router", "google", "mock"] = "9router"
    translation_model: str = Field(default="", max_length=160)
    source_text: str | None = Field(default=None, max_length=4000)
    context: str | None = Field(default=None, max_length=4000)
    max_words: int | None = Field(default=None, ge=1, le=300)


class ShortenTextResponse(BaseModel):
    text: str
    max_words: int
    target_duration: float
    provider: str
    model: str


class BatchPipelineResult(BaseModel):
    request_id: str
    results: list[PipelineResult]
    failed: list[str] = Field(default_factory=list)


class DubbingResponse(BaseModel):
    request_id: str
    status: Literal["completed"]
    output_video_path: str
    subtitle_path: str | None = None
    segments_count: int


class BatchDubbingResponse(BaseModel):
    request_id: str
    status: Literal["completed", "partial"]
    completed_count: int
    failed_count: int
    output_video_paths: list[str]
    failures: list[str] = Field(default_factory=list)


class DouyinRequest(BaseModel):
    url: str = Field(min_length=1)
    mode: Literal["single", "user"] = "single"
    max_items: int = Field(default=10, ge=1, le=100)
    config: PipelineConfig = Field(default_factory=PipelineConfig)
