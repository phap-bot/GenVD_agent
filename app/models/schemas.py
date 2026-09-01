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
    language: str | None = Field(default=None, max_length=16)
    language_probability: float | None = Field(default=None, ge=0, le=1)

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
    asr_engine: Literal["auto", "whisper", "paraformer"] = "auto"
    whisper_model: str = Field(default="auto", min_length=1, max_length=32)
    whisper_beam_size: int = Field(default=1, ge=1, le=10)
    whisper_batch_size: int = Field(default=8, ge=1, le=32)
    whisper_vad_filter: bool = True
    segment_language_detection: bool = True
    default_source_language: str | None = Field(default="zh-CN", max_length=16)
    fill_speech_gaps: bool = True
    speech_gap_max_s: float = Field(default=8.0, ge=0, le=30)

    compute_type: Literal["int8", "float16"] = "float16"
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
    source_mode: Literal["auto", "voice", "subtitle", "hybrid"] = "auto"
    ocr_max_frames: int = Field(default=80, ge=1, le=2000)
    ocr_adaptive: bool = True
    ocr_scene_threshold: float = Field(default=0.28, ge=0.02, le=1.0)
    vocal_separation: bool = False

    # Shared translation, timing and audio quality controls. These fields are
    # persisted with queued jobs so resumed renders use the same policy.
    translate_batch_size: int = Field(default=24, ge=1, le=100)
    translate_analysis: bool = True
    translate_review: bool = True
    translate_cps_budget: float = Field(default=12.5, gt=1, le=80)
    video_speed: float = Field(default=1.0, gt=0.25, le=2.0)
    voice_speed: float = Field(default=1.0, gt=0.5, le=2.0)
    soft_timing_fit: bool = True
    timing_max_drift_s: float = Field(default=1.5, ge=0, le=10)
    timing_min_gap_s: float = Field(default=0.12, ge=0, le=2)
    timing_max_atempo: float = Field(default=1.1, ge=0.5, le=2.0)
    hq_background: bool = True
    voice_postprocess: bool = True
    voice_target_lufs: float = Field(default=-16.0, ge=-40, le=-1)
    bg_duck_voice_db: float = Field(default=-7.0, ge=-30, le=0)
    checkpoint_enabled: bool = True
    checkpoint_root: str = Field(default="temp/checkpoints", min_length=1, max_length=512)
    # Additive profile flag.  The regular Clone Video workflow keeps the
    # existing semantic stitching/TTS batching policy; Short Video opts into
    # the dedicated per-segment policy without duplicating the whole pipeline.
    short_video: bool = False

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
    source_language: str | None = Field(default=None, max_length=16)
    language_probability: float | None = Field(default=None, ge=0, le=1)

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
    source_language: str | None = Field(default=None, max_length=16)
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
    vocal_separation: bool = False
    ocr_fallback: bool = True
    ocr_force: bool = False
    ocr_model: str = Field(default="gemini/gemini-2.5-flash", min_length=1, max_length=160)
    ocr_interval_seconds: float = Field(default=0.75, ge=0.25, le=5.0)
    ocr_crop_bottom_ratio: float = Field(default=0.35, ge=0.12, le=0.85)
    asr_engine: Literal["auto", "whisper", "paraformer"] = "auto"
    whisper_model: str = "auto"
    whisper_beam_size: int = Field(default=1, ge=1, le=10)
    segment_language_detection: bool = True
    translate_batch_size: int = Field(default=24, ge=1, le=100)
    translate_analysis: bool = True
    translate_review: bool = True
    translate_cps_budget: float = Field(default=12.5, gt=1, le=80)
    video_speed: float = Field(default=1.0, gt=0.25, le=2.0)
    voice_speed: float = Field(default=1.0, gt=0.5, le=2.0)
    soft_timing_fit: bool = True
    timing_max_drift_s: float = Field(default=1.5, ge=0, le=10)
    timing_min_gap_s: float = Field(default=0.12, ge=0, le=2)
    timing_max_atempo: float = Field(default=1.1, ge=0.5, le=2.0)
    hq_background: bool = True
    voice_postprocess: bool = True
    voice_target_lufs: float = Field(default=-16.0, ge=-40, le=-1)
    bg_duck_voice_db: float = Field(default=-7.0, ge=-30, le=0)
    segments: list[DubbingScriptSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def require_selected_voice_source(self):
        self.voice_model = self.voice_model.strip()
        self.segments = [
            segment.model_copy(
                update={"voice_model": segment.voice_model.strip() or self.voice_model}
            )
            for segment in self.segments
        ]
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


class ShortVideoProfile(BaseModel):
    """Backend-resolved processing profile for the additive Short Video workflow."""

    name: Literal["micro", "short", "short_extended", "long"]
    route: Literal["short_video", "clone_video"]
    duration_seconds: float = Field(ge=0)
    max_short_seconds: float = Field(default=300, gt=0)
    asr_model: str
    source_mode: Literal["auto", "voice", "subtitle", "hybrid"] = "auto"
    vocal_separation_default: bool = False


class ShortVideoInspectResponse(BaseModel):
    media_id: str
    filename: str
    input_url: str
    duration_seconds: float = Field(ge=0)
    has_audio: bool
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    profile: ShortVideoProfile


class ShortVideoProfilesResponse(BaseModel):
    profiles: list[ShortVideoProfile]
    short_video_max_seconds: float = Field(gt=0)


class ShortVideoRenderRequest(BaseModel):
    media_id: str = Field(min_length=1, max_length=64)
    source_language: str | None = Field(default=None, max_length=16)
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
    vocal_separation: bool = False
    ocr_fallback: bool = False
    ocr_force: bool = False
    ocr_model: str = Field(default="gemini/gemini-2.5-flash", min_length=1, max_length=160)
    ocr_interval_seconds: float = Field(default=0.5, ge=0.25, le=5.0)
    ocr_crop_bottom_ratio: float = Field(default=0.35, ge=0.12, le=0.85)
    asr_engine: Literal["auto", "whisper", "paraformer"] = "auto"
    whisper_model: str = "auto"
    whisper_beam_size: int = Field(default=1, ge=1, le=10)
    segment_language_detection: bool = True
    translate_batch_size: int = Field(default=24, ge=1, le=100)
    translate_analysis: bool = True
    translate_review: bool = True
    translate_cps_budget: float = Field(default=12.5, gt=1, le=80)
    video_speed: float = Field(default=1.0, gt=0.25, le=2.0)
    voice_speed: float = Field(default=1.0, gt=0.5, le=2.0)
    soft_timing_fit: bool = True
    timing_max_drift_s: float = Field(default=1.5, ge=0, le=10)
    timing_min_gap_s: float = Field(default=0.12, ge=0, le=2)
    timing_max_atempo: float = Field(default=1.1, ge=0.5, le=2.0)
    hq_background: bool = True
    voice_postprocess: bool = True
    voice_target_lufs: float = Field(default=-16.0, ge=-40, le=-1)
    bg_duck_voice_db: float = Field(default=-7.0, ge=-30, le=0)
    segments: list[DubbingScriptSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def require_selected_voice_source(self):
        if self.voice_mode == "clone" and not (self.clone_reference_audio_path or "").strip():
            raise ValueError("Clone voice mode requires clone_reference_audio_path.")
        return self
