from __future__ import annotations

"""Dedicated processing policy for the Short Video workspace.

The heavy model/render implementation remains shared with the proven pipeline,
but this profile intentionally changes the two places that made short clips
feel like long-video jobs: semantic stitching and adjacent TTS batching.
"""

from app.models.schemas import DubbingScriptSegment, PipelineConfig, TranscriptSegment
from app.services.pipeline import AutoDubbingPipeline, _TTSGroup
from app.services.timeline_service import TimelineService


SHORT_PIPELINE_VERSION = "short-v1-per-segment"


class ShortVideoPipeline(AutoDubbingPipeline):
    """Auto-dubbing pipeline with a precise, non-merged short-video policy."""

    def __init__(self, config: PipelineConfig, cancel_event=None) -> None:
        if not config.short_video:
            config = config.model_copy(update={"short_video": True})
        super().__init__(config, cancel_event=cancel_event)

    def _checkpoint_config(self, stage: str) -> dict[str, object]:
        payload = super()._checkpoint_config(stage)
        payload["pipeline_profile"] = SHORT_PIPELINE_VERSION
        return payload

    def _source_timeline(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        # ASR/OCR timestamps are already the source of truth for Shorts.  Do
        # not semantically stitch adjacent cues: the editor needs each cue to
        # remain independently reviewable and renderable.
        return TimelineService().from_transcript(
            segments,
            merge_semantic=False,
            source_language=self.config.source_language,
        )

    def _segment_duration_limit(self, text: str) -> float:
        # Keep each spoken unit close to a second or two.  Word timestamps are
        # still preferred; these limits are the fallback for engines that only
        # return a paragraph without word-level timing.
        return 2.2 if self._contains_cjk(text) else 3.0

    def _segment_text_limit(self, text: str) -> int:
        return 32 if self._contains_cjk(text) else 88

    def _build_tts_groups(
        self,
        timeline_segments: list[TranscriptSegment],
        voice_segments: list[DubbingScriptSegment],
    ) -> list[_TTSGroup]:
        # One TTS request per timeline segment preserves exact start/end
        # timing and prevents long-video batching from hiding short cues.
        ordered_voice = [
            item
            for item in sorted(voice_segments, key=lambda item: (item.start, item.end, item.id))
            if (item.translated_text or item.original_text).strip()
        ]
        groups: list[_TTSGroup] = []
        for index, segment in enumerate(timeline_segments):
            voice_model = ""
            if self.config.voice_mode == "system" and index < len(ordered_voice):
                voice_model = ordered_voice[index].voice_model.strip()
            groups.append(
                _TTSGroup(
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    first_segment_id=segment.id,
                    segment_count=1,
                    voice_model=voice_model,
                )
            )
        return groups
