from __future__ import annotations

import logging
from threading import Event

from app.models.schemas import PipelineConfig, TranscriptSegment
from utils.translation import normalize_translation_model, translate_segments

logger = logging.getLogger(__name__)


class TranslationService:
    """Timestamp-preserving batch translation service."""

    def __init__(self, config: PipelineConfig, cancel_event: Event | None = None) -> None:
        self.config = config
        self.cancel_event = cancel_event

    def translate(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        resolved_model = normalize_translation_model(self.config.translation_model)
        logger.info(
            "translation.stage.start segments=%s source=%s target=%s provider=%s model=%s mock=%s",
            len(segments),
            self.config.source_language or "auto",
            self.config.target_language,
            self.config.translation_provider,
            resolved_model,
            self.config.mock_translation,
        )
        if self.config.mock_translation:
            translated_texts = [f"[{self.config.target_language}] {segment.text}" for segment in segments]
        else:
            translated_texts = translate_segments(
                [segment.text for segment in segments],
                source_language=self.config.source_language,
                target_language=self.config.target_language,
                provider=self.config.translation_provider,
                model=resolved_model,
                cancel_event=self.cancel_event,
            )

        translated: list[TranscriptSegment] = []
        for segment, translated_text in zip(segments, translated_texts):
            translated.append(
                segment.model_copy(
                    update={"text": translated_text}
                )
            )
        logger.info("translation.stage.done segments=%s", len(translated))
        return translated
