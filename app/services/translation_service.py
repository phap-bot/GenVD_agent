from __future__ import annotations

import logging
from threading import Event

from app.models.schemas import PipelineConfig, TranscriptSegment
from utils.translation import normalize_translation_model, shorten_segments_for_duration, translate_segments

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
        context = self._analysis_context(segments) if self.config.translate_analysis else ""
        if context:
            logger.info("translation.analysis.done terms=%s languages=%s", context.count("term="), self._language_counts(segments))

        # Keep batches homogeneous when the ASR detected a language switch.
        # This is the key difference from a single whole-video language hint.
        translated_texts: list[str] = [""] * len(segments)
        groups = self._language_groups(segments)
        for group in groups:
            source_language = segments[group[0]].language or self.config.source_language
            texts = [segments[index].text for index in group]
            durations = [max(0.1, segments[index].end - segments[index].start) for index in group]
            if self.config.mock_translation:
                values = [f"[{self.config.target_language}] {text}" for text in texts]
            else:
                values = translate_segments(
                    texts,
                    target_durations=durations,
                    source_language=source_language,
                    target_language=self.config.target_language,
                    provider=self.config.translation_provider,
                    model=resolved_model,
                    cancel_event=self.cancel_event,
                    batch_size=self.config.translate_batch_size,
                    context=context,
                    cps_budget=self.config.translate_cps_budget,
                )
            for index, value in zip(group, values):
                translated_texts[index] = value

        if self.config.translate_review:
            translated_texts = self._review_timing(translated_texts, segments, context=context)

        translated: list[TranscriptSegment] = []
        for segment, translated_text in zip(segments, translated_texts):
            translated.append(
                segment.model_copy(
                    update={"text": translated_text}
                )
            )

        # Source segmentation is already sentence-aware. Keep a strict 1:1
        # mapping so original text, translated text, voice and timestamps can
        # never drift to neighboring scenes.
        logger.info("translation.stage.done segments=%s mapping=one_to_one", len(translated))
        return translated

    def _language_groups(self, segments: list[TranscriptSegment]) -> list[list[int]]:
        groups: list[list[int]] = []
        current: list[int] = []
        current_language: str | None = None
        for index, segment in enumerate(segments):
            language = (segment.language or self.config.source_language or "auto").lower()
            if current and language != current_language:
                groups.append(current)
                current = []
            current.append(index)
            current_language = language
        if current:
            groups.append(current)
        return groups

    def _language_counts(self, segments: list[TranscriptSegment]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for segment in segments:
            key = segment.language or self.config.source_language or "auto"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _analysis_context(self, segments: list[TranscriptSegment]) -> str:
        """Build a compact continuity brief without another model dependency."""
        terms: list[str] = []
        for segment in segments:
            for token in segment.text.replace("\n", " ").split():
                clean = token.strip(".,!?;:()[]{}\"'")
                if len(clean) >= 3 and (clean[:1].isupper() or any("\u4e00" <= char <= "\u9fff" for char in clean)):
                    if clean not in terms:
                        terms.append(clean)
                if len(terms) >= 18:
                    break
            if len(terms) >= 18:
                break
        languages = ", ".join(f"{key}:{value}" for key, value in self._language_counts(segments).items())
        return f"languages={languages}; " + "; ".join(f"term={term}" for term in terms)

    def _review_timing(
        self,
        texts: list[str],
        segments: list[TranscriptSegment],
        *,
        context: str = "",
    ) -> list[str]:
        reviewed = [text.strip() for text in texts]
        candidate_indices: list[int] = []
        candidate_limits: list[int] = []
        candidate_durations: list[float] = []
        for index, (text, segment) in enumerate(zip(texts, segments)):
            duration = max(0.1, segment.end - segment.start)
            max_chars = max(1, int(duration * self.config.translate_cps_budget))
            if len(text.strip()) > max_chars:
                candidate_indices.append(index)
                candidate_limits.append(max(1, int(max_chars / 4)))
                candidate_durations.append(duration)
        if not candidate_indices:
            return reviewed

        logger.info(
            "translation.review.start candidates=%s total_segments=%s batch_size=%s",
            len(candidate_indices),
            len(segments),
            self.config.translate_batch_size,
        )
        shortened = shorten_segments_for_duration(
            [texts[index] for index in candidate_indices],
            target_durations=candidate_durations,
            target_language=self.config.target_language,
            source_texts=[segments[index].text for index in candidate_indices],
            context=context,
            provider=self.config.translation_provider,
            model=self.config.translation_model,
            max_words=candidate_limits,
            batch_size=self.config.translate_batch_size,
            cancel_event=self.cancel_event,
        )
        for index, value in zip(candidate_indices, shortened):
            reviewed[index] = value.strip() or reviewed[index]
        logger.info(
            "translation.review.done candidates=%s batches_max=%s",
            len(candidate_indices),
            self.config.translate_batch_size,
        )
        return reviewed
