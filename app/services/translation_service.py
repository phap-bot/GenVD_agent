from __future__ import annotations

from app.models.schemas import PipelineConfig, TranscriptSegment


class TranslationService:
    """Timestamp-preserving translation service.

    The current implementation is intentionally mock-friendly. Replace
    `_translate_text` with a real API client when credentials are available.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def translate(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        translated: list[TranscriptSegment] = []
        for segment in segments:
            translated.append(
                segment.model_copy(
                    update={"text": self._translate_text(segment.text)}
                )
            )
        return translated

    def _translate_text(self, text: str) -> str:
        if self.config.mock_translation:
            return f"[{self.config.target_language}] {text}"

        # Hook for a real provider call. Keep the method synchronous because the
        # pipeline contract is synchronous end to end.
        raise NotImplementedError("Configure a real translation provider here.")

