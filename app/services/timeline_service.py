from __future__ import annotations

import re
from typing import Iterable

from app.models.schemas import DubbingScriptSegment, TranscriptSegment

MIN_TIMELINE_DURATION = 0.1
SEMANTIC_MAX_GAP = 0.35
SEMANTIC_SHORT_SUFFIX_MAX_GAP = 1.0
SEMANTIC_MAX_GROUP_DURATION = 5.5
SEMANTIC_MAX_CJK_CHARS = 48
SEMANTIC_MAX_TEXT_CHARS = 120
SEMANTIC_MAX_SEGMENTS = 3
SEMANTIC_SHORT_CJK_SUFFIX_CHARS = 6
SEMANTIC_SHORT_TEXT_SUFFIX_CHARS = 18
TERMINAL_PUNCTUATION = set(".!?。！？；;")


class TimelineService:
    """Build the single canonical timeline used by subtitles and TTS."""

    _whitespace_pattern = re.compile(r"\s+")

    def from_transcript(
        self,
        segments: Iterable[TranscriptSegment],
        *,
        merge_semantic: bool = False,
        source_language: str | None = None,
    ) -> list[TranscriptSegment]:
        normalized = [
            TranscriptSegment(
                id=index,
                start=max(0.0, float(segment.start)),
                end=max(0.0, float(segment.end)),
                text=self.normalize_text(segment.text),
                words=segment.words,
            )
            for index, segment in enumerate(sorted(segments, key=lambda item: (item.start, item.end)))
            if self.normalize_text(segment.text)
        ]
        fixed = self._fix_timing(normalized)
        if merge_semantic:
            fixed = self._merge_semantic_units(fixed, source_language=source_language)
        return self._fix_timing(fixed)

    def from_script(self, segments: Iterable[DubbingScriptSegment]) -> list[TranscriptSegment]:
        normalized = [
            TranscriptSegment(
                id=index,
                start=max(0.0, float(segment.start)),
                end=max(0.0, float(segment.end)),
                text=self.normalize_text(segment.translated_text or segment.original_text),
                words=[],
            )
            for index, segment in enumerate(sorted(segments, key=lambda item: (item.start, item.end)))
            if self.normalize_text(segment.translated_text or segment.original_text)
        ]
        return self._fix_timing(normalized)

    def normalize_text(self, text: str) -> str:
        return self._whitespace_pattern.sub(" ", text).strip()

    def _fix_timing(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        fixed: list[TranscriptSegment] = []
        for index, segment in enumerate(segments):
            start = max(0.0, segment.start)
            end = max(segment.end, start + MIN_TIMELINE_DURATION)
            if index + 1 < len(segments):
                next_start = max(0.0, segments[index + 1].start)
                if end > next_start:
                    end = max(start + MIN_TIMELINE_DURATION, next_start)

            fixed.append(
                TranscriptSegment(
                    id=index,
                    start=start,
                    end=end,
                    text=segment.text,
                    words=segment.words,
                )
            )
        return fixed

    def _merge_semantic_units(
        self,
        segments: list[TranscriptSegment],
        *,
        source_language: str | None,
    ) -> list[TranscriptSegment]:
        merged: list[TranscriptSegment] = []
        current: list[TranscriptSegment] = []

        for segment in segments:
            if not current:
                current = [segment]
                continue
            if self._should_merge_semantic(current, segment, source_language=source_language):
                current.append(segment)
                continue

            merged.append(self._semantic_group_to_segment(len(merged), current))
            current = [segment]

        if current:
            merged.append(self._semantic_group_to_segment(len(merged), current))
        return merged

    def _should_merge_semantic(
        self,
        group: list[TranscriptSegment],
        candidate: TranscriptSegment,
        *,
        source_language: str | None,
    ) -> bool:
        previous = group[-1]
        previous_text = previous.text.strip()
        candidate_text = candidate.text.strip()
        if not previous_text or not candidate_text:
            return False
        if self._ends_sentence(previous_text):
            return False

        combined_text = self._join_text_chunks([*[item.text for item in group], candidate_text])
        contains_cjk = self._contains_cjk(combined_text, source_language)
        short_suffix = self._is_short_semantic_suffix(candidate_text, contains_cjk=contains_cjk)

        gap = max(0.0, candidate.start - previous.end)
        max_gap = SEMANTIC_SHORT_SUFFIX_MAX_GAP if short_suffix else SEMANTIC_MAX_GAP
        if gap > max_gap or len(group) >= SEMANTIC_MAX_SEGMENTS:
            return False

        combined_duration = max(candidate.end, group[0].end) - group[0].start
        if combined_duration > SEMANTIC_MAX_GROUP_DURATION:
            return False

        text_limit = SEMANTIC_MAX_CJK_CHARS if contains_cjk else SEMANTIC_MAX_TEXT_CHARS
        if len(combined_text) > text_limit and not short_suffix:
            return False

        if contains_cjk:
            return True
        return self._looks_like_continuation(candidate_text)

    def _semantic_group_to_segment(self, next_id: int, group: list[TranscriptSegment]) -> TranscriptSegment:
        return TranscriptSegment(
            id=next_id,
            start=group[0].start,
            end=max(item.end for item in group),
            text=self._join_text_chunks([item.text for item in group]),
            words=[word for item in group for word in item.words],
        )

    def _join_text_chunks(self, chunks: list[str]) -> str:
        clean_chunks = [self.normalize_text(chunk) for chunk in chunks if self.normalize_text(chunk)]
        if not clean_chunks:
            return ""
        joined = "".join(clean_chunks)
        if self._contains_cjk(joined, None):
            return joined
        return " ".join(clean_chunks)

    def _contains_cjk(self, text: str, source_language: str | None) -> bool:
        language = (source_language or "").lower()
        return language.startswith(("zh", "ja", "ko")) or bool(
            re.search(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]", text)
        )

    def _ends_sentence(self, text: str) -> bool:
        clean_text = text.rstrip()
        return bool(clean_text and clean_text[-1] in TERMINAL_PUNCTUATION)

    def _is_short_semantic_suffix(self, text: str, *, contains_cjk: bool) -> bool:
        clean_text = text.strip()
        if not clean_text:
            return False
        if contains_cjk:
            return len(clean_text) <= SEMANTIC_SHORT_CJK_SUFFIX_CHARS
        return len(clean_text) <= SEMANTIC_SHORT_TEXT_SUFFIX_CHARS and self._looks_like_continuation(clean_text)

    def _looks_like_continuation(self, text: str) -> bool:
        clean_text = text.lstrip()
        if not clean_text:
            return False
        return clean_text[0].islower() or clean_text[0] in ",;:)]}" or clean_text.startswith(("and ", "or ", "but ", "so "))
