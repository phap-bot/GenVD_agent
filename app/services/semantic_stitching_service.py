from __future__ import annotations

import logging
import re
from typing import Sequence

from app.models.schemas import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)

TERMINAL_PUNCTUATION = set(".!?;\u3002\uff01\uff1f\uff1b")
DANGLING_ENDINGS_CJK = {
    "\u800c", "\u4e14", "\u4e26", "\u5e76", "\u4f46", "\u56e0", "\u70ba", "\u4e3a",
    "\u540c", "\u8207", "\u4e0e", "\u6216", "\u8005", "\u9019", "\u8fd9", "\u90a3", "\u7684", "\u4e16",
}
DANGLING_STARTS_CJK = {
    "\u754c", "\u676f", "\u4e14", "\u800c", "\u540c", "\u4f46", "\u662f", "\u4e26", "\u5e76",
    "\u6a5f", "\u673a", "\u5236",
}
DANGLING_ENDINGS_VI = {
    "m\u00e0", "nh\u01b0ng", "v\u00e0", "ho\u1eb7c", "v\u00ec", "n\u00ean", "n\u1ebfu",
    "b\u1edfi", "do", "l\u00e0", "th\u00ec",
}
DANGLING_STARTS_VI = {
    "c\u00f2n c\u00f3", "m\u00e0", "th\u00ec", "n\u00ean", "ho\u1eb7c", "v\u00e0", "nh\u01b0ng",
}


class SemanticStitchingService:
    """Intelligent sentence boundary restitcher for ASR/OCR timelines.

    Repairs CJK severed words, merges dangling clauses across acoustic pauses,
    and preserves semantic coherence prior to translation and TTS synthesis.
    """

    def stitch_transcript_segments(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        source_language: str | None = None,
        max_gap: float = 1.0,
        max_duration: float = 8.0,
        max_chars: int = 140,
    ) -> list[TranscriptSegment]:
        if not segments:
            return []

        clean_segments = [s for s in segments if s.text and s.text.strip()]
        if not clean_segments:
            return []

        # Step 1: Pre-pass to repair severed CJK words
        repaired = self.repair_cjk_severed_words(clean_segments)

        # Step 2: Merge dangling clauses into complete semantic sentences
        stitched: list[TranscriptSegment] = []
        current_group: list[TranscriptSegment] = []

        for segment in repaired:
            if not current_group:
                current_group = [segment]
                continue

            if self._should_stitch_with_group(
                current_group,
                segment,
                source_language=source_language,
                max_gap=max_gap,
                max_duration=max_duration,
                max_chars=max_chars,
            ):
                current_group.append(segment)
            else:
                stitched.append(self._merge_group(len(stitched), current_group))
                current_group = [segment]

        if current_group:
            stitched.append(self._merge_group(len(stitched), current_group))

        logger.info(
            "semantic_stitching.done input_segments=%d stitched_segments=%d",
            len(segments),
            len(stitched),
        )
        return stitched

    def repair_cjk_severed_words(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Detect and repair multi-character CJK words broken by acoustic silence."""
        if len(segments) < 2:
            return segments

        repaired: list[TranscriptSegment] = []
        skip_next = False

        for index in range(len(segments)):
            if skip_next:
                skip_next = False
                continue

            current = segments[index]
            if index + 1 >= len(segments):
                repaired.append(current)
                break

            next_seg = segments[index + 1]
            gap = max(0.0, next_seg.start - current.end)

            if gap <= 1.2 and self._is_severed_cjk_pair(current.text, next_seg.text):
                # Merge severed CJK word across boundary
                merged_text = f"{current.text.strip()}{next_seg.text.strip()}"
                merged_words: list[WordTimestamp] = [*current.words, *next_seg.words]
                repaired.append(
                    TranscriptSegment(
                        id=current.id,
                        start=current.start,
                        end=next_seg.end,
                        text=merged_text,
                        words=merged_words,
                        language=current.language if current.language == next_seg.language else None,
                        language_probability=current.language_probability,
                    )
                )
                skip_next = True
            else:
                repaired.append(current)

        return repaired

    def smooth_translated_clauses(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Smooth dangling conjunctions at segment boundaries post-translation."""
        if len(segments) < 2:
            return segments

        smoothed: list[TranscriptSegment] = []
        skip_next = False

        for index in range(len(segments)):
            if skip_next:
                skip_next = False
                continue

            current = segments[index]
            if index + 1 >= len(segments):
                smoothed.append(current)
                break

            next_seg = segments[index + 1]
            current_text = current.text.strip()
            next_text = next_seg.text.strip()

            # Check if current ends with a dangling conjunction or next starts with continuation
            if self._is_dangling_vi_boundary(current_text, next_text):
                gap = max(0.0, next_seg.start - current.end)
                combined_duration = next_seg.end - current.start
                if gap <= 0.8 and combined_duration <= 12.0:
                    joined_text = f"{current_text} {next_text}"
                    smoothed.append(
                        TranscriptSegment(
                            id=len(smoothed),
                            start=current.start,
                            end=next_seg.end,
                            text=joined_text,
                            words=[*current.words, *next_seg.words],
                            language=current.language if current.language == next_seg.language else None,
                            language_probability=current.language_probability,
                        )
                    )
                    skip_next = True
                    continue

            smoothed.append(
                current.model_copy(update={"id": len(smoothed)})
            )

        return smoothed

    def _should_stitch_with_group(
        self,
        group: list[TranscriptSegment],
        candidate: TranscriptSegment,
        *,
        source_language: str | None,
        max_gap: float,
        max_duration: float,
        max_chars: int,
    ) -> bool:
        prev = group[-1]
        prev_text = prev.text.strip()
        cand_text = candidate.text.strip()

        if not prev_text or not cand_text:
            return False
        if prev.language and candidate.language and prev.language != candidate.language:
            return False

        # If previous segment clearly ends a sentence with terminal punctuation, do not merge
        if self._ends_sentence(prev_text):
            return False

        gap = max(0.0, candidate.start - prev.end)
        combined_duration = max(candidate.end, group[0].end) - group[0].start
        combined_text = self._join_texts([*[item.text for item in group], cand_text])

        if combined_duration > max_duration or len(combined_text) > max_chars:
            return False

        # If gap is small, or if there is an explicit linguistic continuation signal, merge
        is_cjk = self._contains_cjk(combined_text, source_language)

        if gap <= max_gap:
            return True

        if is_cjk and gap <= 1.2:
            return self._is_cjk_continuation(prev_text, cand_text)

        if not is_cjk and gap <= 1.2:
            return self._is_text_continuation(prev_text, cand_text)

        return False

    def _is_severed_cjk_pair(self, left_text: str, right_text: str) -> bool:
        left_clean = left_text.strip()
        right_clean = right_text.strip()
        if not left_clean or not right_clean:
            return False

        last_char = left_clean[-1]
        first_char = right_clean[0]

        if not self._is_cjk_char(last_char) or not self._is_cjk_char(first_char):
            return False

        # Check known severed word patterns (e.g. 世 + 界 / 界 + 杯 / 競 + 賽 / 機 + 制)
        if last_char in DANGLING_ENDINGS_CJK or first_char in DANGLING_STARTS_CJK:
            return True

        # Common CJK 2-char words broken at boundary
        known_pairs = {
            "\u4e16\u754c", "\u754c\u676f", "\u6bd4\u8cfd", "\u6a5f\u5236",
            "\u89c0\u8cfd", "\u7bc0\u76ee", "\u89aa\u53cb", "\u670b\u53cb",
        }
        pair = f"{last_char}{first_char}"
        return pair in known_pairs

    def _is_cjk_continuation(self, prev_text: str, cand_text: str) -> bool:
        if self._ends_sentence(prev_text):
            return False
        return prev_text[-1] in DANGLING_ENDINGS_CJK or cand_text[0] in DANGLING_STARTS_CJK

    def _is_text_continuation(self, prev_text: str, cand_text: str) -> bool:
        if self._ends_sentence(prev_text):
            return False
        cand_clean = cand_text.lstrip()
        if not cand_clean:
            return False
        if cand_clean[0].islower() or cand_clean[0] in ",;:)]}":
            return True
        return cand_clean.lower().startswith(("and ", "or ", "but ", "so ", "because ", "that ", "which "))

    def _is_dangling_vi_boundary(self, prev_text: str, cand_text: str) -> bool:
        prev_words = prev_text.rstrip().split()
        if not prev_words:
            return False
        last_word = prev_words[-1].lower().strip(",;:.")
        if last_word in DANGLING_ENDINGS_VI:
            return True

        cand_lower = cand_text.lstrip().lower()
        return any(cand_lower.startswith(prefix) for prefix in DANGLING_STARTS_VI)

    def _ends_sentence(self, text: str) -> bool:
        clean = text.rstrip()
        return bool(clean and clean[-1] in TERMINAL_PUNCTUATION)

    def _contains_cjk(self, text: str, source_language: str | None = None) -> bool:
        lang = (source_language or "").lower()
        if lang.startswith(("zh", "ja", "ko")):
            return True
        return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))

    def _is_cjk_char(self, char: str) -> bool:
        return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", char))

    def _merge_group(self, next_id: int, group: list[TranscriptSegment]) -> TranscriptSegment:
        return TranscriptSegment(
            id=next_id,
            start=group[0].start,
            end=max(item.end for item in group),
            text=self._join_texts([item.text for item in group]),
            words=[word for item in group for word in item.words],
            language=group[0].language if all(item.language == group[0].language for item in group) else None,
            language_probability=min(
                (item.language_probability for item in group if item.language_probability is not None),
                default=None,
            ),
        )

    def _join_texts(self, chunks: list[str]) -> str:
        clean = [c.strip() for c in chunks if c and c.strip()]
        if not clean:
            return ""
        joined = "".join(clean)
        if self._contains_cjk(joined):
            return joined
        return " ".join(clean)
