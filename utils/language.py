from __future__ import annotations

"""Small, dependency-free language hints for segment-level dubbing.

Whisper reports one language for an entire decode. Short videos frequently
switch between speech languages, captions, or quoted names, so we annotate
each segment with a conservative script-based hint. An optional ``langdetect``
installation can refine Latin-script text; the deterministic fallback keeps
the pipeline usable in the isolated ASR virtualenvs.
"""

import re
from collections import Counter


_VIETNAMESE_MARKERS = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")


def detect_language(text: str, *, fallback: str | None = None) -> tuple[str, float]:
    """Return an ISO-ish language code and a confidence in ``[0, 1]``.

    The detector intentionally prefers script certainty over guessing from a
    short Latin fragment. This prevents a proper name or an English product
    label from changing the translation language of an entire timeline.
    """

    clean = (text or "").strip()
    if not clean:
        return (fallback or "und", 0.0)

    counts = Counter()
    counts["zh"] = len(_HAN_RE.findall(clean))
    counts["ja"] = len(_HIRAGANA_KATAKANA_RE.findall(clean))
    counts["ko"] = len(_HANGUL_RE.findall(clean))
    counts["vi"] = sum(char in _VIETNAMESE_MARKERS for char in clean)
    latin = len(_LATIN_RE.findall(clean))
    counts["latin"] = latin

    if counts["ja"]:
        total = max(1, counts["ja"] + counts["zh"])
        return "ja", min(0.99, 0.72 + counts["ja"] / total * 0.27)
    if counts["ko"]:
        return "ko", min(0.99, 0.75 + counts["ko"] / max(1, len(clean)) * 0.24)
    if counts["zh"]:
        total = max(1, counts["zh"] + counts["vi"])
        return "zh", min(0.98, 0.72 + counts["zh"] / total * 0.26)
    if counts["vi"] >= 1:
        return "vi", min(0.98, 0.62 + counts["vi"] / max(1, latin) * 0.36)
    if latin:
        # Optional package support is best-effort and never required at boot.
        try:
            from langdetect import detect_langs  # type: ignore

            candidate = detect_langs(clean)[0]
            code = str(candidate.lang).lower()
            probability = float(candidate.prob)
            if code in {"en", "fr", "de", "es", "it", "pt", "ru", "th", "id"}:
                return code, max(0.5, min(0.99, probability))
        except Exception:
            pass
        return (fallback or "en"), 0.52

    return (fallback or "und"), 0.25


def annotate_text_language(text: str, *, fallback: str | None = None) -> tuple[str, float]:
    return detect_language(text, fallback=fallback)
