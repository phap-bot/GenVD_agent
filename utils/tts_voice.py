from __future__ import annotations

from typing import Any

DEFAULT_VIENEUV3_VOICE = "Trúc Ly"

VOICE_ALIASES = {
    "VieNeu - Trúc Ly": "Trúc Ly",
    "VieNeu - Ngọc Linh": "Ngọc Linh",
    "VieNeu - Ngọc Lan": "Ngọc Linh",
    "VieNeu - Gia Bảo": "Thái Sơn",
    "VieNeu - Xuân Vĩnh": "Xuân Vĩnh",
    "Ngọc Lan": "Ngọc Linh",
    "Mỹ Duyên": "Mai Anh",
    "Gia Bảo": "Thái Sơn",
    "Đức Trí": "Minh Đức",
    "Trọng Hữu": "Phạm Tuyên",
    "Bình An": "Thanh Bình",
    "Nữ Tiktok_nu": "Trúc Ly",
    "Nu Tiktok_nu": "Trúc Ly",
    "Mặc định": DEFAULT_VIENEUV3_VOICE,
    "Default": DEFAULT_VIENEUV3_VOICE,
}


def normalize_vieneu_voice(voice: str | None, fallback: str | None = DEFAULT_VIENEUV3_VOICE) -> str:
    candidate = (voice or fallback or DEFAULT_VIENEUV3_VOICE).strip()
    return VOICE_ALIASES.get(candidate, candidate) or DEFAULT_VIENEUV3_VOICE


def available_vieneu_voices(model: Any) -> set[str]:
    voices = getattr(model, "_preset_voices", None)
    if isinstance(voices, dict):
        return {str(name) for name in voices}
    return set()


def resolve_vieneu_voice(
    model: Any,
    requested_voice: str | None,
    fallback_voice: str | None = DEFAULT_VIENEUV3_VOICE,
) -> str:
    requested = normalize_vieneu_voice(requested_voice, fallback_voice)
    voices = available_vieneu_voices(model)
    if not voices or requested in voices:
        return requested

    fallback = normalize_vieneu_voice(fallback_voice, DEFAULT_VIENEUV3_VOICE)
    if fallback in voices:
        return fallback

    default_voice = normalize_vieneu_voice(getattr(model, "_default_voice", None), DEFAULT_VIENEUV3_VOICE)
    if default_voice in voices:
        return default_voice

    return sorted(voices)[0]


def infer_stable_vieneu_audio(model: Any, text: str, voice: str):
    return model.infer(
        text=text.strip() or " ",
        voice=voice,
        emotion="natural",
        temperature=0.35,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.25,
        silence_p=0.05,
        crossfade_p=0.0,
        apply_watermark=False,
    )
