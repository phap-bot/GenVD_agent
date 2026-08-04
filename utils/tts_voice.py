from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def available_vieneu_voices(model: Any) -> set[str]:
    voices = getattr(model, "_preset_voices", None)
    if isinstance(voices, dict):
        return {str(name) for name in voices}
    return set()


@dataclass(frozen=True)
class ClonedVieneuVoice:
    voice_payload: dict[str, Any]


def resolve_vieneu_voice(
    model: Any,
    requested_voice: str | None,
) -> str:
    requested = (requested_voice or "").strip()
    if requested.startswith("VieNeu - "):
        requested = requested.removeprefix("VieNeu - ").strip()
    if not requested:
        raise ValueError("A system voice ID is required.")
    voices = available_vieneu_voices(model)
    if not voices:
        raise RuntimeError("VieNeu-TTS did not expose any system voices.")
    if requested not in voices:
        raise ValueError(
            f"System voice '{requested}' is unavailable. "
            f"Available voices: {', '.join(sorted(voices))}"
        )
    return requested


def _reference_part_is_empty(value: Any) -> bool:
    if value is None:
        return True
    size = getattr(value, "size", None)
    if size is not None:
        try:
            return int(size) <= 0
        except (TypeError, ValueError):
            pass
    try:
        return len(value) <= 0
    except TypeError:
        return False


def encode_cloned_vieneu_voice(model: Any, reference_audio_path: str | Path) -> ClonedVieneuVoice:
    reference_path = Path(reference_audio_path).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Clone reference audio not found: {reference_path}")
    if reference_path.stat().st_size <= 0:
        raise ValueError(f"Clone reference audio is empty: {reference_path}")

    encoded = model.encode_reference(str(reference_path))
    if encoded is None:
        raise RuntimeError("VieNeu-TTS returned no reference codes for the cloned voice.")

    # VieNeu 3.1 returns (speaker_emb, ref_codes). Newer v3 Turbo builds return
    # ref_codes directly. Keep both paths explicit so infer never receives
    # voice=None and can never select the model's default system voice.
    if isinstance(encoded, tuple) and len(encoded) == 2:
        speaker_emb, ref_codes = encoded
        if _reference_part_is_empty(speaker_emb) or _reference_part_is_empty(ref_codes):
            raise RuntimeError("VieNeu-TTS returned incomplete cloned voice reference data.")
        return ClonedVieneuVoice(
            voice_payload={"speaker_emb": speaker_emb, "codes": ref_codes},
        )

    if _reference_part_is_empty(encoded):
        raise RuntimeError("VieNeu-TTS returned empty reference codes for the cloned voice.")
    return ClonedVieneuVoice(voice_payload={"codes": encoded})


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


def infer_stable_cloned_vieneu_audio(model: Any, text: str, reference: ClonedVieneuVoice):
    if not isinstance(reference, ClonedVieneuVoice):
        raise TypeError("A validated cloned voice reference is required.")

    if not reference.voice_payload:
        raise RuntimeError("Cloned voice reference data is missing.")

    return model.infer(
        text=text.strip() or " ",
        emotion="natural",
        temperature=0.35,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.25,
        silence_p=0.05,
        crossfade_p=0.0,
        apply_watermark=False,
        voice=reference.voice_payload,
        use_ref_codes=True,
    )
