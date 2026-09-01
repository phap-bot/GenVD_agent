from __future__ import annotations

import os
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


def _config_value(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        for line in env_path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean.startswith(f"{name}="):
                return clean.split("=", 1)[1].strip().strip('"').strip("'") or default
    except OSError:
        pass
    return default


def _vieneu_batch_size() -> int:
    try:
        value = int(_config_value("AUTODUB_VIENEU_BATCH_SIZE", "16"))
    except ValueError:
        value = 16
    return max(1, min(32, value))


def _vieneu_max_chars() -> int:
    try:
        value = int(_config_value("AUTODUB_VIENEU_MAX_CHARS", "384"))
    except ValueError:
        value = 256
    return max(64, min(512, value))


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
        max_chars=_vieneu_max_chars(),
        emotion="natural",
        temperature=0.35,
        top_k=15,
        top_p=0.85,
        repetition_penalty=1.2,
        silence_p=0.03,
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
        top_k=15,
        top_p=0.85,
        repetition_penalty=1.2,
        silence_p=0.03,
        crossfade_p=0.0,
        apply_watermark=False,
        voice=reference.voice_payload,
        use_ref_codes=True,
        max_chars=_vieneu_max_chars(),
    )


def infer_stable_vieneu_audio_batch(model: Any, texts: list[str], voice: str) -> list[Any]:
    """Batch system-voice synthesis when the installed VieNeu supports it."""
    clean_texts = [text.strip() or " " for text in texts]
    if not clean_texts:
        return []
    infer_batch = getattr(model, "infer_batch", None)
    if not callable(infer_batch) or len(clean_texts) == 1:
        return [infer_stable_vieneu_audio(model, text, voice) for text in clean_texts]
    return infer_batch(
        texts=clean_texts,
        voice=voice,
        style="tu_nhien",
        temperature=0.35,
        top_k=15,
        top_p=0.85,
        repetition_penalty=1.2,
        max_chars=_vieneu_max_chars(),
        batch_size=_vieneu_batch_size(),
        apply_watermark=False,
    )


def infer_stable_cloned_vieneu_audio_batch(
    model: Any,
    texts: list[str],
    reference: ClonedVieneuVoice,
) -> list[Any]:
    """Batch cloned-voice synthesis with a sequential compatibility fallback."""
    if not isinstance(reference, ClonedVieneuVoice):
        raise TypeError("A validated cloned voice reference is required.")
    clean_texts = [text.strip() or " " for text in texts]
    if not clean_texts:
        return []
    infer_batch = getattr(model, "infer_batch", None)
    if not callable(infer_batch) or len(clean_texts) == 1:
        return [infer_stable_cloned_vieneu_audio(model, text, reference) for text in clean_texts]
    return infer_batch(
        texts=clean_texts,
        voice=reference.voice_payload,
        style="tu_nhien",
        use_ref_codes=True,
        temperature=0.35,
        top_k=15,
        top_p=0.85,
        repetition_penalty=1.2,
        max_chars=_vieneu_max_chars(),
        batch_size=_vieneu_batch_size(),
        apply_watermark=False,
    )
