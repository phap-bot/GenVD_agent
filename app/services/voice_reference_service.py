from __future__ import annotations

import hashlib
import logging
import math
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.services.dependency_service import DependencyService


logger = logging.getLogger(__name__)

VOICE_REFERENCE_SECONDS = 3.0
VOICE_REFERENCE_SAMPLE_RATE = 24_000
VOICE_REFERENCE_FRAMES = int(VOICE_REFERENCE_SECONDS * VOICE_REFERENCE_SAMPLE_RATE)
VOICE_REFERENCE_MIN_SECONDS = 2.98
VOICE_REFERENCE_MAX_SECONDS = 3.05


class VoiceReferenceValidationError(ValueError):
    """The uploaded voice sample is not a valid fixed-duration clone reference."""


@dataclass(frozen=True)
class VoiceReferenceMetadata:
    duration_seconds: float
    sample_rate: int
    frame_count: int
    sha256: str


class VoiceReferenceService:
    """Validate and canonicalize the client-cut clip used by VieNeu.

    The browser uploads only the selected three-second WAV. The backend rejects
    longer inputs instead of silently taking their first three seconds, then
    writes one deterministic mono PCM clip for every downstream render path.
    """

    def __init__(
        self,
        *,
        ffmpeg_binary: str | None = None,
        ffmpeg_timeout_seconds: float = 30.0,
    ) -> None:
        dependencies = DependencyService()
        self.ffmpeg_binary = ffmpeg_binary or dependencies.resolve_binary("ffmpeg")
        self.ffmpeg_timeout_seconds = max(1.0, float(ffmpeg_timeout_seconds))

    def canonicalize(self, source_path: Path, destination_path: Path) -> VoiceReferenceMetadata:
        source = source_path.resolve()
        destination = destination_path.resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise VoiceReferenceValidationError("The clone reference audio is empty.")
        if not self.ffmpeg_binary:
            raise RuntimeError("FFmpeg is required to normalize clone reference audio.")

        source_duration = self._inspect_source_pcm_wav(source)
        if not VOICE_REFERENCE_MIN_SECONDS <= source_duration <= VOICE_REFERENCE_MAX_SECONDS:
            raise VoiceReferenceValidationError(
                "Clone reference must contain only the selected 3-second clip "
                f"(received {source_duration:.3f} seconds)."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate = destination.parent / f".{destination.stem}.{uuid4().hex}.tmp.wav"
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-protocol_whitelist",
            "file",
            "-f",
            "wav",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            (
                f"aresample={VOICE_REFERENCE_SAMPLE_RATE},"
                f"apad=whole_len={VOICE_REFERENCE_FRAMES},"
                f"atrim=end_sample={VOICE_REFERENCE_FRAMES},asetpts=N/SR/TB"
            ),
            "-ac",
            "1",
            "-ar",
            str(VOICE_REFERENCE_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(candidate),
        ]
        try:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.ffmpeg_timeout_seconds,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except subprocess.TimeoutExpired as exc:
                logger.warning("voice_reference.ffmpeg_timeout source=%s", source)
                raise VoiceReferenceValidationError("Timed out while decoding clone reference audio.") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown FFmpeg error").strip()
                logger.warning("voice_reference.ffmpeg_failed source=%s detail=%s", source, detail[:1000])
                raise VoiceReferenceValidationError("Unable to decode clone reference audio.")

            metadata = self.validate_canonical(candidate)
            os.replace(candidate, destination)
            return metadata
        finally:
            candidate.unlink(missing_ok=True)

    def validate_canonical(self, path: Path) -> VoiceReferenceMetadata:
        resolved = path.resolve()
        try:
            with wave.open(str(resolved), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
        except (OSError, EOFError, wave.Error) as exc:
            raise VoiceReferenceValidationError("Clone reference is not a valid PCM WAV file.") from exc

        if channels != 1 or sample_width != 2:
            raise VoiceReferenceValidationError("Clone reference must be mono 16-bit PCM WAV.")
        if sample_rate != VOICE_REFERENCE_SAMPLE_RATE or frame_count != VOICE_REFERENCE_FRAMES:
            raise VoiceReferenceValidationError(
                "Clone reference must be the canonical 3-second, 24 kHz clip."
            )

        return VoiceReferenceMetadata(
            duration_seconds=frame_count / sample_rate,
            sample_rate=sample_rate,
            frame_count=frame_count,
            sha256=self._sha256(resolved),
        )

    @staticmethod
    def _inspect_source_pcm_wav(path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                compression = wav_file.getcomptype()
        except (OSError, EOFError, wave.Error) as exc:
            raise VoiceReferenceValidationError("Clone reference is not a valid PCM WAV file.") from exc
        if channels < 1 or sample_width != 2 or sample_rate <= 0 or frame_count <= 0 or compression != "NONE":
            raise VoiceReferenceValidationError("Clone reference must be a non-empty 16-bit PCM WAV clip.")
        duration = frame_count / sample_rate
        if not math.isfinite(duration) or duration <= 0:
            raise VoiceReferenceValidationError("Unable to determine clone reference duration.")
        return duration

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
