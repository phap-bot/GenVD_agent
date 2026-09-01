from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import threading
import wave
import math

from app.utils.cancel import PipelineCancelledError
from app.utils.vram import VRAMManager
from utils.model_registry import model_registry

logger = logging.getLogger(__name__)


def _resolve_torch_device(torch, requested_device: str) -> str:
    requested = (requested_device or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
        logger.warning("vocal_separation.invalid_device requested=%s fallback=auto", requested_device)
        requested = "auto"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "vocal_separation.cuda_unavailable requested=%s fallback=cpu",
            requested_device,
        )
        return "cpu"
    return requested if requested.startswith("cuda") else "cpu"


def _load_pcm_wav(torch, audio_path: Path):
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        compression = wav_file.getcomptype()
        payload = wav_file.readframes(frame_count)

    if channels <= 0 or frame_count <= 0 or compression != "NONE" or sample_width != 2:
        raise RuntimeError("The Demucs fallback loader requires a non-empty PCM16 WAV file.")
    samples = torch.frombuffer(bytearray(payload), dtype=torch.int16)
    expected_samples = frame_count * channels
    if samples.numel() != expected_samples:
        raise RuntimeError("PCM WAV sample count does not match its header.")
    waveform = samples.reshape(frame_count, channels).transpose(0, 1).to(torch.float32)
    waveform.div_(32768.0)
    return waveform, sample_rate


def _load_audio(torch, torchaudio, audio_path: Path):
    try:
        return torchaudio.load(str(audio_path))
    except Exception as backend_error:
        if audio_path.suffix.lower() != ".wav":
            raise
        logger.warning(
            "vocal_separation.torchaudio_load_unavailable fallback=wave path=%s error=%s",
            audio_path,
            backend_error,
        )
        try:
            return _load_pcm_wav(torch, audio_path)
        except Exception as fallback_error:
            raise RuntimeError(
                "Unable to decode Demucs input. Install SoundFile or provide a valid PCM16 WAV file."
            ) from fallback_error


def _save_pcm_wav(torch, output_path: Path, waveform, sample_rate: int) -> None:
    pcm = waveform.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
    pcm = pcm.mul(32767.0).round().to(torch.int16).transpose(0, 1).contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(int(waveform.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.numpy().tobytes())


def _save_audio(torch, torchaudio, output_path: Path, waveform, sample_rate: int) -> None:
    try:
        torchaudio.save(str(output_path), waveform, sample_rate)
    except Exception as backend_error:
        logger.warning(
            "vocal_separation.torchaudio_save_unavailable fallback=wave path=%s error=%s",
            output_path,
            backend_error,
        )
        _save_pcm_wav(torch, output_path, waveform, sample_rate)


def _resolve_demucs_chunk_seconds(value: str | float | int | None = None) -> float:
    """Return a bounded chunk size for Demucs inference.

    Demucs' ``split=True`` option still materializes the complete output for
    the input tensor.  Feeding a long movie as one tensor therefore causes a
    very large temporary allocation.  Chunking keeps the model's peak memory
    bounded while retaining Demucs' own overlap handling inside each chunk.
    """
    raw = value if value is not None else os.environ.get("AUTODUB_DEMUCS_CHUNK_SECONDS", "60")
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        logger.warning("vocal_separation.invalid_chunk_seconds value=%s fallback=60", raw)
        return 60.0
    if not math.isfinite(seconds):
        logger.warning("vocal_separation.invalid_chunk_seconds value=%s fallback=60", raw)
        return 60.0
    # Chunks below ten seconds add a large amount of model overhead; very
    # large values recreate the original full-file memory problem.
    return min(600.0, max(10.0, seconds))


@dataclass(frozen=True)
class VocalSeparationResult:
    vocals_path: Path
    accompaniment_path: Path


class VocalSeparationService:
    """Service to separate vocal tracks from background audio using Demucs (htdemucs)."""

    def separate(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        device: str = "cpu",
        cancel_event: threading.Event | None = None,
    ) -> VocalSeparationResult:
        """Separate audio into vocals.wav and accompaniment.wav.

        Runs Demucs inference on specified device (default 'cpu' to preserve VRAM for ASR/TTS).
        """
        if cancel_event and cancel_event.is_set():
            raise PipelineCancelledError("Vocal separation cancelled before execution.")

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vocals_path = output_dir / "vocals.wav"
        accompaniment_path = output_dir / "accompaniment.wav"

        logger.info(
            "vocal_separation.start input=%s device=%s output_dir=%s",
            audio_path,
            device,
            output_dir,
        )

        try:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch is required for vocal separation. Install the project's requirements."
                ) from exc
            try:
                import torchaudio
            except ImportError as exc:
                raise RuntimeError(
                    "torchaudio is required for vocal separation. Install the project's requirements."
                ) from exc
            try:
                from demucs.apply import apply_model
            except ImportError as exc:
                raise RuntimeError(
                    "demucs is required for vocal separation. Install it with `pip install demucs`."
                ) from exc

            requested_device = device or os.environ.get("AUTODUB_DEMUCS_DEVICE", "auto")
            resolved_device = _resolve_torch_device(torch, requested_device)
            with model_registry.acquire_demucs(device=resolved_device) as model:
                if cancel_event and cancel_event.is_set():
                    raise PipelineCancelledError("Vocal separation cancelled during model acquisition.")

                # Demucs quality depends heavily on receiving the original
                # stereo, full-bandwidth mix. Downsampling belongs to ASR,
                # never to source separation.
                wav, sr = _load_audio(torch, torchaudio, audio_path)

                # Resample to model samplerate if necessary (htdemucs defaults to 44100)
                if sr != model.samplerate:
                    resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                    wav = resampler(wav)
                    sr = model.samplerate

                # Convert to stereo if mono (Demucs expects 2 channels)
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                elif wav.shape[0] > 2:
                    wav = wav[:2, :]

                wav = wav.to(resolved_device)
                original_wav = wav

                # Match the normalization used by the official Demucs
                # separator. Calling apply_model on an unnormalized mix leaves
                # substantially more vocal energy in the accompaniment stem.
                # Keep only scalar statistics here; normalizing the complete
                # movie before inference would allocate another full-duration
                # tensor.
                reference_mean = wav.mean()
                reference_std = wav.std().clamp_min(1e-8)

                total_samples = int(wav.shape[-1])
                chunk_seconds = _resolve_demucs_chunk_seconds()
                chunk_samples = max(1, int(round(chunk_seconds * sr)))
                total_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
                logger.info(
                    "vocal_separation.infer audio_shape=%s sr=%s chunk_seconds=%.1f chunks=%s",
                    [1, int(wav.shape[0]), total_samples],
                    sr,
                    chunk_seconds,
                    total_chunks,
                )

                # Process one bounded chunk at a time. Demucs' split=True
                # already applies overlap inside each chunk, and concatenating
                # only the vocal stem avoids retaining all four full-duration
                # source tensors in memory.
                source_names = model.sources
                vocals_idx = source_names.index("vocals") if "vocals" in source_names else -1
                vocals_chunks = []
                for chunk_index, start in enumerate(range(0, total_samples, chunk_samples), start=1):
                    if cancel_event and cancel_event.is_set():
                        raise PipelineCancelledError("Vocal separation cancelled during inference.")
                    end = min(total_samples, start + chunk_samples)
                    normalized_chunk = (
                        (wav[..., start:end] - reference_mean) / reference_std
                    ).unsqueeze(0)
                    with torch.no_grad():
                        # apply_model returns [batch, sources, channels, time].
                        sources = apply_model(
                            model,
                            normalized_chunk,
                            device=resolved_device,
                            shifts=1,
                            split=True,
                            overlap=0.5,
                            progress=False,
                        )
                    sources.mul_(reference_std.to(sources.device)).add_(reference_mean.to(sources.device))
                    vocals_source = sources[0, vocals_idx if vocals_idx >= 0 else -1].detach()
                    vocals_chunk = vocals_source.to(device="cpu", dtype=torch.float32).clone()
                    expected_samples = end - start
                    if vocals_chunk.shape[-1] < expected_samples:
                        vocals_chunk = torch.nn.functional.pad(vocals_chunk, (0, expected_samples - vocals_chunk.shape[-1]))
                    elif vocals_chunk.shape[-1] > expected_samples:
                        vocals_chunk = vocals_chunk[..., :expected_samples]
                    vocals_chunks.append(vocals_chunk)
                    logger.info(
                        "vocal_separation.chunk_done index=%s/%s start=%s end=%s progress=%s",
                        chunk_index,
                        total_chunks,
                        start,
                        end,
                        round(chunk_index / total_chunks * 100),
                    )
                    del vocals_source, vocals_chunk, sources, normalized_chunk
                    if resolved_device.startswith("cuda"):
                        torch.cuda.empty_cache()

                vocals_tensor = torch.cat(vocals_chunks, dim=-1)[..., :total_samples]
                del vocals_chunks, reference_mean, reference_std

                # Preserve the original ambience, stereo image and transients.
                # Reuse the original audio buffer for the residual to avoid two
                # additional full-duration allocations.
                acc_tensor = original_wav.cpu()
                acc_tensor.sub_(vocals_tensor).clamp_(-1.0, 1.0)

                # Save wav files
                _save_audio(torch, torchaudio, vocals_path, vocals_tensor, sr)
                _save_audio(torch, torchaudio, accompaniment_path, acc_tensor, sr)

                logger.info(
                    "vocal_separation.completed vocals=%s accompaniment=%s",
                    vocals_path,
                    accompaniment_path,
                )

                return VocalSeparationResult(
                    vocals_path=vocals_path,
                    accompaniment_path=accompaniment_path,
                )
        finally:
            VRAMManager.cleanup()
