from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import threading

from app.utils.cancel import PipelineCancelledError
from app.utils.vram import VRAMManager
from utils.model_registry import model_registry

logger = logging.getLogger(__name__)


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

            with model_registry.acquire_demucs(device=device) as model:
                if cancel_event and cancel_event.is_set():
                    raise PipelineCancelledError("Vocal separation cancelled during model acquisition.")

                # Demucs quality depends heavily on receiving the original
                # stereo, full-bandwidth mix. Downsampling belongs to ASR,
                # never to source separation.
                wav, sr = torchaudio.load(str(audio_path))

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

                wav = wav.to(device)
                original_wav = wav

                # Match the normalization used by the official Demucs
                # separator. Calling apply_model on an unnormalized mix leaves
                # substantially more vocal energy in the accompaniment stem.
                reference = wav.mean(dim=0)
                reference_mean = reference.mean()
                reference_std = reference.std().clamp_min(1e-8)
                normalized_wav = ((wav - reference_mean) / reference_std).unsqueeze(0)

                logger.info("vocal_separation.infer audio_shape=%s sr=%s", list(normalized_wav.shape), sr)

                with torch.no_grad():
                    # apply_model returns tensor of shape [batch, sources, channels, time]
                    # for htdemucs sources are usually: ['drums', 'bass', 'other', 'vocals']
                    sources = apply_model(
                        model,
                        normalized_wav,
                        device=device,
                        shifts=1,
                        split=True,
                        overlap=0.5,
                        progress=False,
                    )
                sources = sources * reference_std + reference_mean

                # Map source names
                source_names = model.sources

                # Extract vocals tensor [2, time]
                vocals_idx = source_names.index("vocals") if "vocals" in source_names else -1
                if vocals_idx >= 0:
                    vocals_tensor = sources[0, vocals_idx].cpu()
                else:
                    vocals_tensor = sources[0, -1].cpu()

                # Preserve the original ambience, stereo image and transients.
                # The residual is more faithful than re-summing independently
                # estimated stems and removes the model's complete vocal stem.
                acc_tensor = original_wav.cpu() - vocals_tensor
                acc_tensor = acc_tensor.clamp(-1.0, 1.0)

                # Save wav files
                torchaudio.save(str(vocals_path), vocals_tensor, sr)
                torchaudio.save(str(accompaniment_path), acc_tensor, sr)

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
