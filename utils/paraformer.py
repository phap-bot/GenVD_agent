from __future__ import annotations

"""Optional FunASR/Paraformer adapter.

The main environment does not import FunASR. When ``asr_engine=paraformer``
is selected, the adapter loads it lazily so the Whisper/VieNeu environments
remain independent.
"""

import logging
import os
import json
import subprocess
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger("auto_dubbing.paraformer")


def _audio_duration_seconds(audio_path: Path) -> float:
    """Read duration locally without importing the heavy ASR environment."""
    try:
        with wave.open(str(audio_path), "rb") as handle:
            return max(0.0, handle.getnframes() / float(handle.getframerate() or 1))
    except (OSError, wave.Error):
        try:
            completed = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(audio_path)],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return max(0.0, float(completed.stdout.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0.0


def transcribe(audio_path: Path, *, language: str | None = None, model_id: str | None = None) -> list[dict[str, Any]]:
    selected_model = model_id or os.environ.get("AUTODUB_PARAFORMER_MODEL", "paraformer-zh")
    try:
        from funasr import AutoModel  # type: ignore
    except ImportError:
        venv = Path(os.environ.get("AUTODUB_PARAFORMER_VENV", ".venv-asr"))
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        worker = Path(__file__).resolve().parents[1] / "scripts" / "paraformer_worker.py"
        if not python.is_file() or not worker.is_file():
            raise RuntimeError(
                "Paraformer is not installed in the ASR environment. Run scripts/setup_venvs.ps1 -InstallAsr."
            )
        completed = subprocess.run(
            [str(python), str(worker), str(audio_path), "--model", selected_model, "--device", os.environ.get("AUTODUB_PARAFORMER_DEVICE", "cpu")],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        result = json.loads(completed.stdout)
    else:
        model_kwargs: dict[str, Any] = {
            "model": selected_model,
            "device": os.environ.get("AUTODUB_PARAFORMER_DEVICE", "cpu"),
            "disable_update": True,
        }
        if not Path(selected_model).is_dir():
            model_kwargs["vad_model"] = os.environ.get("AUTODUB_PARAFORMER_VAD", "fsmn-vad")
            model_kwargs["punc_model"] = os.environ.get("AUTODUB_PARAFORMER_PUNC", "ct-punc")
        model = AutoModel(**model_kwargs)
        result = model.generate(input=str(audio_path), batch_size_s=300, sentence_timestamp=True)
    if isinstance(result, dict):
        result = [result]

    normalized: list[dict[str, Any]] = []
    audio_duration = _audio_duration_seconds(audio_path)
    for item in result or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        timestamps = item.get("sentence_info") or item.get("timestamp") or []
        if timestamps and isinstance(timestamps, list):
            for sentence in timestamps:
                if not isinstance(sentence, dict):
                    continue
                sentence_text = str(sentence.get("text") or sentence.get("sentence") or text).strip()
                if not sentence_text:
                    continue
                start = float(sentence.get("start", sentence.get("begin", 0)) or 0) / 1000.0
                end = float(sentence.get("end", sentence.get("time", 0)) or 0) / 1000.0
                if end <= start and audio_duration > start:
                    end = audio_duration
                normalized.append({"start": start, "end": max(start, end), "text": sentence_text})
        else:
            start = float(item.get("start", 0) or 0)
            end = float(item.get("end", 0) or 0)
            if end <= start and audio_duration > start:
                # Local Paraformer checkpoints may omit sentence timestamps
                # when VAD/punctuation companions are unavailable. Preserve
                # the full speech duration so the canonical timeline can split
                # the transcript instead of collapsing it to 0.1 seconds.
                end = audio_duration
            normalized.append({"start": start, "end": max(start, end), "text": text})
    logger.info("paraformer.done model=%s language=%s segments=%s", selected_model, language or "auto", len(normalized))
    return normalized
