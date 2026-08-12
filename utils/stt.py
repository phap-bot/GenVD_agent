from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import subprocess
import uuid
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("auto_dubbing.stt")

DEFAULT_STT_BASE_URL = "http://localhost:20128/v1"
DEFAULT_STT_MODEL = "gemini/gemini-2.5-flash"
REMOTE_STT_PROVIDERS = {"9router", "remote", "openai-compatible", "openai_compatible", "gemini"}
LOCAL_ASR_MODELS = [
    ("tiny", "WhisperX local (tiny int8)"),
    ("base", "WhisperX local (base int8)"),
    ("small", "WhisperX local (small int8)"),
    ("medium", "WhisperX local (medium int8)"),
    ("large-v3", "WhisperX local (large-v3 float16)"),
]
LOCAL_ASR_MODEL_IDS = {model_id for model_id, _label in LOCAL_ASR_MODELS}
FALLBACK_STT_MODELS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-2.5-pro",
]
DEFAULT_STT_FALLBACK_MODELS = [
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-3-flash-preview",
]

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass


def remote_stt_enabled(asr_model: str | None = None) -> bool:
    clean_model = (asr_model or "").strip()
    if clean_model in LOCAL_ASR_MODEL_IDS:
        return False
    if "/" in clean_model:
        return True

    provider = os.environ.get("AUTODUB_ASR_PROVIDER", "").strip().lower()
    if provider in REMOTE_STT_PROVIDERS:
        return True
    return False


def list_stt_models(timeout: float = 2.0) -> dict[str, object]:
    base_url = _stt_base_url()
    api_key = _stt_api_key()
    provider = os.environ.get("AUTODUB_ASR_PROVIDER", "whisperx").strip().lower() or "whisperx"
    remote_models = FALLBACK_STT_MODELS
    source = "fallback"

    try:
        fetched_models = _list_openai_compatible_models(base_url, api_key, timeout=timeout)
        if fetched_models:
            remote_models = _dedupe_models([*FALLBACK_STT_MODELS, *fetched_models])
            source = "9router"
    except (OSError, TimeoutError, URLError) as exc:
        logger.warning("stt.models.unavailable base_url=%s error=%s", base_url, exc)
    except Exception:
        logger.warning("stt.models.fetch_failed base_url=%s", base_url, exc_info=True)

    configured_default = _selected_stt_model(None)
    default_model = configured_default if provider in REMOTE_STT_PROVIDERS else "base"
    if default_model not in {item[0] for item in LOCAL_ASR_MODELS} and default_model not in remote_models:
        remote_models = [default_model, *remote_models]

    models = [
        *({"id": model_id, "label": label} for model_id, label in LOCAL_ASR_MODELS),
        *({"id": model, "label": _stt_model_label(model)} for model in _dedupe_models(remote_models)),
    ]
    return {
        "provider": provider,
        "base_url": base_url,
        "api_key_configured": bool(api_key),
        "default_model": default_model,
        "source": source,
        "models": models,
    }


def transcribe_audio_remote(
    audio_path: Path,
    *,
    source_language: str | None = None,
    model: str | None = None,
    prompt: str | None = None,
    timeout: float | None = None,
) -> list[dict[str, object]]:
    selected_model = _selected_stt_model(model)
    duration = _audio_duration(audio_path)
    selected_prompt = prompt if prompt is not None else os.environ.get("AUTODUB_STT_PROMPT")
    selected_timeout = timeout if timeout is not None else _env_float("AUTODUB_STT_TIMEOUT", 120.0)
    upload_path = _prepare_upload_audio(audio_path)
    try:
        payload = _request_transcription_with_model_fallback(
            upload_path,
            selected_model=selected_model,
            source_language=source_language,
            prompt=selected_prompt,
            timeout=selected_timeout,
        )
    finally:
        _cleanup_upload_audio(upload_path, audio_path)
    segments = _segments_from_payload(payload, fallback_duration=duration)
    logger.info(
        "stt.remote.done model=%s language=%s segments=%s duration=%.3f",
        str(payload.get("_autodub_model") or selected_model),
        source_language or "auto",
        len(segments),
        duration,
    )
    return segments


def _prepare_upload_audio(audio_path: Path) -> Path:
    upload_format = os.environ.get("AUTODUB_STT_UPLOAD_FORMAT", "mp3").strip().lower()
    if upload_format in {"", "source", "wav", "none", "off"}:
        logger.info(
            "stt.upload_audio.source path=%s bytes=%s",
            audio_path,
            audio_path.stat().st_size if audio_path.exists() else 0,
        )
        return audio_path

    if upload_format not in {"mp3", "m4a", "opus"}:
        logger.warning("stt.upload_audio.unsupported_format format=%s using_source", upload_format)
        return audio_path

    bitrate = os.environ.get("AUTODUB_STT_UPLOAD_BITRATE", "32k").strip() or "32k"
    upload_path = audio_path.with_name(f"{audio_path.stem}_stt_upload_{uuid.uuid4().hex[:8]}.{upload_format}")
    codec_args = {
        "mp3": ["-codec:a", "libmp3lame"],
        "m4a": ["-codec:a", "aac"],
        "opus": ["-codec:a", "libopus"],
    }[upload_format]
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        *codec_args,
        "-b:a",
        bitrate,
        str(upload_path),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        original_bytes = audio_path.stat().st_size if audio_path.exists() else 0
        upload_bytes = upload_path.stat().st_size if upload_path.exists() else 0
        if upload_bytes <= 0:
            logger.warning("stt.upload_audio.empty output=%s using_source", upload_path)
            _cleanup_upload_audio(upload_path, audio_path)
            return audio_path
        logger.info(
            "stt.upload_audio.compressed source=%s source_bytes=%s upload=%s upload_bytes=%s format=%s bitrate=%s",
            audio_path,
            original_bytes,
            upload_path,
            upload_bytes,
            upload_format,
            bitrate,
        )
        return upload_path
    except Exception as exc:
        stderr = getattr(exc, "stderr", "") or ""
        logger.warning(
            "stt.upload_audio.compress_failed format=%s source=%s stderr=%s using_source",
            upload_format,
            audio_path,
            stderr.strip(),
        )
        _cleanup_upload_audio(upload_path, audio_path)
        return audio_path


def _cleanup_upload_audio(upload_path: Path, source_path: Path) -> None:
    if upload_path == source_path:
        return
    try:
        upload_path.unlink(missing_ok=True)
    except Exception:
        logger.warning("stt.upload_audio.cleanup_failed path=%s", upload_path, exc_info=True)


def _request_transcription_with_model_fallback(
    audio_path: Path,
    *,
    selected_model: str,
    source_language: str | None,
    prompt: str | None,
    timeout: float,
) -> dict[str, object]:
    last_error: Exception | None = None
    for model in _stt_model_attempts(selected_model):
        try:
            payload = _request_transcription(
                audio_path,
                model=model,
                source_language=source_language,
                prompt=prompt,
                timeout=timeout,
            )
            payload["_autodub_model"] = model
            return payload
        except Exception as exc:
            last_error = exc
            logger.warning(
                "stt.remote.model_attempt_failed model=%s language=%s audio=%s error=%s",
                model,
                source_language or "auto",
                audio_path,
                exc,
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("Remote STT failed before any model attempt")


def _request_transcription(
    audio_path: Path,
    *,
    model: str,
    source_language: str | None,
    prompt: str | None,
    timeout: float,
) -> dict[str, object]:
    fields: dict[str, str] = {"model": model}
    if source_language and source_language != "auto":
        fields["language"] = source_language
    if prompt:
        fields["prompt"] = prompt

    response_format = os.environ.get("AUTODUB_STT_RESPONSE_FORMAT", "").strip()
    if response_format:
        fields["response_format"] = response_format

    body, content_type = _multipart_body(fields, "file", audio_path)
    request = Request(
        _api_url(_stt_base_url(), "audio/transcriptions"),
        data=body,
        headers={
            **_auth_headers(),
            "Content-Type": content_type,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote STT request failed with HTTP {exc.code}: {detail}") from exc

    parsed = json.loads(raw_body)
    if not isinstance(parsed, dict):
        raise RuntimeError("Remote STT response is not a JSON object")
    return parsed


def _segments_from_payload(payload: dict[str, object], *, fallback_duration: float) -> list[dict[str, object]]:
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        segments: list[dict[str, object]] = []
        for index, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
            segments.append(
                {
                    "id": index,
                    "start": max(0.0, start),
                    "end": max(start + 0.1, end),
                    "text": text,
                    "words": item.get("words", []),
                }
            )
        if segments:
            return segments

    text = str(payload.get("text", "")).strip()
    if not text:
        return []
    return _split_text_over_duration(text, fallback_duration)


def _split_text_over_duration(text: str, duration: float) -> list[dict[str, object]]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？；;])\s*", text) if part.strip()]
    if len(parts) <= 1:
        parts = [part.strip() for part in re.split(r"[\n\r]+", text) if part.strip()]
    if len(parts) <= 1:
        parts = [text]

    total_chars = sum(max(1, len(part)) for part in parts)
    cursor = 0.0
    safe_duration = max(duration, len(parts) * 0.5)
    segments: list[dict[str, object]] = []
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            end = safe_duration
        else:
            end = cursor + safe_duration * (max(1, len(part)) / total_chars)
        segments.append(
            {
                "id": index,
                "start": cursor,
                "end": max(cursor + 0.1, end),
                "text": part,
                "words": [],
            }
        )
        cursor = end
    return segments


def _multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----autodub-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    content_type = mimetypes.guess_type(file_path.name)[0] or "audio/wav"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            frame_rate = wav.getframerate() or 1
            return wav.getnframes() / frame_rate
    except Exception:
        logger.warning("stt.audio_duration.failed path=%s", path, exc_info=True)
        return 0.0


def _selected_stt_model(model: str | None) -> str:
    if model and "/" in model:
        return model
    return (os.environ.get("AUTODUB_STT_MODEL") or DEFAULT_STT_MODEL).strip()


def _stt_model_attempts(primary_model: str) -> list[str]:
    configured_fallbacks = os.environ.get("AUTODUB_STT_FALLBACK_MODELS", "")
    fallback_models = (
        [item.strip() for item in configured_fallbacks.split(",") if item.strip()]
        if configured_fallbacks.strip()
        else DEFAULT_STT_FALLBACK_MODELS
    )
    return _dedupe_models([primary_model, *fallback_models])


def _stt_base_url() -> str:
    return (
        os.environ.get("AUTODUB_STT_BASE_URL")
        or os.environ.get("AUTODUB_9ROUTER_BASE_URL")
        or DEFAULT_STT_BASE_URL
    ).rstrip("/")


def _auth_headers() -> dict[str, str]:
    api_key = _stt_api_key()
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _stt_api_key() -> str | None:
    api_key = (
        os.environ.get("AUTODUB_STT_API_KEY")
        or os.environ.get("AUTODUB_9ROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        return None
    if api_key.lower() in {"replace-with-your-9router-token", "changeme", "your-api-key"}:
        return None
    if any(mask in api_key for mask in ("\u2022", "*", "...", "\u2026")):
        return None
    return api_key


def _list_openai_compatible_models(base_url: str, api_key: str | None, timeout: float) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = Request(_api_url(base_url, "models"), headers=headers, method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    raw_models = payload.get("data", payload)
    if isinstance(raw_models, dict):
        raw_models = raw_models.get("models", raw_models.get("data", []))
    if not isinstance(raw_models, list):
        return []

    model_ids: list[str] = []
    for item in raw_models:
        if isinstance(item, str):
            model_ids.append(item)
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("model")
            if isinstance(model_id, str):
                model_ids.append(model_id)

    filtered = [model for model in model_ids if _is_likely_stt_model(model)]
    return _dedupe_models(filtered or model_ids)


def _is_likely_stt_model(model: str) -> bool:
    lowered = model.lower()
    if any(blocked in lowered for blocked in ("image", "tts", "embed", "rerank", "lyria")):
        return False
    if any(keyword in lowered for keyword in ("whisper", "transcrib", "speech", "stt")):
        return True
    return lowered.startswith("gemini/") and any(
        keyword in lowered
        for keyword in (
            "2.5-pro",
            "2.5-flash",
            "2.5-flash-lite",
            "3-flash",
            "3.1-flash",
            "3.1-pro",
        )
    )


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for model in models:
        clean_model = model.strip()
        if clean_model and clean_model not in seen:
            seen.add(clean_model)
            deduped.append(clean_model)
    return deduped


def _stt_model_label(model: str) -> str:
    labels = {
        "gemini/gemini-2.5-pro": "Gemini 2.5 Pro STT qua 9Router",
        "gemini/gemini-2.5-flash": "Gemini 2.5 Flash STT qua 9Router",
        "gemini/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite STT qua 9Router",
        "gemini/gemini-3-flash-preview": "Gemini 3 Flash Preview STT qua 9Router",
    }
    return labels.get(model, model)


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _env_float(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, fallback))
    except (TypeError, ValueError):
        return fallback
