from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from threading import Event
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger("auto_dubbing.translation")

DEFAULT_9ROUTER_BASE_URL = "http://localhost:20128/v1"
DEFAULT_TRANSLATION_MODEL = "ag/gemini-3-flash-agent"
GATEWAY_MODEL_PREFIXES = ("9router/",)
GATEWAY_MODEL_ALIASES = {
    "deepseek-v4-flash-free": "oc/deepseek-v4-flash-free",
    "mimo-v2.5-free": "oc/mimo-v2.5-free",
}
DEFAULT_TRANSLATION_FALLBACK_MODELS = [
    "ag/gemini-3-flash-agent",
    "ag/gemini-3.5-flash-low",
    "ag/gemini-3-flash",
    "gemini/gemini-3-flash-preview",
]
SHORTEN_WORDS_PER_SECOND = 3.0
VI_ZH_SHORT_TRANSLATION_OVERRIDES = {
    "不负责": "Không chịu trách nhiệm",
    "给我赚两块钱": "Kiếm cho tôi hai tệ đi",
    "给我赚两块钱吧": "Kiếm cho tôi hai tệ đi",
}
FALLBACK_TRANSLATION_MODELS = [
    "ag/gemini-3-flash-agent",
    "oc/deepseek-v4-flash-free",
    "oc/mimo-v2.5-free",
    "ag/gemini-pro-agent",
    "ag/gemini-3.1-pro-low",
    "ag/gemini-3.5-flash-low",
    "ag/gemini-3.5-flash-extra-low",
    "ag/gemini-3-flash",
    "openrouter/google/gemini-2.5-pro-exp-03-25:free",
    "openrouter/google/gemini-2.0-flash-exp:free",
    "openrouter/google/gemini-2.0-flash-thinking-exp:free",
    "openrouter/google/gemini-2.0-flash-lite-preview-02-05:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/google/gemma-4-31b-it:free",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-3.1-flash-lite-preview",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemma-4-31b-it",
]

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass


def translate_text(
    text: str,
    *,
    target_language: str,
    source_language: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    cancel_event: Event | None = None,
    failed_models: set[str] | None = None,
) -> str:
    clean_text = text.strip()
    if not clean_text:
        return text

    target = _normalize_language(target_language, fallback="vi")
    source = _normalize_language(source_language, fallback="auto")
    selected_provider = (provider or _config_value("AUTODUB_TRANSLATION_PROVIDER") or "9router").strip().lower()
    selected_model = _selected_translation_model(model)

    if selected_provider in {"mock", "none", "off"}:
        return f"[{target}] {clean_text}"

    try:
        _raise_if_cancelled(cancel_event)
        if selected_provider in {"9router", "ninerouter", "openai-compatible", "openai_compatible"}:
            translated = _translate_9router_text_with_model_fallback(
                clean_text,
                source=source,
                target=target,
                selected_model=selected_model,
                base_url=_nine_router_base_url(),
                api_key=_nine_router_api_key() or "",
                timeout=timeout,
                cancel_event=cancel_event,
                failed_models=failed_models,
            )
            return _fallback_if_bad_translation(
                clean_text,
                translated,
                source=source,
                target=target,
                timeout=timeout,
            )
        translated = _translate_google_gtx_cached(clean_text, source, target, timeout)
        return _fallback_if_bad_translation(
            clean_text,
            translated,
            source=source,
            target=target,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "translation.failed provider=%s model=%s source=%s target=%s text_len=%s error=%s",
            selected_provider,
            selected_model,
            source,
            target,
            len(clean_text),
            exc,
        )
        if (_config_value("AUTODUB_TRANSLATION_FALLBACK") or "google").strip().lower() == "google":
            return _translate_google_gtx_cached(clean_text, source, target, timeout)
        return clean_text


def translate_segments(
    texts: list[str],
    *,
    target_language: str,
    target_durations: list[float] | None = None,
    source_language: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    timeout: float = 45.0,
    cancel_event: Event | None = None,
    batch_size: int | None = None,
    context: str | None = None,
    cps_budget: float | None = None,
) -> list[str]:
    if not texts:
        return []

    target = _normalize_language(target_language, fallback="vi")
    source = _normalize_language(source_language, fallback="auto")
    selected_provider = (provider or _config_value("AUTODUB_TRANSLATION_PROVIDER") or "9router").strip().lower()
    selected_model = _selected_translation_model(model)
    clean_texts = [text.strip() for text in texts]
    clean_durations = [max(0.1, float(value)) for value in (target_durations or [])]
    if len(clean_durations) != len(clean_texts):
        clean_durations = [0.0] * len(clean_texts)
    failed_models: set[str] = set()

    if selected_provider in {"mock", "none", "off"}:
        return [f"[{target}] {text}" if text else text for text in clean_texts]

    if selected_provider not in {"9router", "ninerouter", "openai-compatible", "openai_compatible"}:
        translated_texts = [
            _translate_google_gtx_cached(text, source, target, timeout) if text else text
            for text in clean_texts
        ]
        return [
            _fallback_if_bad_translation(
                source_text,
                translated_text,
                source=source,
                target=target,
                timeout=timeout,
            )
            if source_text
            else source_text
            for source_text, translated_text in zip(clean_texts, translated_texts)
        ]

    translated: list[str] = []
    batch_offset = 0
    for batch in _translation_batches(clean_texts, max_items=batch_size):
        _raise_if_cancelled(cancel_event)
        batch_durations = tuple(clean_durations[batch_offset : batch_offset + len(batch)])
        batch_offset += len(batch)
        try:
            translated.extend(
                _translate_9router_segments_with_model_fallback(
                    tuple(batch),
                    target_durations=batch_durations,
                    source=source,
                    target=target,
                    selected_model=selected_model,
                    base_url=_nine_router_base_url(),
                    api_key=_nine_router_api_key() or "",
                    timeout=timeout,
                    cancel_event=cancel_event,
                    failed_models=failed_models,
                    context=context or "",
                    cps_budget=cps_budget or 12.5,
                )
            )
        except Exception as exc:
            logger.warning(
                "translation.batch_failed provider=%s model=%s source=%s target=%s batch_size=%s error=%s",
                selected_provider,
                selected_model,
                source,
                target,
                len(batch),
                exc,
            )
            translated.extend(
                [
                    translate_text(
                        text,
                        source_language=source,
                        target_language=target,
                        provider=selected_provider,
                        model=selected_model,
                        timeout=timeout,
                        cancel_event=cancel_event,
                        failed_models=failed_models,
                    )
                    if text
                    else text
                    for text in batch
                ]
            )

    if len(translated) != len(clean_texts):
        logger.warning(
            "translation.batch_size_mismatch input=%s output=%s falling_back_per_segment",
            len(clean_texts),
            len(translated),
        )
        translated = [
            translate_text(
                text,
                source_language=source,
                target_language=target,
                provider=selected_provider,
                model=selected_model,
                timeout=timeout,
                cancel_event=cancel_event,
                failed_models=failed_models,
            )
            if text
            else text
            for text in clean_texts
        ]

    return [
        _fallback_if_bad_translation(
            source_text,
            translated_text,
            source=source,
            target=target,
            timeout=timeout,
        )
        if source_text
        else source_text
        for source_text, translated_text in zip(clean_texts, translated)
    ]


def list_translation_models(timeout: float = 2.0) -> dict[str, object]:
    base_url = _nine_router_base_url()
    api_key = _nine_router_api_key()
    models = FALLBACK_TRANSLATION_MODELS
    source = "fallback"

    try:
        fetched_models = _list_9router_models(base_url, api_key, timeout=timeout)
        if fetched_models:
            models = _dedupe_models([_gateway_model_id(model) for model in [*FALLBACK_TRANSLATION_MODELS, *fetched_models]])
            source = "9router"
    except (OSError, TimeoutError, URLError) as exc:
        try:
            from utils.ninerouter import ensure_9router_running
            if ensure_9router_running(base_url, wait_timeout=5.0):
                fetched_models = _list_9router_models(base_url, api_key, timeout=timeout)
                if fetched_models:
                    models = _dedupe_models([_gateway_model_id(model) for model in [*FALLBACK_TRANSLATION_MODELS, *fetched_models]])
                    source = "9router"
            else:
                logger.warning("translation.models.unavailable base_url=%s error=%s", base_url, exc)
        except Exception:
            logger.warning("translation.models.unavailable base_url=%s error=%s", base_url, exc)
    except Exception:
        logger.warning("translation.models.fetch_failed base_url=%s", base_url, exc_info=True)

    models = _dedupe_models([_gateway_model_id(model) for model in models])
    configured_default = _selected_translation_model(None)
    has_explicit_default = bool(_config_value("AUTODUB_TRANSLATION_MODEL"))
    default_model = (
        configured_default
        if configured_default in models or has_explicit_default
        else _preferred_default_model(models)
    )
    if default_model not in models:
        models = [default_model, *models]

    return {
        "provider": "9router",
        "base_url": base_url,
        "api_key_configured": bool(api_key),
        "default_model": default_model,
        "source": source,
        "models": [{"id": model, "label": _translation_model_label(model)} for model in models],
    }


def normalize_translation_model(model: str | None) -> str:
    return _selected_translation_model(model)


def shorten_text_for_duration(
    text: str,
    *,
    target_duration: float,
    target_language: str = "vi",
    source_text: str | None = None,
    context: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_words: int | None = None,
    timeout: float = 20.0,
) -> tuple[str, int, str, str]:
    clean_text = text.strip()
    selected_provider = (provider or _config_value("AUTODUB_TRANSLATION_PROVIDER") or "9router").strip().lower()
    selected_model = _selected_translation_model(model)
    word_limit = _shorten_word_limit(target_duration, max_words)
    if not clean_text:
        return text, word_limit, selected_provider, selected_model

    target = _normalize_language(target_language, fallback="vi")
    if _count_words(clean_text) <= word_limit:
        return clean_text, word_limit, selected_provider, selected_model

    if selected_provider in {"mock", "none", "off", "google"}:
        return _shorten_heuristic(clean_text, word_limit), word_limit, selected_provider, selected_model

    try:
        if selected_provider in {"9router", "ninerouter", "openai-compatible", "openai_compatible"}:
            shortened = _shorten_9router_text_with_model_fallback(
                clean_text,
                source_text=(source_text or "").strip(),
                context=(context or "").strip(),
                target=target,
                target_duration=max(0.1, target_duration),
                max_words=word_limit,
                selected_model=selected_model,
                base_url=_nine_router_base_url(),
                api_key=_nine_router_api_key() or "",
                timeout=timeout,
            )
            return _enforce_shortened_text(shortened, clean_text, word_limit), word_limit, selected_provider, selected_model
    except Exception as exc:
        logger.warning(
            "shorten.failed provider=%s model=%s target=%s text_len=%s max_words=%s error=%s",
            selected_provider,
            selected_model,
            target,
            len(clean_text),
            word_limit,
            exc,
        )

    return _shorten_heuristic(clean_text, word_limit), word_limit, selected_provider, selected_model


def shorten_segments_for_duration(
    texts: list[str],
    *,
    target_durations: list[float],
    target_language: str = "vi",
    source_texts: list[str] | None = None,
    context: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_words: list[int] | None = None,
    timeout: float = 45.0,
    batch_size: int = 40,
    cancel_event: Event | None = None,
) -> list[str]:
    """Review timing-constrained lines in bounded batches, never one request per line."""
    if not texts:
        return []
    if len(target_durations) != len(texts):
        raise ValueError("target_durations must match texts")
    clean_texts = [text.strip() for text in texts]
    clean_sources = [text.strip() for text in (source_texts or texts)]
    if len(clean_sources) != len(clean_texts):
        raise ValueError("source_texts must match texts")
    limits = list(max_words or [
        _shorten_word_limit(duration, None) for duration in target_durations
    ])
    if len(limits) != len(clean_texts):
        raise ValueError("max_words must match texts")
    limits = [max(1, int(value)) for value in limits]

    selected_provider = (provider or _config_value("AUTODUB_TRANSLATION_PROVIDER") or "9router").strip().lower()
    selected_model = _selected_translation_model(model)
    target = _normalize_language(target_language, fallback="vi")
    if selected_provider not in {"9router", "ninerouter", "openai-compatible", "openai_compatible"}:
        return [_shorten_heuristic(text, limit) for text, limit in zip(clean_texts, limits)]

    results: list[str] = []
    offset = 0
    batches = _translation_batches(clean_texts, max_items=batch_size)
    logger.info(
        "shorten.batch_plan segments=%s batches=%s max_items=%s",
        len(clean_texts),
        len(batches),
        batch_size,
    )
    for batch_number, batch in enumerate(batches, start=1):
        _raise_if_cancelled(cancel_event)
        size = len(batch)
        logger.info(
            "shorten.batch_start batch=%s/%s segments=%s",
            batch_number,
            len(batches),
            size,
        )
        batch_durations = tuple(max(0.1, float(value)) for value in target_durations[offset : offset + size])
        batch_sources = tuple(clean_sources[offset : offset + size])
        batch_limits = tuple(limits[offset : offset + size])
        try:
            shortened = _shorten_9router_segments_cached(
                tuple(batch),
                batch_sources,
                batch_durations,
                batch_limits,
                target,
                selected_model,
                _nine_router_base_url(),
                _nine_router_api_key() or "",
                _translation_attempt_timeout(timeout),
                context or "",
            )
            results.extend(
                _enforce_shortened_text(value, original, limit)
                for value, original, limit in zip(shortened, batch, batch_limits)
            )
            logger.info(
                "shorten.batch_done batch=%s/%s segments=%s",
                batch_number,
                len(batches),
                size,
            )
        except Exception as exc:
            logger.warning(
                "shorten.batch_failed provider=%s model=%s target=%s batch_size=%s fallback=heuristic error=%s",
                selected_provider,
                selected_model,
                target,
                size,
                exc,
            )
            results.extend(
                _shorten_heuristic(original, limit)
                for original, limit in zip(batch, batch_limits)
            )
        offset += size
    return results


def _translate_9router_text_with_model_fallback(
    text: str,
    *,
    source: str,
    target: str,
    selected_model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    cancel_event: Event | None = None,
    failed_models: set[str] | None = None,
) -> str:
    last_error: Exception | None = None
    _raise_if_cancelled(cancel_event)
    attempt_timeout = _translation_attempt_timeout(timeout)
    attempt_models = _translation_model_attempts(selected_model, failed_models=failed_models)
    if not attempt_models:
        raise RuntimeError("No 9Router translation models left after failed attempts")
    for model in attempt_models:
        try:
            return _with_connection_refused_retries(
                lambda: _translate_9router_cached(text, source, target, model, base_url, api_key, attempt_timeout),
                model=model,
                source=source,
                target=target,
                batch_size=1,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            last_error = exc
            _remember_failed_translation_model(model, failed_models)
            logger.warning(
                "translation.model_attempt_failed model=%s source=%s target=%s error=%s",
                model,
                source,
                target,
                exc,
            )
    if last_error is not None:
        raise last_error
    return text


def _translate_9router_segments_with_model_fallback(
    texts: tuple[str, ...],
    *,
    target_durations: tuple[float, ...],
    source: str,
    target: str,
    selected_model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    cancel_event: Event | None = None,
    failed_models: set[str] | None = None,
    context: str = "",
    cps_budget: float = 12.5,
) -> list[str]:
    last_error: Exception | None = None
    _raise_if_cancelled(cancel_event)
    attempt_timeout = _translation_attempt_timeout(timeout)
    attempt_models = _translation_model_attempts(selected_model, failed_models=failed_models)
    if not attempt_models:
        raise RuntimeError("No 9Router translation models left after failed attempts")
    for model in attempt_models:
        try:
            return _with_connection_refused_retries(
                lambda: _translate_9router_segments_cached(
                    texts,
                    target_durations,
                    source,
                    target,
                    model,
                    base_url,
                    api_key,
                    attempt_timeout,
                    context,
                    float(cps_budget),
                ),
                model=model,
                source=source,
                target=target,
                batch_size=len(texts),
                cancel_event=cancel_event,
            )
        except Exception as exc:
            last_error = exc
            _remember_failed_translation_model(model, failed_models)
            logger.warning(
                "translation.batch_model_attempt_failed model=%s source=%s target=%s batch_size=%s error=%s",
                model,
                source,
                target,
                len(texts),
                exc,
            )

    if len(texts) > 1:
        logger.warning(
            "translation.batch_split_fallback source=%s target=%s batch_size=%s reason=%s",
            source,
            target,
            len(texts),
            last_error,
        )
        midpoint = max(1, len(texts) // 2)
        return [
            *_translate_9router_segments_with_model_fallback(
                texts[:midpoint],
                target_durations=target_durations[:midpoint],
                source=source,
                target=target,
                selected_model=selected_model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                cancel_event=cancel_event,
                failed_models=failed_models,
                context=context,
                cps_budget=cps_budget,
            ),
            *_translate_9router_segments_with_model_fallback(
                texts[midpoint:],
                target_durations=target_durations[midpoint:],
                source=source,
                target=target,
                selected_model=selected_model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                cancel_event=cancel_event,
                failed_models=failed_models,
                context=context,
                cps_budget=cps_budget,
            ),
        ]
    if last_error is not None:
        raise last_error
    return list(texts)


def _shorten_9router_text_with_model_fallback(
    text: str,
    *,
    source_text: str,
    context: str,
    target: str,
    target_duration: float,
    max_words: int,
    selected_model: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> str:
    last_error: Exception | None = None
    for attempt_model in _translation_model_attempts(selected_model):
        try:
            return _shorten_9router_cached(
                text,
                source_text,
                context,
                target,
                target_duration,
                max_words,
                attempt_model,
                base_url,
                api_key,
                timeout,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "shorten.model_attempt_failed model=%s target=%s max_words=%s error=%s",
                attempt_model,
                target,
                max_words,
                exc,
            )
    if last_error is not None:
        raise last_error
    return text


@lru_cache(maxsize=2048)
def _translate_9router_cached(
    text: str,
    source: str,
    target: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "temperature": 0.0,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior audiovisual translator and Vietnamese film-review storyteller. "
                    "Translate the meaning faithfully, then phrase it as natural spoken narration for "
                    "a movie review: cinematic, confident, creative, and lightly inspirational when "
                    "the source allows it. Preserve speaker intent, tone, negation, names, numbers, "
                    "money amounts, slang intensity, and implied relationships. Do not invent plot "
                    "facts, jokes, motives, or expert claims not supported by the source. If the "
                    "source is a short fragment, keep a natural short fragment in the target language. "
                    "Return only the translated line, with no markdown, labels, notes, romanization, "
                    "pinyin, or explanation."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "source_language": _language_name(source),
                        "target_language": _language_name(target),
                        "text": text,
                        "rules": _translation_rules(target),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = Request(
        _api_url(base_url, "chat/completions"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_openai_compatible_headers(api_key),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 401:
            raise RuntimeError(
                "9Router rejected the request with 401 Unauthorized. "
                "Set AUTODUB_9ROUTER_API_KEY to your 9Router bearer token, "
                "or configure 9Router to allow local OpenAI-compatible clients."
            ) from exc
        raise RuntimeError(f"9Router request failed with HTTP {exc.code}: {detail}") from exc

    translated = _parse_chat_completion_body(body, content_type).strip()
    logger.info(
        "translation.done provider=9router model=%s source=%s target=%s input_len=%s output_len=%s",
        model,
        source,
        target,
        len(text),
        len(translated),
    )
    return translated or text


@lru_cache(maxsize=1024)
def _shorten_9router_cached(
    text: str,
    source_text: str,
    context: str,
    target: str,
    target_duration: float,
    max_words: int,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "temperature": 0.2,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional dubbing script editor for cinematic review narration. "
                    "Shorten translated dialogue only as much as needed for a natural speaking pace, "
                    "while preserving the core meaning, tone, names, numbers, and intent. Keep the "
                    "line vivid and review-like when possible, but never summarize away the meaning. "
                    "Return only the shortened line, with no markdown, labels, quotes, or explanation."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": _language_name(target),
                        "target_duration_seconds": round(target_duration, 3),
                        "absolute_max_words": max_words,
                        "reading_speed_rule": "Use 3 spoken words per second. Never exceed absolute_max_words.",
                        "source_text": source_text,
                        "current_translation": text,
                        "context": context,
                        "rules": [
                            "Keep the result natural for dubbing and cinematic review narration, not a literal summary.",
                            "Preserve the main action, speaker attitude, names, numbers, money amounts, and negation.",
                            "Drop filler words, particles, duplicate phrasing, and nonessential politeness first.",
                            "Do not remove the hook, emotional direction, or key judgment if it is present in the source.",
                            "For Vietnamese, use short spoken Vietnamese and avoid Chinese/Japanese/Korean characters unless they are names.",
                            "Return exactly one shortened line.",
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = Request(
        _api_url(base_url, "chat/completions"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_openai_compatible_headers(api_key),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"9Router shorten request failed with HTTP {exc.code}: {detail}") from exc

    shortened = _clean_shortened_text(_parse_chat_completion_body(body, content_type))
    logger.info(
        "shorten.done provider=9router model=%s target=%s max_words=%s input_words=%s output_words=%s",
        model,
        target,
        max_words,
        _count_words(text),
        _count_words(shortened),
    )
    return shortened or text


@lru_cache(maxsize=256)
def _shorten_9router_segments_cached(
    texts: tuple[str, ...],
    source_texts: tuple[str, ...],
    target_durations: tuple[float, ...],
    max_words: tuple[int, ...],
    target: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    context: str = "",
) -> list[str]:
    payload = {
        "model": model,
        "temperature": 0.1,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional dubbing script editor. Shorten every numbered translated "
                    "segment only enough to fit its timing budget while preserving meaning, names, "
                    "numbers, negation, tone, and the main action. Return only valid JSON. Do not skip, "
                    "merge, split, or reorder ids."
                    + (f" Continuity context (do not repeat it): {context}" if context else "")
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": _language_name(target),
                        "response_schema": {"segments": [{"id": 0, "text": "shortened segment 0"}]},
                        "rules": [
                            "Return exactly one object for every input id and no extra ids.",
                            "Never exceed max_spoken_words for each segment.",
                            "Preserve the key meaning; remove filler and repetition first.",
                            "Use natural complete spoken phrasing; never cut a word or end on a dangling conjunction.",
                        ],
                        "segments": [
                            {
                                "id": index,
                                "source_text": source_texts[index],
                                "current_translation": text,
                                "target_duration_seconds": round(target_durations[index], 3),
                                "max_spoken_words": max_words[index],
                            }
                            for index, text in enumerate(texts)
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = Request(
        _api_url(base_url, "chat/completions"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_openai_compatible_headers(api_key),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"9Router batch shorten request failed with HTTP {exc.code}: {detail}") from exc

    content = _parse_chat_completion_body(body, content_type).strip()
    shortened = _parse_translation_response(content, expected_count=len(texts))
    if len(shortened) != len(texts):
        raise ValueError(f"Expected {len(texts)} shortened segments, got {len(shortened)}")
    logger.info(
        "shorten.done provider=9router_batch model=%s target=%s segments=%s input_words=%s output_words=%s",
        model,
        target,
        len(texts),
        sum(_count_words(text) for text in texts),
        sum(_count_words(text) for text in shortened),
    )
    return shortened


@lru_cache(maxsize=256)
def _translate_9router_segments_cached(
    texts: tuple[str, ...],
    target_durations: tuple[float, ...],
    source: str,
    target: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    context: str = "",
    cps_budget: float = 12.5,
) -> list[str]:
    payload = {
        "model": model,
        "temperature": 0.0,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior audiovisual translator and Vietnamese film-review storyteller "
                    "for dubbing and subtitle timelines. Translate each numbered segment faithfully "
                    "while using neighboring segments only as context. Phrase the Vietnamese like a "
                    "natural movie-review narration: cinematic, confident, creative, and gently "
                    "inspirational when the source supports it. Preserve the timing-friendly brevity "
                    "of short lines without dropping important meaning. Infer omitted subjects "
                    "conservatively; do not add new facts, jokes, moralizing, plot details, or "
                    "expert opinions not implied by the source. Return only valid JSON, with no "
                    "markdown, notes, romanization, pinyin, or source-language text unless it is a "
                    "proper name."
                    + (f" Continuity context (do not repeat it): {context}" if context else "")
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "source_language": _language_name(source),
                        "target_language": _language_name(target),
                        "response_schema": {
                            "translations": [
                                {"id": 0, "text": "translated segment 0"},
                                {"id": 1, "text": "translated segment 1"},
                            ]
                        },
                        "rules": [
                            "Return one translation object for every input id. Do not skip, merge, split, or reorder ids.",
                            "The JSON must contain exactly the same ids as input and no extra ids.",
                            "Each text value must contain only that segment's translation.",
                            "Write complete spoken sentences; never end on a dangling conjunction or split a word.",
                            "Respect max_spoken_words for timing. Prefer concise natural phrasing over literal verbosity.",
                            *_translation_rules(target),
                        ],
                        "segments": [
                            {
                                "id": index,
                                "text": text,
                                "target_duration_seconds": round(target_durations[index], 3),
                                "max_spoken_words": (
                                    max(1, int(min(
                                        target_durations[index] * SHORTEN_WORDS_PER_SECOND,
                                        target_durations[index] * float(cps_budget) / 4.0,
                                    )))
                                    if target_durations[index] > 0
                                    else None
                                ),
                                "max_characters": (
                                    max(1, int(target_durations[index] * float(cps_budget)))
                                    if target_durations[index] > 0
                                    else None
                                ),
                            }
                            for index, text in enumerate(texts)
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = Request(
        _api_url(base_url, "chat/completions"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_openai_compatible_headers(api_key),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"9Router batch request failed with HTTP {exc.code}: {detail}") from exc

    content = _parse_chat_completion_body(body, content_type).strip()
    translated = _parse_translation_response(content, expected_count=len(texts))
    if len(translated) != len(texts):
        raise ValueError(f"Expected {len(texts)} translations, got {len(translated)}")

    logger.info(
        "translation.done provider=9router_batch model=%s source=%s target=%s segments=%s input_len=%s output_len=%s",
        model,
        source,
        target,
        len(texts),
        sum(len(text) for text in texts),
        sum(len(text) for text in translated),
    )
    return [item.strip() or source_text for item, source_text in zip(translated, texts)]


def _parse_chat_completion_body(body: str, content_type: str) -> str:
    if "text/event-stream" in content_type or body.lstrip().startswith("data:"):
        return _parse_chat_completion_sse(body)

    payload = json.loads(body)
    return (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def _parse_chat_completion_sse(body: str) -> str:
    parts: list[str] = []
    for line in body.splitlines():
        clean_line = line.strip()
        if not clean_line.startswith("data:"):
            continue
        data = clean_line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        choice = payload.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        message = choice.get("message", {})
        content = delta.get("content") or message.get("content") or ""
        if content:
            parts.append(content)
    return "".join(parts)


def _parse_translation_response(content: str, *, expected_count: int | None = None) -> list[str]:
    clean = content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("[")
        end = clean.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(clean[start : end + 1])

    if isinstance(parsed, dict):
        parsed = parsed.get("translations") or parsed.get("segments") or parsed.get("items")
    if not isinstance(parsed, list):
        raise ValueError("Translation response is not a JSON array")

    if parsed and all(isinstance(item, dict) for item in parsed):
        return _translation_texts_from_objects(parsed, expected_count=expected_count)

    translated = [str(item).strip() for item in parsed]
    if expected_count is not None and len(translated) > expected_count:
        translated = translated[:expected_count]
    return translated


def _translation_texts_from_objects(items: list[object], *, expected_count: int | None) -> list[str]:
    object_items = [item for item in items if isinstance(item, dict)]
    by_id: dict[int, str] = {}
    in_order: list[str] = []

    for fallback_id, item in enumerate(object_items):
        text_value = (
            item.get("text")
            or item.get("translation")
            or item.get("translated_text")
            or item.get("value")
            or ""
        )
        text = str(text_value).strip()
        raw_id = item.get("id", item.get("index", fallback_id))
        try:
            segment_id = int(raw_id)
        except (TypeError, ValueError):
            segment_id = fallback_id
        by_id[segment_id] = text
        in_order.append(text)

    if expected_count is None:
        return in_order

    if all(index in by_id for index in range(expected_count)):
        return [by_id[index] for index in range(expected_count)]

    return in_order[:expected_count]


def _translation_rules(target: str) -> list[str]:
    rules = [
        "Translate the intent and pragmatic meaning, not a literal word-by-word gloss.",
        "Stay semantically close: do not change who did what, the judgment, the emotion, or the stakes.",
        "Preserve negation, modality, speaker attitude, names, numbers, units, and money amounts.",
        "Keep very short replies short. If the source is a fragment, return a natural target-language fragment.",
        "Do not add explanations, apologies, safety disclaimers, plot facts, or content not implied by the source.",
        "Preserve profanity/slang intensity when present, but make it natural in the target language.",
    ]
    if target.startswith("vi"):
        rules.extend(
            [
                "Use natural spoken Vietnamese suitable for dubbing and movie-review narration.",
                "Let the phrasing sound like a sharp, charismatic reviewer, but keep every factual meaning anchored to the source.",
                "Do not copy Chinese/Japanese/Korean characters into Vietnamese output unless they are proper names.",
                "Avoid word-by-word Sino-Vietnamese. Prefer everyday Vietnamese phrasing.",
                "Translate Chinese money colloquialisms like 块/块钱 as tệ/đồng depending on context, not đô la.",
            ]
        )
    return rules


@lru_cache(maxsize=2048)
def _translate_google_gtx_cached(text: str, source: str, target: str, timeout: float) -> str:
    params = urlencode(
        {
            "client": "gtx",
            "sl": source or "auto",
            "tl": target,
            "dt": "t",
            "q": text,
        }
    )
    request = Request(
        f"https://translate.googleapis.com/translate_a/single?{params}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    translated = "".join(part[0] for part in payload[0] if part and part[0])
    logger.info(
        "translation.done provider=google_gtx source=%s target=%s input_len=%s output_len=%s",
        source,
        target,
        len(text),
        len(translated),
    )
    return translated.strip() or text


def _translation_batches(texts: list[str], *, max_items: int | None = None) -> list[list[str]]:
    max_items = max_items or _env_int("AUTODUB_TRANSLATION_BATCH_SIZE", 6, minimum=1, maximum=80)
    max_items = max(1, min(80, int(max_items)))
    max_chars = _env_int("AUTODUB_TRANSLATION_BATCH_CHARS", 900, minimum=120, maximum=20000)
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        projected_chars = current_chars + len(text)
        if current and (len(current) >= max_items or projected_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += len(text)
    if current:
        batches.append(current)
    return batches


def _fallback_if_bad_translation(
    source_text: str,
    translated_text: str,
    *,
    source: str,
    target: str,
    timeout: float,
) -> str:
    clean_translation = _clean_translated_text(translated_text)
    clean_translation = _postprocess_translation(source_text, clean_translation, target)
    if not _translation_needs_fallback(source_text, clean_translation, target):
        return clean_translation

    logger.warning(
        "translation.bad_output_fallback source=%s target=%s source_text=%s translated_text=%s",
        source,
        target,
        source_text,
        clean_translation,
    )
    if (_config_value("AUTODUB_TRANSLATION_FALLBACK") or "google").strip().lower() != "google":
        return clean_translation or source_text

    try:
        fallback = _translate_google_gtx_cached(source_text, source, target, timeout)
        fallback = _postprocess_translation(source_text, _clean_translated_text(fallback), target)
        return fallback or clean_translation or source_text
    except Exception:
        logger.exception("translation.google_fallback_failed source=%s target=%s", source, target)
        return clean_translation or source_text


def _translation_needs_fallback(source_text: str, translated_text: str, target: str) -> bool:
    if not translated_text:
        return True
    if _normalize_for_compare(source_text) == _normalize_for_compare(translated_text):
        return True
    if target.startswith("vi") and _contains_cjk(translated_text):
        return True
    return False


def _clean_translated_text(text: str) -> str:
    clean = text.strip()
    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {'"', "'"}:
        clean = clean[1:-1].strip()
    return clean


def _shorten_word_limit(target_duration: float, max_words: int | None) -> int:
    if max_words is not None:
        return max(1, max_words)
    return max(1, int(max(0.1, target_duration) * SHORTEN_WORDS_PER_SECOND))


def _count_words(text: str) -> int:
    clean = text.strip()
    if not clean:
        return 0
    return len([part for part in re.split(r"\s+", clean) if part])


def _clean_shortened_text(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:text|markdown)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    clean = clean.strip()
    clean = re.sub(r"^(?:câu rút gọn|rút gọn|shortened|output)\s*:\s*", "", clean, flags=re.IGNORECASE)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if lines:
        clean = lines[0]
    clean = re.sub(r"^\s*[-*•]\s*", "", clean).strip()
    if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {'"', "'", "“", "”"}:
        clean = clean[1:-1].strip()
    return clean


def _enforce_shortened_text(candidate: str, original: str, max_words: int) -> str:
    clean = _clean_shortened_text(candidate)
    if clean and _count_words(clean) <= max_words:
        return clean
    if clean:
        return _shorten_heuristic(clean, max_words)
    return _shorten_heuristic(original, max_words)


def _shorten_heuristic(text: str, max_words: int) -> str:
    words = [part for part in re.split(r"\s+", text.strip()) if part]
    if len(words) <= max_words:
        return text.strip()
    shortened = " ".join(words[:max_words]).strip()
    return re.sub(r"[\s,;:]+$", "", shortened).strip() or text.strip()


def _postprocess_translation(source_text: str, translated_text: str, target: str) -> str:
    if not re.search(r"(?:\.{2,}|…)", source_text):
        translated_text = re.sub(r"^\s*(?:\.{2,}|…)+\s*", "", translated_text)
        translated_text = re.sub(r"\s*(?:\.{2,}|…)+\s*$", "", translated_text)

    if not target.startswith("vi"):
        return translated_text

    normalized_source = _normalize_for_compare(source_text)
    override = VI_ZH_SHORT_TRANSLATION_OVERRIDES.get(normalized_source)
    if override:
        return override

    if "块钱" in source_text and "đô la" in translated_text.lower():
        translated_text = re.sub("đô la", "tệ", translated_text, flags=re.IGNORECASE)
    return translated_text


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]", text))


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def _env_int(name: str, fallback: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(_config_value(name) or fallback)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, value))


def _translation_attempt_timeout(timeout: float) -> float:
    try:
        configured = float(_config_value("AUTODUB_TRANSLATION_ATTEMPT_TIMEOUT") or 30.0)
    except (TypeError, ValueError):
        configured = 30.0
    return max(5.0, min(float(timeout), configured))


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Translation cancelled")


def _with_connection_refused_retries(
    operation,
    *,
    model: str,
    source: str,
    target: str,
    batch_size: int,
    cancel_event: Event | None = None,
):
    retries = _env_int("AUTODUB_TRANSLATION_CONNECTION_REFUSED_RETRIES", 2, minimum=0, maximum=8)
    for attempt in range(retries + 1):
        try:
            _raise_if_cancelled(cancel_event)
            return operation()
        except Exception as exc:
            if not _is_connection_refused_error(exc) or attempt >= retries:
                if _is_connection_refused_error(exc):
                    logger.error(
                        "translation.connection_refused_exhausted cause=9router_unreachable base_url=%s hint=%s model=%s source=%s target=%s batch_size=%s attempts=%s error=%s",
                        _nine_router_base_url(),
                        "No service is listening on the configured 9Router/OpenAI-compatible endpoint.",
                        model,
                        source,
                        target,
                        batch_size,
                        attempt + 1,
                        exc,
                    )
                raise

            delay = _connection_refused_retry_delay(attempt)
            logger.warning(
                "translation.connection_refused_retry cause=9router_unreachable base_url=%s hint=%s model=%s source=%s target=%s batch_size=%s attempt=%s retries=%s delay=%.2f error=%s",
                _nine_router_base_url(),
                "Auto-starting 9Router gateway...",
                model,
                source,
                target,
                batch_size,
                attempt + 1,
                retries,
                delay,
                exc,
            )
            try:
                from utils.ninerouter import ensure_9router_running
                ensure_9router_running(_nine_router_base_url())
            except Exception as auto_err:
                logger.warning("translation.autostart_9router_failed error=%s", auto_err)
            deadline = time.monotonic() + delay
            while True:
                _raise_if_cancelled(cancel_event)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.2, remaining))
    raise RuntimeError("unreachable translation retry state")


def _connection_refused_retry_delay(attempt: int) -> float:
    try:
        base_delay = float(_config_value("AUTODUB_TRANSLATION_CONNECTION_REFUSED_RETRY_DELAY") or 1.5)
    except (TypeError, ValueError):
        base_delay = 1.5
    return max(0.0, min(15.0, base_delay * (attempt + 1)))


def _is_connection_refused_error(exc: BaseException) -> bool:
    if isinstance(exc, ConnectionRefusedError):
        return True
    if isinstance(exc, URLError):
        return _is_connection_refused_reason(exc.reason)
    return _is_connection_refused_reason(getattr(exc, "__cause__", None))


def _is_connection_refused_reason(reason: object) -> bool:
    if isinstance(reason, ConnectionRefusedError):
        return True
    if isinstance(reason, OSError) and getattr(reason, "winerror", None) == 10061:
        return True
    return "WinError 10061" in str(reason)


def _config_value(name: str) -> str | None:
    dotenv_value = _dotenv_value(name)
    if dotenv_value:
        return dotenv_value
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _dotenv_value(name: str) -> str | None:
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return None
    try:
        with open(env_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != name:
                    continue
                clean_value = value.strip().strip('"').strip("'")
                return clean_value or None
    except OSError:
        return None
    return None


def _list_9router_models(base_url: str, api_key: str | None, timeout: float) -> list[str]:
    request = Request(
        _api_url(base_url, "models"),
        headers=_openai_compatible_headers(api_key or ""),
        method="GET",
    )
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
                model_ids.append(_gateway_model_id(model_id))

    filtered = [model for model in model_ids if _is_likely_translation_model(model)]
    return _dedupe_models(filtered or model_ids)


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for model in models:
        clean_model = model.strip()
        if clean_model and clean_model not in seen:
            seen.add(clean_model)
            deduped.append(clean_model)
    return deduped


def _is_likely_translation_model(model: str) -> bool:
    lowered = model.lower()
    if any(blocked in lowered for blocked in ("lyria", "image", "tts", "stt", "embed", "speech")):
        return False
    return any(
        keyword in lowered
        for keyword in (
            "gemini",
            "gemma",
            "gpt",
            "claude",
            "grok",
            "qwen",
            "deepseek",
            "mistral",
            "minimax",
            "kimi",
            "glm",
            "llama",
        )
    )


def _preferred_default_model(models: list[str]) -> str:
    for preferred in FALLBACK_TRANSLATION_MODELS:
        if preferred in models:
            return preferred
    for model in models:
        if model.startswith("gemini/"):
            return model
    return models[0]


def _gateway_model_id(model: str) -> str:
    clean_model = model.strip()
    for prefix in GATEWAY_MODEL_PREFIXES:
        if clean_model[: len(prefix)].lower() == prefix:
            clean_model = clean_model[len(prefix) :].strip()
            break
    return GATEWAY_MODEL_ALIASES.get(clean_model, clean_model)


def _translation_model_label(model: str) -> str:
    labels = {
        "ag/gemini-pro-agent": "Antigravity Gemini Pro Agent",
        "ag/gemini-3-flash-agent": "Antigravity Gemini 3 Flash Agent",
        "ag/gemini-3.1-pro-low": "Antigravity Gemini 3.1 Pro Low",
        "ag/gemini-3.5-flash-low": "Antigravity Gemini 3.5 Flash Low",
        "ag/gemini-3.5-flash-extra-low": "Antigravity Gemini 3.5 Flash Extra Low",
        "ag/gemini-3-flash": "Antigravity Gemini 3 Flash",
        "oc/deepseek-v4-flash-free": "OpenCode DeepSeek V4 Flash Free",
        "oc/mimo-v2.5-free": "OpenCode MiMo 2.5 Free",
        "openrouter/google/gemini-2.5-pro-exp-03-25:free": "Gemini 2.5 Pro Exp Free",
        "openrouter/google/gemini-2.0-flash-exp:free": "Gemini 2.0 Flash Exp Free",
        "openrouter/google/gemini-2.0-flash-thinking-exp:free": "Gemini 2.0 Flash Thinking Free",
        "openrouter/google/gemini-2.0-flash-lite-preview-02-05:free": "Gemini 2.0 Flash Lite Preview Free",
        "openrouter/google/gemma-4-26b-a4b-it:free": "Gemma 4 26B IT Free",
        "openrouter/google/gemma-4-31b-it:free": "Gemma 4 31B IT Free",
        "gemini/gemini-3-flash-preview": "Gemini 3 Flash Preview",
        "gemini/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite Preview",
        "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
        "gemini/gemma-4-31b-it": "Gemma 4 31B IT",
        "gemini/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite (direct quota)",
        "gemini/gemini-2.5-flash": "Gemini 2.5 Flash (direct quota)",
        "gemini/gemini-2.5-pro": "Gemini 2.5 Pro (direct quota)",
    }
    return labels.get(model, model)


def _nine_router_base_url() -> str:
    return (
        _config_value("AUTODUB_9ROUTER_BASE_URL")
        or _config_value("NINEROUTER_BASE_URL")
        or _config_value("9ROUTER_BASE_URL")
        or DEFAULT_9ROUTER_BASE_URL
    ).rstrip("/")


def _nine_router_api_key() -> str | None:
    key = (
        _config_value("AUTODUB_9ROUTER_API_KEY")
        or _config_value("NINEROUTER_API_KEY")
        or _config_value("9ROUTER_API_KEY")
        or _config_value("OPENAI_API_KEY")
    )
    if not key:
        return None
    clean_key = key.strip()
    if clean_key.lower() in {"replace-with-your-9router-token", "changeme", "your-api-key"}:
        return None
    if any(mask in clean_key for mask in ("•", "*", "...", "…")):
        return None
    return clean_key


def _selected_translation_model(model: str | None) -> str:
    return _gateway_model_id(
        model
        or _config_value("AUTODUB_TRANSLATION_MODEL")
        or DEFAULT_TRANSLATION_MODEL
    )


def _translation_model_attempts(primary_model: str, *, failed_models: set[str] | None = None) -> list[str]:
    configured_fallbacks = _config_value("AUTODUB_TRANSLATION_FALLBACK_MODELS") or ""
    fallback_models = (
        [item.strip() for item in configured_fallbacks.split(",") if item.strip()]
        if configured_fallbacks.strip()
        else DEFAULT_TRANSLATION_FALLBACK_MODELS
    )
    attempts = _dedupe_models([_gateway_model_id(model) for model in [primary_model, *fallback_models]])
    if not failed_models:
        return attempts

    skipped = {model for model in attempts if model in failed_models}
    if skipped:
        logger.info(
            "translation.model_attempts.skip_failed models=%s",
            ",".join(sorted(skipped)),
        )
    return [model for model in attempts if model not in failed_models]


def _remember_failed_translation_model(model: str, failed_models: set[str] | None) -> None:
    if failed_models is None:
        return
    normalized = _gateway_model_id(model)
    if normalized in failed_models:
        return
    failed_models.add(normalized)
    logger.warning("translation.model_marked_failed model=%s", normalized)


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _openai_compatible_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _normalize_language(language: str | None, *, fallback: str) -> str:
    if not language or language == "auto":
        return fallback
    normalized = language.strip().lower().replace("_", "-")
    aliases = {
        "zh-cn": "zh-CN",
        "zh-hans": "zh-CN",
        "zh-tw": "zh-TW",
        "zh-hant": "zh-TW",
    }
    return aliases.get(normalized, normalized)


def _language_name(language: str) -> str:
    names: dict[str, str] = {
        "auto": "auto",
        "en": "English",
        "vi": "Vietnamese",
        "zh": "Chinese",
        "zh-CN": "Simplified Chinese",
        "zh-TW": "Traditional Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
    }
    return names.get(language, language)






