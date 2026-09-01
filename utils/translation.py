from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
TRANSLATION_PROVIDERS = {"9router", "ninerouter", "openai-compatible", "openai_compatible"}
DEFAULT_TRANSLATION_BATCH_ITEMS = 24
DEFAULT_TRANSLATION_BATCH_WORDS = 750
DEFAULT_TRANSLATION_BATCH_CHARS = 12000
DEFAULT_TRANSLATION_CONCURRENCY = 4
DEFAULT_TRANSLATION_ATTEMPT_TIMEOUT_SECONDS = 18.0
TRANSLATION_CONTEXT_WINDOW = 4
TRANSLATION_CONTEXT_MAX_CHARS = 2400
TRANSLATION_POLICY_VERSION = "batch-context-coherence-qa-v2"
DEFAULT_COHERENCE_BATCH_ITEMS = 24
DEFAULT_COHERENCE_TIMEOUT_SECONDS = 18.0
TRANSLATION_SCENE_BREAK_SECONDS = 8.0
REPEATED_SOUND_MIN_TOKENS = 6
VI_REPEATED_SOUND_TRANSLATIONS = {
    "흥": "Hừm",
    "哼": "Hừm",
    "嗯": "Ừm",
    "啊": "À",
    "哦": "Ồ",
    "哈": "Ha",
}
VI_INCOMPLETE_LINE_ENDINGS = frozenset(
    {
        "đang",
        "đã",
        "sẽ",
        "vừa",
        "mới",
        "chưa",
        "được",
        "bị",
        "là",
        "và",
        "hay",
        "hoặc",
        "mà",
        "để",
        "với",
        "của",
        "cho",
        "từ",
        "vì",
        "nếu",
        "khi",
        "nên",
        "thì",
    }
)
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

    repeated_sound = _deterministic_repeated_sound_translation(clean_text, target)
    if repeated_sound:
        logger.info(
            "translation.repetition_short_circuit source=%s target=%s source_text=%s replacement=%s",
            source,
            target,
            clean_text,
            repeated_sound,
        )
        return repeated_sound

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
            fallback = _translate_google_gtx_cached(clean_text, source, target, timeout)
            return _fallback_if_bad_translation(
                clean_text,
                fallback,
                source=source,
                target=target,
                timeout=timeout,
            )
        return clean_text


def translate_segments(
    texts: list[str],
    *,
    target_language: str,
    target_durations: list[float] | None = None,
    source_intervals: list[tuple[float, float]] | None = None,
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

    if selected_provider not in TRANSLATION_PROVIDERS:
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

    intervals = _normalize_intervals(source_intervals, len(clean_texts))
    batches = _translation_batches(clean_texts, max_items=batch_size)
    specs: list[tuple[int, list[str], tuple[float, ...], tuple[tuple[float, float], ...], str]] = []
    batch_offset = 0
    for batch_index, batch in enumerate(batches):
        size = len(batch)
        specs.append(
            (
                batch_index,
                batch,
                tuple(clean_durations[batch_offset : batch_offset + size]),
                tuple(intervals[batch_offset : batch_offset + size]),
                _build_batch_continuity_context(
                    clean_texts,
                    batch_start=batch_offset,
                    batch_end=batch_offset + size,
                    base_context=context or "",
                ),
            )
        )
        batch_offset += size

    logger.info(
        "translation.batch_plan provider=%s segments=%s batches=%s max_items=%s concurrency=%s",
        selected_provider,
        len(clean_texts),
        len(specs),
        batch_size or DEFAULT_TRANSLATION_BATCH_ITEMS,
        _translation_concurrency(),
    )
    translated_by_batch = _translate_batches_concurrently(
        specs,
        source=source,
        target=target,
        selected_model=selected_model,
        base_url=_nine_router_base_url(),
        api_key=_nine_router_api_key() or "",
        timeout=timeout,
        cancel_event=cancel_event,
        context=context or "",
        cps_budget=cps_budget or 12.5,
    )
    translated = [text for batch_index in range(len(specs)) for text in translated_by_batch[batch_index]]

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

    normalized = [
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
    return _repair_translation_boundaries(normalized, target=target)


def review_translation_coherence(
    texts: list[str],
    *,
    target_durations: list[float],
    source_intervals: list[tuple[float, float]] | None = None,
    source_texts: list[str] | None = None,
    target_language: str = "vi",
    provider: str | None = None,
    model: str | None = None,
    context: str | None = None,
    batch_size: int | None = None,
    timeout: float = 45.0,
    cancel_event: Event | None = None,
) -> list[str]:
    """Review contiguous translation batches for fidelity and narrative flow.

    This is deliberately a batch-level pass. It may repair omitted predicates,
    bad references, or broken review cadence, but it never changes the number
    or order of timeline cues.
    """
    if not texts:
        return []
    if len(target_durations) != len(texts):
        raise ValueError("target_durations must match texts")

    clean_texts = [text.strip() for text in texts]
    clean_sources = [text.strip() for text in (source_texts or texts)]
    if len(clean_sources) != len(clean_texts):
        raise ValueError("source_texts must match texts")

    selected_provider = (provider or _config_value("AUTODUB_TRANSLATION_PROVIDER") or "9router").strip().lower()
    if selected_provider not in TRANSLATION_PROVIDERS:
        return clean_texts

    target = _normalize_language(target_language, fallback="vi")
    selected_model = _selected_translation_model(model)
    intervals = _normalize_intervals(source_intervals, len(clean_texts))
    requested_batch_size = batch_size or DEFAULT_TRANSLATION_BATCH_ITEMS
    coherence_batch_size = min(
        max(1, int(requested_batch_size)),
        _env_int(
            "AUTODUB_TRANSLATION_COHERENCE_BATCH_SIZE",
            DEFAULT_COHERENCE_BATCH_ITEMS,
            minimum=8,
            maximum=40,
        ),
    )
    # Keep a hard upper bound on the number of QA waves. Long silences are
    # marked in each item below instead of turning every silence into another
    # batch, which could otherwise reintroduce a multi-minute timeout chain.
    batches = _translation_batches(clean_sources, max_items=coherence_batch_size)
    specs: list[
        tuple[
            int,
            int,
            int,
            tuple[str, ...],
            tuple[str, ...],
            tuple[float, ...],
            tuple[tuple[float, float], ...],
            str,
        ]
    ] = []
    offset = 0
    for batch_index, batch in enumerate(batches):
        size = len(batch)
        specs.append(
            (
                batch_index,
                offset,
                offset + size,
                tuple(clean_texts[offset : offset + size]),
                tuple(clean_sources[offset : offset + size]),
                tuple(max(0.1, float(value)) for value in target_durations[offset : offset + size]),
                tuple(intervals[offset : offset + size]),
                _build_adjacent_translation_context(
                    clean_sources,
                    clean_texts,
                    batch_start=offset,
                    batch_end=offset + size,
                    base_context=context or "",
                    intervals=intervals,
                ),
            )
        )
        offset += size

    logger.info(
        "translation.coherence.batch_plan segments=%s batches=%s max_items=%s concurrency=%s",
        len(clean_texts),
        len(specs),
        coherence_batch_size,
        _translation_concurrency(),
    )

    def run_one(spec):
        (
            batch_index,
            _batch_start,
            _batch_end,
            batch_texts,
            batch_sources,
            batch_durations,
            batch_intervals,
            batch_context,
        ) = spec
        _raise_if_cancelled(cancel_event)
        try:
            reviewed = _review_9router_segments_with_model_fallback(
                batch_texts,
                source_texts=batch_sources,
                target_durations=batch_durations,
                source_intervals=batch_intervals,
                target=target,
                selected_model=selected_model,
                base_url=_nine_router_base_url(),
                api_key=_nine_router_api_key() or "",
                timeout=_coherence_attempt_timeout(timeout),
                context=batch_context,
                cancel_event=cancel_event,
            )
            if len(reviewed) != len(batch_texts):
                raise ValueError(f"Expected {len(batch_texts)} coherence results, got {len(reviewed)}")
            values = []
            for candidate, original, source, duration in zip(
                reviewed,
                batch_texts,
                batch_sources,
                batch_durations,
            ):
                clean_candidate = _clean_translated_text(candidate)
                if _coherence_candidate_is_unsafe(
                    clean_candidate,
                    original=original,
                    source_text=source,
                    target=target,
                    target_duration=duration,
                ):
                    values.append(original)
                else:
                    values.append(clean_candidate or original)
            logger.info("translation.coherence.batch_done batch=%s segments=%s", batch_index + 1, len(batch_texts))
            return batch_index, values
        except Exception as exc:
            logger.warning(
                "translation.coherence.batch_failed batch=%s segments=%s fallback=original error=%s",
                batch_index + 1,
                len(batch_texts),
                exc,
            )
            return batch_index, list(batch_texts)

    results: dict[int, list[str]] = {}
    concurrency = min(_translation_concurrency(), len(specs))
    if concurrency <= 1:
        for spec in specs:
            batch_index, values = run_one(spec)
            results[batch_index] = values
    else:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="translation-coherence") as executor:
            futures = [executor.submit(run_one, spec) for spec in specs]
            for future in as_completed(futures):
                batch_index, values = future.result()
                results[batch_index] = values

    return [value for batch_index in range(len(specs)) for value in results[batch_index]]


def _translate_batches_concurrently(
    specs: list[tuple[int, list[str], tuple[float, ...], tuple[tuple[float, float], ...], str]],
    *,
    source: str,
    target: str,
    selected_model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    cancel_event: Event | None,
    context: str,
    cps_budget: float,
) -> dict[int, list[str]]:
    """Translate independent contiguous batches concurrently and restore order."""
    if not specs:
        return {}
    results: dict[int, list[str]] = {}

    def run_one(spec: tuple[int, list[str], tuple[float, ...], tuple[tuple[float, float], ...], str]) -> tuple[int, list[str]]:
        batch_index, batch, durations, intervals, batch_context = spec
        _raise_if_cancelled(cancel_event)
        # Each request owns its failed-model state. Sharing a mutable set across
        # workers makes model fallback order nondeterministic and can cause a
        # healthy model to be skipped by a different batch.
        failed_models: set[str] = set()
        try:
            translated = _translate_9router_segments_with_model_fallback(
                tuple(batch),
                target_durations=durations,
                source_intervals=intervals,
                source=source,
                target=target,
                selected_model=selected_model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                cancel_event=cancel_event,
                failed_models=failed_models,
                context=batch_context,
                cps_budget=cps_budget,
            )
            if len(translated) != len(batch):
                raise ValueError(f"Expected {len(batch)} translations, got {len(translated)}")
            logger.info("translation.batch_done batch=%s segments=%s", batch_index + 1, len(batch))
            return batch_index, translated
        except Exception as exc:
            logger.warning(
                "translation.batch_failed batch=%s segments=%s error=%s",
                batch_index + 1,
                len(batch),
                exc,
            )
            # Preserve the bounded request strategy even when a provider
            # rejects a large JSON response. The recursive helper has already
            # attempted smaller batches; this is the final per-line safety net.
            fallback = [
                translate_text(
                    text,
                    source_language=source,
                    target_language=target,
                    provider="9router",
                    model=selected_model,
                    timeout=timeout,
                    cancel_event=cancel_event,
                    failed_models=failed_models,
                )
                if text
                else text
                for text in batch
            ]
            return batch_index, fallback

    concurrency = min(_translation_concurrency(), len(specs))
    if concurrency <= 1:
        for spec in specs:
            batch_index, translated = run_one(spec)
            results[batch_index] = translated
        return results

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="translation") as executor:
        futures = [executor.submit(run_one, spec) for spec in specs]
        for future in as_completed(futures):
            batch_index, translated = future.result()
            results[batch_index] = translated
    return results


def _normalize_intervals(
    intervals: list[tuple[float, float]] | None,
    expected_count: int,
) -> list[tuple[float, float]]:
    if intervals is None:
        return [(0.0, 0.0) for _ in range(expected_count)]
    if len(intervals) != expected_count:
        raise ValueError("source_intervals must match texts")
    normalized: list[tuple[float, float]] = []
    previous_end = 0.0
    for start, end in intervals:
        clean_start = max(0.0, float(start))
        clean_end = max(clean_start, float(end))
        if clean_start < previous_end:
            logger.warning(
                "translation.timeline_overlap start=%.3f previous_end=%.3f",
                clean_start,
                previous_end,
            )
        if clean_start - previous_end > 8.0 and previous_end > 0:
            logger.warning(
                "translation.timeline_long_gap gap=%.3f start=%.3f previous_end=%.3f",
                clean_start - previous_end,
                clean_start,
                previous_end,
            )
        normalized.append((clean_start, clean_end))
        previous_end = clean_end
    return normalized


def _repair_translation_boundaries(texts: list[str], *, target: str) -> list[str]:
    """Remove only obvious cross-cue duplication without rewriting meaning."""
    repaired: list[str] = []
    for text in texts:
        clean = _clean_translated_text(text)
        if not repaired or not clean:
            repaired.append(clean)
            continue
        previous = repaired[-1]
        previous_words = re.findall(r"[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?", previous.lower(), flags=re.UNICODE)
        current_words = re.findall(r"[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?", clean.lower(), flags=re.UNICODE)
        if _contains_cjk(previous) or _contains_cjk(clean):
            # CJK output has no whitespace tokenization. Remove only an exact
            # short suffix/prefix overlap; this catches a model repeating the
            # last clause at a batch boundary without touching normal shared
            # characters in unrelated sentences.
            duplicate_chars = 0
            for width in (8, 6, 4, 3, 2):
                if len(previous) >= width and len(clean) > width and previous[-width:] == clean[:width]:
                    duplicate_chars = width
                    break
            if duplicate_chars:
                clean = clean[duplicate_chars:].lstrip(" \t,，。.!！?？;；:")
                logger.warning(
                    "translation.boundary_duplicate_removed chars=%s target=%s",
                    duplicate_chars,
                    target,
                )
        elif previous_words and current_words:
            duplicate_count = 0
            for width in (3, 2, 1):
                if len(previous_words) >= width and len(current_words) >= width:
                    if previous_words[-width:] == current_words[:width]:
                        duplicate_count = width
                        break
            if duplicate_count:
                tokens = clean.split()
                if duplicate_count < len(tokens):
                    clean = " ".join(tokens[duplicate_count:]).strip()
                    logger.warning("translation.boundary_duplicate_removed words=%s target=%s", duplicate_count, target)
        repaired.append(clean or text.strip())
    return repaired


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
            return (
                _enforce_shortened_text(
                    shortened,
                    clean_text,
                    word_limit,
                    source_text=source_text or "",
                    target=target,
                ),
                word_limit,
                selected_provider,
                selected_model,
            )
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
    source_intervals: list[tuple[float, float]] | None = None,
    target_language: str = "vi",
    source_texts: list[str] | None = None,
    context: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_words: list[int] | None = None,
    timeout: float = 45.0,
    batch_size: int = DEFAULT_TRANSLATION_BATCH_ITEMS,
    cancel_event: Event | None = None,
) -> list[str]:
    """Review timing-constrained lines in bounded batches, never one request per line."""
    if not texts:
        return []
    if len(target_durations) != len(texts):
        raise ValueError("target_durations must match texts")
    intervals = _normalize_intervals(source_intervals, len(texts))
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
    if selected_provider not in TRANSLATION_PROVIDERS:
        return [_shorten_heuristic(text, limit) for text, limit in zip(clean_texts, limits)]

    batches = _translation_batches(clean_texts, max_items=batch_size)
    logger.info(
        "shorten.batch_plan segments=%s batches=%s max_items=%s concurrency=%s",
        len(clean_texts),
        len(batches),
        batch_size,
        _translation_concurrency(),
    )
    specs: list[tuple[int, list[str], tuple[str, ...], tuple[float, ...], tuple[int, ...], tuple[tuple[float, float], ...]]] = []
    offset = 0
    for batch_index, batch in enumerate(batches):
        size = len(batch)
        specs.append(
            (
                batch_index,
                batch,
                tuple(clean_sources[offset : offset + size]),
                tuple(max(0.1, float(value)) for value in target_durations[offset : offset + size]),
                tuple(limits[offset : offset + size]),
                tuple(intervals[offset : offset + size]),
            )
        )
        offset += size

    def run_one(spec: tuple[int, list[str], tuple[str, ...], tuple[float, ...], tuple[int, ...], tuple[tuple[float, float], ...]]) -> tuple[int, list[str]]:
        batch_index, batch, batch_sources, batch_durations, batch_limits, batch_intervals = spec
        _raise_if_cancelled(cancel_event)
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
                batch_intervals,
            )
            values = [
                _enforce_shortened_text(
                    value,
                    original,
                    limit,
                    source_text=source,
                    target=target,
                )
                for value, original, limit, source in zip(shortened, batch, batch_limits, batch_sources)
            ]
            logger.info("shorten.batch_done batch=%s segments=%s", batch_index + 1, len(batch))
            return batch_index, values
        except Exception as exc:
            logger.warning(
                "shorten.batch_failed provider=%s model=%s target=%s batch_size=%s fallback=heuristic error=%s",
                selected_provider,
                selected_model,
                target,
                len(batch),
                exc,
            )
            return batch_index, [
                _enforce_shortened_text(
                    _shorten_heuristic(original, limit),
                    original,
                    limit,
                    source_text=source,
                    target=target,
                )
                for original, limit, source in zip(batch, batch_limits, batch_sources)
            ]

    shortened_by_batch: dict[int, list[str]] = {}
    concurrency = min(_translation_concurrency(), len(specs))
    if concurrency <= 1:
        for spec in specs:
            batch_index, values = run_one(spec)
            shortened_by_batch[batch_index] = values
    else:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="translation-review") as executor:
            futures = [executor.submit(run_one, spec) for spec in specs]
            for future in as_completed(futures):
                batch_index, values = future.result()
                shortened_by_batch[batch_index] = values
    return [value for batch_index in range(len(specs)) for value in shortened_by_batch[batch_index]]


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
    source_intervals: tuple[tuple[float, float], ...] = (),
) -> list[str]:
    # Do not send ASR hallucination loops to the gateway. Besides producing
    # useless output, a large batch containing these lines can consume the
    # whole request timeout before the per-line normalizer gets a chance to
    # collapse them.
    repeated_replacements = [
        _deterministic_repeated_sound_translation(text, target)
        for text in texts
    ]
    if any(repeated_replacements):
        active_indexes = [index for index, replacement in enumerate(repeated_replacements) if not replacement]
        if not active_indexes:
            logger.info(
                "translation.batch_repetition_short_circuit segments=%s skipped=%s",
                len(texts),
                len(texts),
            )
            return repeated_replacements

        logger.info(
            "translation.batch_repetition_filter segments=%s skipped=%s remaining=%s",
            len(texts),
            len(texts) - len(active_indexes),
            len(active_indexes),
        )
        active_texts = tuple(texts[index] for index in active_indexes)
        active_durations = tuple(target_durations[index] for index in active_indexes)
        active_intervals = (
            tuple(source_intervals[index] for index in active_indexes)
            if source_intervals
            else ()
        )
        active_translations = _translate_9router_segments_with_model_fallback(
            active_texts,
            target_durations=active_durations,
            source_intervals=active_intervals,
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
        )
        if len(active_translations) != len(active_indexes):
            raise ValueError(
                f"Expected {len(active_indexes)} active translations, got {len(active_translations)}"
            )
        merged = list(repeated_replacements)
        for index, translated in zip(active_indexes, active_translations):
            merged[index] = translated
        return merged

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
                    source_intervals,
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
                source_intervals=source_intervals[:midpoint],
                source=source,
                target=target,
                selected_model=selected_model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                cancel_event=cancel_event,
                # A timeout for a large payload does not mean the model is
                # unhealthy. Retry each smaller payload with a fresh model
                # budget instead of immediately reporting "no models left".
                failed_models=set(),
                context=context,
                cps_budget=cps_budget,
            ),
            *_translate_9router_segments_with_model_fallback(
                texts[midpoint:],
                target_durations=target_durations[midpoint:],
                source_intervals=source_intervals[midpoint:],
                source=source,
                target=target,
                selected_model=selected_model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                cancel_event=cancel_event,
                failed_models=set(),
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
                    "You are a professional dubbing script editor for natural spoken Vietnamese. "
                    "Shorten translated dialogue only as much as needed for a natural speaking pace, "
                    "while preserving the complete proposition, tone, names, numbers, and intent. "
                    "Keep the line vivid when the source is vivid, but never summarize away the meaning "
                    "or remove the predicate that makes the sentence complete. "
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
                        "reading_speed_rule": (
                            "Treat absolute_max_words as a soft pacing target. Never trade away a "
                            "complete meaning, verb, question, negation, or emotional beat just to hit it."
                        ),
                        "source_text": source_text,
                        "current_translation": text,
                        "context": context,
                        "rules": [
                            "Keep the result natural for dubbing; vividness must come from the source, not invention.",
                            "Preserve the main action, speaker attitude, names, numbers, money amounts, and negation.",
                            "Drop filler words, particles, duplicate phrasing, and nonessential politeness first.",
                            "Do not remove the hook, emotional direction, or key judgment if it is present in the source.",
                            "A complete short sentence is always better than an incomplete fragment.",
                            "Preserve question and exclamation force and never end a Vietnamese line on a dangling word such as đang, và, mà, để, với, or của.",
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
    source_intervals: tuple[tuple[float, float], ...] = (),
) -> list[str]:
    if source_intervals and len(source_intervals) != len(texts):
        raise ValueError("source_intervals must match texts")
    intervals = source_intervals or tuple((0.0, 0.0) for _ in texts)
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
                            "Treat max_spoken_words as a soft pacing target, not permission to delete the predicate or emotional beat.",
                            "Preserve the complete proposition and key meaning; remove filler and repetition first.",
                            "Use natural complete spoken phrasing; never cut a word or end on a dangling auxiliary, conjunction, or preposition.",
                            "Preserve question marks, exclamation force, negation, aspect, and speaker attitude.",
                        ],
                        "segments": [
                            {
                                "id": index,
                                "start_seconds": round(intervals[index][0], 3),
                                "end_seconds": round(intervals[index][1], 3),
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


def _review_9router_segments_with_model_fallback(
    texts: tuple[str, ...],
    *,
    source_texts: tuple[str, ...],
    target_durations: tuple[float, ...],
    source_intervals: tuple[tuple[float, float], ...],
    target: str,
    selected_model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    context: str,
    cancel_event: Event | None = None,
) -> list[str]:
    last_error: Exception | None = None
    _raise_if_cancelled(cancel_event)
    # Coherence QA is an optional polish pass. The primary translation has
    # already succeeded, so do not multiply latency with model fallbacks or
    # recursive per-line retries when the gateway is slow.
    for attempt_model in (selected_model,):
        try:
            return _review_9router_segments_cached(
                texts,
                source_texts,
                target_durations,
                target,
                attempt_model,
                base_url,
                api_key,
                timeout,
                context,
                source_intervals,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "translation.coherence.model_attempt_failed model=%s target=%s batch_size=%s error=%s",
                attempt_model,
                target,
                len(texts),
                exc,
            )

    if last_error is not None:
        raise last_error
    return list(texts)


@lru_cache(maxsize=256)
def _review_9router_segments_cached(
    texts: tuple[str, ...],
    source_texts: tuple[str, ...],
    target_durations: tuple[float, ...],
    target: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    context: str = "",
    source_intervals: tuple[tuple[float, float], ...] = (),
) -> list[str]:
    if len(source_texts) != len(texts):
        raise ValueError("source_texts must match texts")
    if len(target_durations) != len(texts):
        raise ValueError("target_durations must match texts")
    if source_intervals and len(source_intervals) != len(texts):
        raise ValueError("source_intervals must match texts")
    intervals = source_intervals or tuple((0.0, 0.0) for _ in texts)
    payload = {
        "model": model,
        "temperature": 0.15,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the final continuity editor for a Vietnamese film-review dubbing script. "
                    "The numbered items form one contiguous narrative batch. Read the whole batch, "
                    "then correct each current translation against its exact source so the result has "
                    "one consistent expert-review voice, clear cause-and-effect, stable references, "
                    "and natural spoken rhythm. Return only valid JSON. Never merge, split, skip, or "
                    "reorder ids."
                    + (f" Continuity context (do not duplicate it): {context}" if context else "")
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": _language_name(target),
                        "response_schema": {
                            "segments": [
                                {"id": 0, "text": "reviewed segment 0"},
                                {"id": 1, "text": "reviewed segment 1"},
                            ]
                        },
                        "rules": [
                            "Return exactly one object for every input id and no extra ids.",
                            "Keep each output aligned to its own timestamp; never move meaning to a neighboring id.",
                            "Preserve every source fact, action, subject, object, name, number, tense, negation, and emotional turn.",
                            "Repair omitted meaning, mistranslation, dangling clauses, abrupt pronoun changes, and broken causal links.",
                            "Maintain a coherent professional movie-review narrator voice across the batch: vivid and insightful when supported by the source, never melodramatic or invented.",
                            "Use natural spoken Vietnamese with purposeful cadence. Do not translate each line as an isolated label.",
                            "For a complete source thought, return a complete sentence. Do not leave a line ending in an auxiliary, conjunction, or preposition.",
                            "For a source segment lasting six seconds or longer, do not collapse a complete thought into a short label; express all supported meaning naturally without padding with new facts.",
                            "Treat timing metadata as guidance only. Never cut meaning to satisfy a word or character budget; TTS handles final rate fitting.",
                            "A silence gap over eight seconds is a scene break; do not invent a causal link across it.",
                            "If the current translation is already accurate and coherent, keep it unchanged.",
                        ],
                        "segments": [
                            {
                                "id": index,
                                "start_seconds": round(intervals[index][0], 3),
                                "end_seconds": round(intervals[index][1], 3),
                                "duration_seconds": round(target_durations[index], 3),
                                "gap_from_previous_seconds": round(
                                    max(0.0, intervals[index][0] - intervals[index - 1][1])
                                    if index > 0
                                    else 0.0,
                                    3,
                                ),
                                "source_text": source_texts[index],
                                "current_translation": text,
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
        raise RuntimeError(f"9Router coherence review failed with HTTP {exc.code}: {detail}") from exc

    content = _parse_chat_completion_body(body, content_type).strip()
    reviewed = _parse_translation_response(content, expected_count=len(texts))
    if len(reviewed) != len(texts):
        raise ValueError(f"Expected {len(texts)} coherence results, got {len(reviewed)}")
    logger.info(
        "translation.coherence.done provider=9router_batch model=%s target=%s segments=%s",
        model,
        target,
        len(texts),
    )
    return reviewed


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
    source_intervals: tuple[tuple[float, float], ...] = (),
) -> list[str]:
    if source_intervals and len(source_intervals) != len(texts):
        raise ValueError("source_intervals must match texts")
    intervals = source_intervals or tuple((0.0, 0.0) for _ in texts)
    payload = {
        "model": model,
        "temperature": 0.0,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior audiovisual translator and Vietnamese dubbing writer for subtitle "
                    "timelines. Translate each numbered segment faithfully while using neighboring "
                    "segments only as context. Use natural spoken Vietnamese with the same emotional "
                    "temperature as the source: vivid and cinematic when the source supports it, plain "
                    "and conversational when it is plain. Preserve the timing-friendly brevity of short "
                    "lines without dropping important meaning. Infer omitted subjects "
                    "conservatively; do not add new facts, jokes, moralizing, plot details, or "
                    "expert opinions not implied by the source. A complete faithful sentence takes "
                    "priority over the timing hint; TTS may adjust speaking rate later. Return only valid JSON, with no "
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
                            "Write complete spoken sentences; never end on a dangling auxiliary, conjunction, or preposition.",
                            "Treat max_spoken_words and max_characters as soft pacing hints. Never cut the predicate, question, negation, or emotional beat to satisfy them.",
                            "Keep the translation anchored to this segment's source text; continuity context can resolve references but must not replace source meaning.",
                            "A silence gap over eight seconds is a scene break; do not invent a causal link across it.",
                            *_translation_rules(target),
                        ],
                        "segments": [
                            {
                                "id": index,
                                "text": text,
                                "start_seconds": round(intervals[index][0], 3),
                                "end_seconds": round(intervals[index][1], 3),
                                "gap_from_previous_seconds": round(
                                    max(0.0, intervals[index][0] - intervals[index - 1][1])
                                    if index > 0
                                    else 0.0,
                                    3,
                                ),
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
                "Give the line life through cadence and emotional register only when the source supports it; do not make a plain line artificially dramatic.",
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


def _translation_batches(
    texts: list[str],
    *,
    max_items: int | None = None,
    intervals: list[tuple[float, float]] | None = None,
) -> list[list[str]]:
    if intervals is not None and len(intervals) != len(texts):
        raise ValueError("intervals must match texts")
    max_items = max_items or _env_int(
        "AUTODUB_TRANSLATION_BATCH_SIZE",
        DEFAULT_TRANSLATION_BATCH_ITEMS,
        minimum=1,
        maximum=100,
    )
    max_items = max(1, min(100, int(max_items)))
    max_words = _env_int(
        "AUTODUB_TRANSLATION_BATCH_WORDS",
        DEFAULT_TRANSLATION_BATCH_WORDS,
        minimum=100,
        maximum=2000,
    )
    max_chars = _env_int(
        "AUTODUB_TRANSLATION_BATCH_CHARS",
        DEFAULT_TRANSLATION_BATCH_CHARS,
        minimum=1200,
        maximum=30000,
    )
    batches: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    current_chars = 0
    previous_end = 0.0
    for index, text in enumerate(texts):
        text_words = _count_words(text)
        projected_words = current_words + text_words
        projected_chars = current_chars + len(text)
        scene_break = bool(
            intervals
            and index > 0
            and intervals[index][0] - previous_end > TRANSLATION_SCENE_BREAK_SECONDS
        )
        if current and (
            scene_break
            or len(current) >= max_items
            or projected_words > max_words
            or projected_chars > max_chars
        ):
            batches.append(current)
            current = []
            current_words = 0
            current_chars = 0
        current.append(text)
        current_words += text_words
        current_chars += len(text)
        if intervals:
            previous_end = intervals[index][1]
    if current:
        batches.append(current)
    return batches


def _build_batch_continuity_context(
    texts: list[str],
    *,
    batch_start: int,
    batch_end: int,
    base_context: str = "",
) -> str:
    """Give each concurrent batch a small source-only window across its edge."""
    before = [text.strip() for text in texts[max(0, batch_start - TRANSLATION_CONTEXT_WINDOW):batch_start] if text.strip()]
    after = [text.strip() for text in texts[batch_end:batch_end + TRANSLATION_CONTEXT_WINDOW] if text.strip()]
    parts = [base_context.strip()] if base_context.strip() else []
    if before:
        parts.append(
            "Previous source context (read for continuity; do not translate or repeat as new ids): "
            + json.dumps(before, ensure_ascii=False)
        )
    if after:
        parts.append(
            "Following source context (read for continuity; do not translate or repeat as new ids): "
            + json.dumps(after, ensure_ascii=False)
        )
    return "\n".join(parts)[:TRANSLATION_CONTEXT_MAX_CHARS]


def _build_adjacent_translation_context(
    source_texts: list[str],
    translated_texts: list[str],
    *,
    batch_start: int,
    batch_end: int,
    base_context: str = "",
    intervals: list[tuple[float, float]] | None = None,
) -> str:
    """Build bounded before/after context for the final continuity QA pass."""
    before_start = max(0, batch_start - TRANSLATION_CONTEXT_WINDOW)
    after_end = min(len(source_texts), batch_end + TRANSLATION_CONTEXT_WINDOW)
    before = [
        {"source": source_texts[index], "translation": translated_texts[index]}
        for index in range(before_start, batch_start)
    ]
    after = [
        {"source": source_texts[index], "translation": translated_texts[index]}
        for index in range(batch_end, after_end)
    ]
    parts = [base_context.strip()] if base_context.strip() else []
    if intervals and batch_start > 0:
        gap_before = intervals[batch_start][0] - intervals[batch_start - 1][1]
        if gap_before > TRANSLATION_SCENE_BREAK_SECONDS:
            parts.append(
                f"Scene break before this batch: silence_gap_seconds={gap_before:.3f}. "
                "Do not force a causal link across this gap."
            )
    if intervals and batch_end < len(intervals):
        gap_after = intervals[batch_end][0] - intervals[batch_end - 1][1]
        if gap_after > TRANSLATION_SCENE_BREAK_SECONDS:
            parts.append(
                f"Scene break after this batch: silence_gap_seconds={gap_after:.3f}. "
                "Do not force a causal link across this gap."
            )
    if before:
        parts.append(
            "Previous translated context (read for continuity; do not duplicate): "
            + json.dumps(before, ensure_ascii=False)
        )
    if after:
        parts.append(
            "Following translated context (read for continuity; do not duplicate): "
            + json.dumps(after, ensure_ascii=False)
        )
    return "\n".join(parts)[:TRANSLATION_CONTEXT_MAX_CHARS]


def _translation_concurrency() -> int:
    return _env_int(
        "AUTODUB_TRANSLATION_CONCURRENCY",
        DEFAULT_TRANSLATION_CONCURRENCY,
        minimum=1,
        maximum=8,
    )


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

    repeated_sound = _deterministic_repeated_sound_translation(source_text, target)
    if repeated_sound:
        if clean_translation != repeated_sound:
            logger.warning(
                "translation.repetition_collapsed source=%s target=%s source_text=%s translated_text=%s replacement=%s",
                source,
                target,
                source_text,
                clean_translation,
                repeated_sound,
            )
        return repeated_sound

    if _is_repetition_loop(clean_translation):
        collapsed = _collapse_repetition_loop(clean_translation)
        if collapsed:
            logger.warning(
                "translation.repetition_collapsed source=%s target=%s source_text=%s translated_text=%s replacement=%s",
                source,
                target,
                source_text,
                clean_translation,
                collapsed,
            )
            return collapsed

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
        repeated_sound = _deterministic_repeated_sound_translation(source_text, target)
        if repeated_sound:
            return repeated_sound
        if _is_repetition_loop(fallback):
            collapsed = _collapse_repetition_loop(fallback)
            if collapsed:
                logger.warning(
                    "translation.repetition_collapsed source=%s target=%s source_text=%s translated_text=%s replacement=%s",
                    source,
                    target,
                    source_text,
                    fallback,
                    collapsed,
                )
                return collapsed
        if fallback and not _translation_needs_fallback(source_text, fallback, target):
            return fallback
        if fallback:
            logger.warning(
                "translation.google_fallback_rejected source=%s target=%s source_text=%s fallback=%s",
                source,
                target,
                source_text,
                fallback,
            )
        return clean_translation or source_text
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
    if _is_repetition_loop(translated_text):
        return True
    if _looks_fragmentary_translation(
        translated_text,
        source_text=source_text,
        original_text=source_text,
        target=target,
    ):
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


def _looks_fragmentary_translation(
    candidate: str,
    *,
    source_text: str = "",
    original_text: str = "",
    target: str = "vi",
) -> bool:
    """Reject timing edits that destroy a complete sentence's syntax or force."""
    clean = _clean_shortened_text(candidate)
    if not clean:
        return True

    if clean[-1] in ",;:，；：":
        return True

    source = source_text.strip()
    original = original_text.strip()
    source_or_original = source or original
    source_is_question = bool(re.search(r"[?？]\s*$", source_or_original))
    source_is_exclamation = bool(re.search(r"[!！]\s*$", source_or_original))
    candidate_has_question = bool(re.search(r"[?？]\s*$", clean))
    candidate_has_exclamation = bool(re.search(r"[!！]\s*$", clean))

    # A model frequently drops the final punctuation while shortening. For a
    # question/exclamation that changes the delivery, so keep the full line.
    if source_is_question and not candidate_has_question:
        return True
    if source_is_exclamation and not candidate_has_exclamation:
        return True

    original_is_complete = bool(re.search(r"[.!?…。！？]\s*$", original))
    if target.startswith("vi") and (source_is_question or source_is_exclamation or original_is_complete):
        words = re.findall(r"[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?", clean.lower(), flags=re.UNICODE)
        if words and words[-1] in VI_INCOMPLETE_LINE_ENDINGS:
            return True

    return False


def _coherence_candidate_is_unsafe(
    candidate: str,
    *,
    original: str,
    source_text: str,
    target: str,
    target_duration: float,
) -> bool:
    """Keep a QA pass from silently replacing a meaningful line with less text."""
    if not candidate:
        return True
    if target.startswith("vi") and _contains_cjk(candidate):
        return True
    if _looks_fragmentary_translation(
        candidate,
        source_text=source_text,
        original_text=original,
        target=target,
    ):
        return True

    original_words = _count_words(original)
    candidate_words = _count_words(candidate)
    if target_duration >= 6.0 and original_words >= 8:
        # A continuity editor may polish, but should not halve a long cue into
        # a label. Keep the existing line when its replacement is suspiciously
        # sparse; the next TTS stage can still fit the original safely.
        minimum_words = max(4, int(original_words * 0.45))
        if candidate_words < minimum_words:
            return True
    return False


def _enforce_shortened_text(
    candidate: str,
    original: str,
    max_words: int,
    *,
    source_text: str = "",
    target: str = "vi",
) -> str:
    clean = _clean_shortened_text(candidate)
    if clean and _count_words(clean) <= max_words:
        bounded = clean
    elif clean:
        bounded = _shorten_heuristic(clean, max_words)
    else:
        bounded = _shorten_heuristic(original, max_words)

    if _looks_fragmentary_translation(
        bounded,
        source_text=source_text,
        original_text=original,
        target=target,
    ):
        logger.warning(
            "translation.shorten_rejected_fragment candidate=%s original=%s",
            bounded,
            original,
        )
        return original.strip()
    return bounded or original.strip()


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

    # Keep this short rhetorical question complete even when a timing budget
    # asks the model to shorten it.
    if re.fullmatch(r"你不是在做事[吗嗎][?？]?", source_text.strip()):
        return "Chẳng phải cậu đang làm việc sao?"

    normalized_source = _normalize_for_compare(source_text)
    override = VI_ZH_SHORT_TRANSLATION_OVERRIDES.get(normalized_source)
    if override:
        return override

    if "块钱" in source_text and "đô la" in translated_text.lower():
        translated_text = re.sub("đô la", "tệ", translated_text, flags=re.IGNORECASE)
    return translated_text


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]", text))


def _repetition_tokens(text: str) -> list[str]:
    """Tokenize words and CJK syllables for detecting ASR repetition loops."""
    return re.findall(
        r"[A-Za-zÀ-ỹĐđ0-9]+(?:['’][A-Za-zÀ-ỹĐđ0-9]+)?|"
        r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]",
        text,
        flags=re.UNICODE,
    )


def _is_repetition_loop(text: str) -> bool:
    tokens = _repetition_tokens(text)
    if len(tokens) < REPEATED_SOUND_MIN_TOKENS:
        return False
    normalized = [token.casefold() for token in tokens]
    counts = {token: normalized.count(token) for token in set(normalized)}
    most_common = max(counts.values(), default=0)
    return most_common / len(normalized) >= 0.8 and len(counts) <= 2


def _collapse_repetition_loop(text: str) -> str:
    tokens = _repetition_tokens(text)
    if not _is_repetition_loop(text) or not tokens:
        return ""
    replacement = tokens[0]
    terminal = re.search(r"([.!?！？。…]+)[\"'”’)]*\s*$", text)
    if terminal:
        replacement += terminal.group(1)
    return replacement


def _deterministic_repeated_sound_translation(text: str, target: str) -> str:
    """Collapse known non-lexical ASR loops before spending an API request."""
    if not target.startswith("vi") or not _is_repetition_loop(text):
        return ""
    tokens = _repetition_tokens(text)
    if not tokens:
        return ""
    return VI_REPEATED_SOUND_TRANSLATIONS.get(tokens[0].casefold(), "")


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
        configured = float(
            _config_value("AUTODUB_TRANSLATION_ATTEMPT_TIMEOUT")
            or DEFAULT_TRANSLATION_ATTEMPT_TIMEOUT_SECONDS
        )
    except (TypeError, ValueError):
        configured = DEFAULT_TRANSLATION_ATTEMPT_TIMEOUT_SECONDS
    return max(5.0, min(float(timeout), configured))


def _coherence_attempt_timeout(timeout: float) -> float:
    try:
        configured = float(
            _config_value("AUTODUB_TRANSLATION_COHERENCE_TIMEOUT")
            or DEFAULT_COHERENCE_TIMEOUT_SECONDS
        )
    except (TypeError, ValueError):
        configured = DEFAULT_COHERENCE_TIMEOUT_SECONDS
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
