from __future__ import annotations

"""Bounded producer/consumer orchestration for block-based dubbing.

The queue owns ordering and failure propagation. Stage functions are kept
outside this module so ASR, translation and TTS can use their existing model
and timing policies. A bounded queue is intentional: a fast ASR stage must
not accumulate an unbounded amount of translated text/audio in memory.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from app.utils.cancel import PipelineCancelledError

logger = logging.getLogger(__name__)

QUEUE_SIZE = 2
_STOP = object()


@dataclass(frozen=True)
class StreamBlock:
    sequence: int
    items: list[Any]


@dataclass(frozen=True)
class StreamingPipelineResult:
    translated: list[Any]
    audio_chunks: list[Any]


class StreamingPipeline:
    """Run ASR -> translation -> TTS with bounded worker queues.

    Each stage receives a complete contiguous block. Results are collected by
    sequence number, so completion order never changes subtitle/audio order.
    The stage callbacks must be cancellation-aware at model/API boundaries.
    """

    def __init__(
        self,
        *,
        asr_blocks: Callable[[], Iterable[list[Any]]],
        translate_block: Callable[[list[Any]], list[Any]],
        tts_block: Callable[[list[Any]], list[Any]],
        cancel_event: threading.Event | None = None,
        queue_size: int = QUEUE_SIZE,
        translation_group_size: int = 1,
    ) -> None:
        self.asr_blocks = asr_blocks
        self.translate_block = translate_block
        self.tts_block = tts_block
        self.cancel_event = cancel_event
        self.queue_size = max(1, int(queue_size))
        self.translation_group_size = max(1, int(translation_group_size))
        self._stop_event = threading.Event()
        self._errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)

    def run(self) -> StreamingPipelineResult:
        asr_queue: queue.Queue[StreamBlock | object] = queue.Queue(maxsize=self.queue_size)
        tts_queue: queue.Queue[StreamBlock | object] = queue.Queue(maxsize=self.queue_size)
        translated_by_sequence: dict[int, list[Any]] = {}
        audio_by_sequence: dict[int, list[Any]] = {}
        result_lock = threading.Lock()

        def check_cancelled() -> None:
            if self._stop_event.is_set():
                raise RuntimeError("streaming pipeline stopped")
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise PipelineCancelledError("Pipeline cancelled")

        def put_with_stop(target: queue.Queue[StreamBlock | object], item: StreamBlock | object) -> bool:
            while not self._stop_event.is_set():
                try:
                    target.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    check_cancelled()
            return False

        def fail(exc: BaseException) -> None:
            if not self._errors.full():
                try:
                    self._errors.put_nowait(exc)
                except queue.Full:
                    pass
            self._stop_event.set()

        def produce() -> None:
            try:
                for sequence, items in enumerate(self.asr_blocks()):
                    check_cancelled()
                    if not items:
                        continue
                    if not put_with_stop(asr_queue, StreamBlock(sequence, list(items))):
                        return
                put_with_stop(asr_queue, _STOP)
            except BaseException as exc:
                fail(exc)

        def translate() -> None:
            pending: list[StreamBlock] = []

            def flush_pending() -> None:
                if not pending:
                    return
                flat_items = [item for block in pending for item in block.items]
                translated = self.translate_block(flat_items)
                if len(translated) != len(flat_items):
                    raise ValueError(
                        f"Translation window returned {len(translated)} "
                        f"items for {len(flat_items)} inputs"
                    )
                offset = 0
                blocks = list(pending)
                pending.clear()
                for block in blocks:
                    size = len(block.items)
                    translated_block = list(translated[offset : offset + size])
                    offset += size
                    with result_lock:
                        translated_by_sequence[block.sequence] = translated_block
                    if not put_with_stop(tts_queue, StreamBlock(block.sequence, translated_block)):
                        return

            try:
                while not self._stop_event.is_set():
                    check_cancelled()
                    try:
                        item = asr_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if item is _STOP:
                        flush_pending()
                        put_with_stop(tts_queue, _STOP)
                        return
                    assert isinstance(item, StreamBlock)
                    pending.append(item)
                    if len(pending) >= self.translation_group_size:
                        flush_pending()
            except BaseException as exc:
                fail(exc)

        def synthesize() -> None:
            try:
                while not self._stop_event.is_set():
                    check_cancelled()
                    try:
                        item = tts_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if item is _STOP:
                        return
                    assert isinstance(item, StreamBlock)
                    chunks = self.tts_block(item.items)
                    with result_lock:
                        audio_by_sequence[item.sequence] = list(chunks)
            except BaseException as exc:
                fail(exc)

        workers = [
            threading.Thread(target=produce, name="dubbing-asr-producer"),
            threading.Thread(target=translate, name="dubbing-translation-consumer"),
            threading.Thread(target=synthesize, name="dubbing-tts-consumer"),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        if not self._errors.empty():
            error = self._errors.get_nowait()
            if isinstance(error, PipelineCancelledError):
                raise error
            logger.error(
                "streaming_pipeline.failed error=%s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            raise RuntimeError(f"Streaming pipeline failed: {error}") from error
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise PipelineCancelledError("Pipeline cancelled")

        sequences = sorted(translated_by_sequence)
        if sequences != sorted(audio_by_sequence):
            raise RuntimeError(
                "Streaming pipeline lost a block: "
                f"translated={sequences} audio={sorted(audio_by_sequence)}"
            )
        translated = [item for sequence in sequences for item in translated_by_sequence[sequence]]
        audio_chunks = [item for sequence in sequences for item in audio_by_sequence[sequence]]
        return StreamingPipelineResult(translated=translated, audio_chunks=audio_chunks)
