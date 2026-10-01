from __future__ import annotations

import logging
import os
import signal
import threading
from dataclasses import dataclass
from typing import Any


class PipelineCancelledError(RuntimeError):
    """Raised when a client disconnects or explicitly cancels processing."""


@dataclass
class ActiveOperation:
    operation_id: str
    cancel_event: threading.Event
    kind: str
    pid: int


class ActiveOperationRegistry:
    """Tracks stream workers so the UI can cancel the exact operation it started."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._operations: dict[str, ActiveOperation] = {}

    def register(self, operation_id: str, cancel_event: threading.Event, kind: str) -> None:
        with self._lock:
            self._operations[operation_id] = ActiveOperation(
                operation_id=operation_id,
                cancel_event=cancel_event,
                kind=kind,
                pid=os.getpid(),
            )

    def unregister(self, operation_id: str) -> None:
        with self._lock:
            self._operations.pop(operation_id, None)

    def cancel(self, operation_id: str) -> ActiveOperation | None:
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is not None:
                operation.cancel_event.set()
            return operation

    def cancel_only_active(self) -> ActiveOperation | None:
        with self._lock:
            if len(self._operations) != 1:
                return None
            operation = next(iter(self._operations.values()))
            operation.cancel_event.set()
            return operation

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"operation_id": item.operation_id, "kind": item.kind, "pid": item.pid}
                for item in self._operations.values()
            ]


ACTIVE_OPERATIONS = ActiveOperationRegistry()


def schedule_process_termination(reason: str, delay_seconds: float = 0.35, pid: int | None = None) -> None:
    """Terminate one backend process after the cancel response has a chance to flush."""

    target_pid = int(pid or os.getpid())

    def terminate() -> None:
        try:
            logging.getLogger("auto_dubbing.cancel").critical(
                "hard_cancel.terminate pid=%s reason=%s", target_pid, reason
            )
            os.kill(target_pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return

    timer = threading.Timer(max(0.05, delay_seconds), terminate)
    timer.daemon = True
    timer.start()
