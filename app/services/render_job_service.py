from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from dotenv import load_dotenv

from app.models.schemas import DubbingScriptSegment, FlashTextTrack, PipelineConfig, RenderScriptRequest
from app.services.pipeline import AutoDubbingPipeline
from app.services.short_video_pipeline import ShortVideoPipeline
from app.utils.workspace import Workspace

logger = logging.getLogger("auto_dubbing.render_jobs")
load_dotenv()

RENDER_JOB_VERSION = "render-job-v1"
DEFAULT_STALE_SECONDS = 300.0
_LOCAL_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autodub-render")
_LOCAL_SCHEDULED: set[tuple[str, int]] = set()
_LOCAL_SCHEDULED_LOCK = threading.Lock()
_JOB_LOCKS: dict[str, threading.RLock] = {}
_JOB_LOCKS_GUARD = threading.Lock()


def _job_lock(job_id: str) -> threading.RLock:
    with _JOB_LOCKS_GUARD:
        return _JOB_LOCKS.setdefault(job_id, threading.RLock())


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_sse_event(event: str) -> dict[str, Any] | None:
    for line in event.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class RenderJobSubmission:
    job_id: str
    attempt: int
    enqueued: bool


class RenderJobStore:
    """Durable render metadata and artifacts; Redis is coordination, not storage."""

    def __init__(self, root: str | Path | None = None, output_root: str | Path = "output") -> None:
        configured_root = root or os.environ.get("AUTODUB_RENDER_JOB_ROOT", "render_jobs")
        self.root = Path(configured_root)
        self.output_root = Path(output_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        payload: RenderScriptRequest,
        config: PipelineConfig,
        source_video_path: Path,
        clone_reference_path: Path | None,
    ) -> str:
        source_video_path = Path(source_video_path)
        if not _nonempty_file(source_video_path):
            raise FileNotFoundError(f"Source video not found: {source_video_path}")

        self.evict_stale_jobs()
        source_hash = _sha256_file(source_video_path)
        clone_hash = _sha256_file(clone_reference_path) if clone_reference_path else ""
        canonical_payload = payload.model_dump(mode="json")
        canonical_payload.pop("source_video_path", None)
        canonical_payload.pop("clone_reference_audio_path", None)
        fingerprint_payload = {
            "version": RENDER_JOB_VERSION,
            "source_sha256": source_hash,
            "clone_reference_sha256": clone_hash,
            "request": canonical_payload,
        }
        fingerprint_json = json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        job_id = hashlib.sha256(fingerprint_json.encode("utf-8")).hexdigest()[:32]

        with _job_lock(job_id):
            job_dir = self.job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            source_blob = self.artifact_root() / "sources" / f"{source_hash}.mp4"
            if not _nonempty_file(source_blob):
                _atomic_copy(source_video_path, source_blob)
            os.utime(source_blob, None)
            source_destination = job_dir / "source.mp4"
            if not _nonempty_file(source_destination):
                _atomic_link_or_copy(source_blob, source_destination)

            clone_destination: Path | None = None
            if clone_reference_path is not None:
                suffix = clone_reference_path.suffix.lower() or ".wav"
                clone_blob = self.artifact_root() / "voices" / f"{clone_hash}{suffix}"
                if not _nonempty_file(clone_blob):
                    _atomic_copy(clone_reference_path, clone_blob)
                os.utime(clone_blob, None)
                clone_destination = job_dir / f"clone_reference{suffix}"
                if not _nonempty_file(clone_destination):
                    _atomic_link_or_copy(clone_blob, clone_destination)

            job_config = config.model_copy(
                update={
                    "clone_reference_audio_path": str(clone_destination) if clone_destination else None,
                }
            )
            request_document = {
                "version": RENDER_JOB_VERSION,
                "job_id": job_id,
                "fingerprint": fingerprint_payload,
                "config": job_config.model_dump(mode="json"),
                "segments": [segment.model_dump(mode="json") for segment in payload.segments],
                "flash_text_tracks": [track.model_dump(mode="json") for track in payload.flash_text_tracks],
            }
            self._atomic_write_json(self.request_path(job_id), request_document)

            if not self.manifest_path(job_id).is_file():
                now = time.time()
                self._atomic_write_json(
                    self.manifest_path(job_id),
                    {
                        "job_id": job_id,
                        "status": "pending",
                        "phase": "prepare",
                        "progress": 0,
                        "attempt": 0,
                        "completed_groups": 0,
                        "created_at": now,
                        "updated_at": now,
                        "source_sha256": source_hash,
                        "clone_reference_sha256": clone_hash,
                        "output_video_path": str(self.output_path(job_id)),
                    },
                )

        logger.info(
            "render_job.prepared job_id=%s source_sha256=%s clone_sha256=%s segments=%d",
            job_id,
            source_hash[:16],
            clone_hash[:16] if clone_hash else "none",
            len(payload.segments),
        )
        return job_id

    def begin_attempt(self, job_id: str) -> tuple[int, bool]:
        with _job_lock(job_id):
            manifest = self.read_manifest(job_id)
            output_path = self.output_path(job_id)
            if manifest.get("status") == "complete" and _nonempty_file(output_path):
                return int(manifest.get("attempt", 1)), False

            status = str(manifest.get("status", "pending"))
            updated_at = float(manifest.get("updated_at", 0.0) or 0.0)
            stale_seconds = float(os.environ.get("AUTODUB_JOB_STALE_SECONDS", DEFAULT_STALE_SECONDS))
            queue_backend = os.environ.get("AUTODUB_QUEUE_BACKEND", "local").strip().lower()
            owner_pid = int(manifest.get("owner_pid", 0) or 0)
            if status in {"queued", "running"}:
                if queue_backend == "local" and _pid_is_running(owner_pid):
                    return int(manifest.get("attempt", 1)), False
                if queue_backend != "local" and time.time() - updated_at < stale_seconds:
                    return int(manifest.get("attempt", 1)), False

            attempt = int(manifest.get("attempt", 0)) + 1
            events_path = self.events_path(job_id, attempt)
            events_path.parent.mkdir(parents=True, exist_ok=True)
            events_path.write_text("", encoding="utf-8")
            manifest.update(
                {
                    "status": "queued",
                    "phase": "queue",
                    "progress": 1,
                    "attempt": attempt,
                    "owner_pid": os.getpid() if queue_backend == "local" else 0,
                    "error": "",
                    "updated_at": time.time(),
                }
            )
            self._atomic_write_json(self.manifest_path(job_id), manifest)
            self.append_event(
                job_id,
                attempt,
                {
                    "step": "processing",
                    "status": "processing",
                    "phase": "queue",
                    "progress": 1,
                    "message": "Render queued",
                    "job_id": job_id,
                },
            )
            return attempt, True

    def mark_running(self, job_id: str, attempt: int) -> None:
        with _job_lock(job_id):
            manifest = self.read_manifest(job_id)
            if int(manifest.get("attempt", attempt)) == attempt:
                manifest["worker_pid"] = os.getpid()
                manifest["updated_at"] = time.time()
                self._atomic_write_json(self.manifest_path(job_id), manifest)
        self.append_event(
            job_id,
            attempt,
            {
                "step": "processing",
                "status": "processing",
                "phase": "prepare",
                "progress": 3,
                "message": "Render worker started",
                "job_id": job_id,
            },
        )

    def mark_failed(self, job_id: str, attempt: int, error: str) -> None:
        self.append_event(
            job_id,
            attempt,
            {
                "step": "error",
                "status": "error",
                "phase": "error",
                "progress": 0,
                "error": error,
                "message": "Render failed",
                "job_id": job_id,
            },
        )

    def append_event(self, job_id: str, attempt: int, payload: dict[str, Any]) -> None:
        event = dict(payload)
        event.setdefault("job_id", job_id)
        event["attempt"] = attempt
        with _job_lock(job_id):
            with self.events_path(job_id, attempt).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            manifest = self.read_manifest(job_id)
            if int(manifest.get("attempt", attempt)) != attempt:
                return
            status = str(event.get("status", "processing"))
            manifest["status"] = (
                "complete"
                if status == "success"
                else ("failed" if status == "error" else ("cancelled" if status == "cancelled" else "running"))
            )
            manifest["phase"] = str(event.get("phase", manifest.get("phase", "prepare")))
            manifest["progress"] = int(event.get("progress", manifest.get("progress", 0)) or 0)
            manifest["updated_at"] = time.time()
            stats = event.get("stats")
            if isinstance(stats, dict):
                chunks = stats.get("chunks")
                groups = stats.get("groups")
                if isinstance(chunks, int):
                    manifest["completed_groups"] = max(int(manifest.get("completed_groups", 0)), chunks)
                if isinstance(groups, int):
                    manifest["total_groups"] = groups
            if status == "error":
                manifest["error"] = str(event.get("error") or event.get("message") or "Render failed")
            if status == "success":
                manifest["output_video_url"] = event.get("video_url")
                manifest["output_subtitle_url"] = event.get("subtitle_url")
            self._atomic_write_json(self.manifest_path(job_id), manifest)

    def mark_cancelled(self, job_id: str, reason: str = "Cancelled by user") -> bool:
        with _job_lock(job_id):
            manifest = self.read_manifest(job_id)
            if manifest.get("status") in {"complete", "failed", "cancelled"}:
                return False
            attempt = int(manifest.get("attempt", 0) or 0)
            if attempt <= 0:
                manifest["status"] = "cancelled"
                manifest["phase"] = "cancelled"
                manifest["error"] = reason
                manifest["updated_at"] = time.time()
                self._atomic_write_json(self.manifest_path(job_id), manifest)
                return True
            self.append_event(
                job_id,
                attempt,
                {
                    "step": "cancelled",
                    "status": "cancelled",
                    "phase": "cancelled",
                    "progress": int(manifest.get("progress", 0) or 0),
                    "error": reason,
                    "message": "Render cancelled by user",
                    "job_id": job_id,
                },
            )
            return True

    def attach_task(self, job_id: str, attempt: int, task_id: str) -> None:
        with _job_lock(job_id):
            manifest = self.read_manifest(job_id)
            if int(manifest.get("attempt", attempt)) != attempt:
                return
            manifest["task_id"] = task_id
            manifest["updated_at"] = time.time()
            self._atomic_write_json(self.manifest_path(job_id), manifest)

    def acquire_execution(self, job_id: str, attempt: int) -> str | None:
        lock_path = self.job_dir(job_id) / "execution.lock"
        token = uuid4().hex
        payload = json.dumps(
            {"token": token, "pid": os.getpid(), "attempt": attempt, "created_at": time.time()},
            separators=(",", ":"),
        ).encode("utf-8")
        for _ in range(2):
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    existing = json.loads(lock_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                owner_pid = int(existing.get("pid", 0) or 0)
                owner_attempt = int(existing.get("attempt", 0) or 0)
                if owner_attempt == attempt and _pid_is_running(owner_pid):
                    return None
                lock_path.unlink(missing_ok=True)
                continue
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return token
        return None

    def release_execution(self, job_id: str, token: str) -> None:
        lock_path = self.job_dir(job_id) / "execution.lock"
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if existing.get("token") == token:
            lock_path.unlink(missing_ok=True)

    def read_manifest(self, job_id: str) -> dict[str, Any]:
        with _job_lock(job_id):
            try:
                return json.loads(self.manifest_path(job_id).read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise KeyError(f"Unknown render job: {job_id}") from None

    def read_events(self, job_id: str, attempt: int) -> list[dict[str, Any]]:
        path = self.events_path(job_id, attempt)
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events

    def load_request(self, job_id: str) -> tuple[PipelineConfig, list[DubbingScriptSegment], list[FlashTextTrack]]:
        document = json.loads(self.request_path(job_id).read_text(encoding="utf-8"))
        config = PipelineConfig.model_validate(document["config"])
        segments = [DubbingScriptSegment.model_validate(item) for item in document["segments"]]
        tracks = [FlashTextTrack.model_validate(item) for item in document.get("flash_text_tracks", [])]
        return config, segments, tracks

    def workspace(self, job_id: str) -> Workspace:
        root = self.job_dir(job_id) / "workspace"
        chunks_dir = root / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        return Workspace(
            request_id=job_id,
            root=root,
            input_video=self.source_path(job_id),
            chunks_dir=chunks_dir,
            output_dir=self.output_root,
        )

    def evict_stale_jobs(self) -> None:
        max_age_days = float(os.environ.get("AUTODUB_RENDER_JOB_MAX_AGE_DAYS", "7"))
        max_age_seconds = max(max_age_days, 1.0) * 24 * 60 * 60
        cutoff = time.time() - max_age_seconds
        for entry in self.root.iterdir():
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            manifest_path = entry / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                updated_at = float(manifest.get("updated_at", 0.0) or 0.0)
            except Exception:
                continue
            if manifest.get("status") in {"complete", "failed"} and updated_at < cutoff:
                shutil.rmtree(entry, ignore_errors=True)

        artifact_root = self.artifact_root()
        if not artifact_root.is_dir():
            return
        for artifact in artifact_root.rglob("*"):
            if not artifact.is_file():
                continue
            try:
                stat = artifact.stat()
                if stat.st_nlink <= 1 and stat.st_mtime < cutoff:
                    artifact.unlink(missing_ok=True)
            except OSError:
                continue

    def artifact_root(self) -> Path:
        path = self.root / "_artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def source_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "source.mp4"

    def request_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "request.json"

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def events_path(self, job_id: str, attempt: int) -> Path:
        return self.job_dir(job_id) / f"events-{attempt}.jsonl"

    def output_path(self, job_id: str) -> Path:
        return self.output_root / f"{job_id}_script_dubbed.mp4"

    @staticmethod
    def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            for retry in range(8):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if retry == 7:
                        raise
                    time.sleep(min(0.01 * (2**retry), 0.2))
        finally:
            temporary.unlink(missing_ok=True)


class RenderJobRunner:
    def __init__(self, store: RenderJobStore | None = None) -> None:
        self.store = store or RenderJobStore()

    def run(
        self,
        job_id: str,
        attempt: int,
        state_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        execution_token = self.store.acquire_execution(job_id, attempt)
        if execution_token is None:
            logger.info("render_job.duplicate_suppressed job_id=%s attempt=%d", job_id, attempt)
            return {"job_id": job_id, "status": "already_running", "attempt": attempt}
        try:
            manifest = self.store.read_manifest(job_id)
            if int(manifest.get("attempt", -1)) != attempt:
                return {"job_id": job_id, "status": "superseded", "attempt": attempt}
            if manifest.get("status") == "complete" and _nonempty_file(self.store.output_path(job_id)):
                return {"job_id": job_id, "status": "complete", "attempt": attempt}

            self.store.mark_running(job_id, attempt)
            config, segments, flash_text_tracks = self.store.load_request(job_id)
            workspace = self.store.workspace(job_id)
            terminal_payload: dict[str, Any] | None = None
            try:
                pipeline_cls = ShortVideoPipeline if config.short_video else AutoDubbingPipeline
                pipeline = pipeline_cls(config)

                def persist_event(payload: dict[str, Any]) -> None:
                    self.store.append_event(job_id, attempt, payload)
                    if state_callback is not None:
                        state_callback(payload)

                set_runtime_event_callback = getattr(pipeline, "set_runtime_event_callback", None)
                if callable(set_runtime_event_callback):
                    set_runtime_event_callback(persist_event)
                render_kwargs = {
                    "workspace": workspace,
                    "source_video_path": self.store.source_path(job_id),
                    "script_segments": segments,
                }
                # Keep the worker compatible with lightweight pipeline doubles
                # and older custom pipeline implementations.  The real
                # AutoDubbingPipeline receives the optional track list when it
                # is present; an empty list has no rendering effect.
                if flash_text_tracks:
                    render_kwargs["flash_text_tracks"] = flash_text_tracks
                for sse_event in pipeline.render_script(**render_kwargs):
                    payload = _parse_sse_event(sse_event)
                    if payload is None:
                        continue
                    persist_event(payload)
                    if payload.get("status") in {"success", "error"}:
                        terminal_payload = payload

                if terminal_payload is None:
                    raise RuntimeError("Render worker exited without a terminal event.")
                if terminal_payload.get("status") == "success" and not _nonempty_file(self.store.output_path(job_id)):
                    raise RuntimeError("Render reported success but output video is missing.")
                return {
                    "job_id": job_id,
                    "status": "complete" if terminal_payload.get("status") == "success" else "failed",
                    "attempt": attempt,
                }
            except Exception as exc:
                logger.exception("render_job.failed job_id=%s attempt=%d", job_id, attempt)
                self.store.mark_failed(job_id, attempt, str(exc))
                raise
        finally:
            self.store.release_execution(job_id, execution_token)


class RenderJobDispatcher:
    def __init__(self, store: RenderJobStore | None = None) -> None:
        self.store = store or RenderJobStore()

    def submit(
        self,
        payload: RenderScriptRequest,
        config: PipelineConfig,
        source_video_path: Path,
        clone_reference_path: Path | None,
    ) -> RenderJobSubmission:
        job_id = self.store.prepare(payload, config, source_video_path, clone_reference_path)
        return self.resume(job_id)

    def resume(self, job_id: str) -> RenderJobSubmission:
        attempt, should_enqueue = self.store.begin_attempt(job_id)
        if not should_enqueue:
            return RenderJobSubmission(job_id=job_id, attempt=attempt, enqueued=False)

        self._enqueue(job_id, attempt)
        return RenderJobSubmission(job_id=job_id, attempt=attempt, enqueued=True)

    def cancel(self, job_id: str, reason: str = "Cancelled by user") -> dict[str, Any]:
        manifest = self.store.read_manifest(job_id)
        self.store.mark_cancelled(job_id, reason)
        task_id = str(manifest.get("task_id", "") or "")
        backend = os.environ.get("AUTODUB_QUEUE_BACKEND", "local").strip().lower()
        if backend == "celery" and task_id:
            try:
                from app.celery_app import celery_app

                celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
            except Exception:
                logger.exception("render_job.cancel_revoke_failed job_id=%s task_id=%s", job_id, task_id)
        return manifest

    def _enqueue(self, job_id: str, attempt: int) -> None:
        backend = os.environ.get("AUTODUB_QUEUE_BACKEND", "local").strip().lower()
        try:
            if backend == "celery":
                from app.celery_app import celery_app

                result = celery_app.send_task(
                    "app.tasks.run_render_job",
                    args=[job_id, attempt],
                    queue="render",
                )
                self.store.attach_task(job_id, attempt, result.id)
            elif backend == "local":
                self._submit_local(job_id, attempt)
            else:
                raise RuntimeError(f"Unsupported AUTODUB_QUEUE_BACKEND: {backend}")
        except Exception as exc:
            self.store.mark_failed(job_id, attempt, f"Unable to enqueue render job: {exc}")
            raise
    def _submit_local(self, job_id: str, attempt: int) -> None:
        key = (job_id, attempt)
        with _LOCAL_SCHEDULED_LOCK:
            if key in _LOCAL_SCHEDULED:
                return
            _LOCAL_SCHEDULED.add(key)

        future = _LOCAL_EXECUTOR.submit(RenderJobRunner(self.store).run, job_id, attempt)

        def release(_future) -> None:
            with _LOCAL_SCHEDULED_LOCK:
                _LOCAL_SCHEDULED.discard(key)

        future.add_done_callback(release)
