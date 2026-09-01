from __future__ import annotations

import json
import asyncio
import threading
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.models.schemas import DubbingScriptSegment, PipelineConfig
from app.services.pipeline import AutoDubbingPipeline
from app.services.render_job_service import RenderJobDispatcher, RenderJobStore, RenderJobSubmission
from app.utils.workspace import WorkspaceManager


def _render_payload(source_path: str = "/media/retry_source.mp4") -> dict[str, object]:
    return {
        "source_video_path": source_path,
        "target_language": "vi",
        "translation_provider": "9router",
        "translation_model": "ag/gemini-3-flash-agent",
        "voice_model": "Truc Ly",
        "voice_mode": "system",
        "tts_device": "cuda",
        "copyright_confirmed": True,
        "copyright_source": "owned",
        "segments": [
            {
                "id": 7,
                "start": 1.25,
                "end": 2.75,
                "original_text": "source text",
                "translated_text": "edited script",
                "voice_model": "Truc Ly",
            }
        ],
    }


class _RenderOnlyPipeline:
    render_calls: list[dict[str, object]] = []

    def __init__(self, config: PipelineConfig, cancel_event=None) -> None:
        self.config = config
        self.cancel_event = cancel_event

    def analyze(self, *args, **kwargs):
        raise AssertionError("Render retry must not analyze the source again")

    def analyze_stream(self, *args, **kwargs):
        raise AssertionError("Render retry must not split the script again")

    def render_script(self, workspace, source_video_path, script_segments):
        type(self).render_calls.append(
            {
                "request_id": workspace.request_id,
                "source_bytes": source_video_path.read_bytes(),
                "segments": [segment.model_dump() for segment in script_segments],
            }
        )
        output_path = workspace.output_dir / f"{workspace.request_id}_script_dubbed.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered-video")
        yield "data: " + json.dumps(
            {
                "status": "success",
                "phase": "complete",
                "progress": 100,
                "video_url": f"/media/{workspace.request_id}_script_dubbed.mp4",
            }
        ) + "\n\n"


class RenderScriptRetryRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        _RenderOnlyPipeline.render_calls.clear()

    def test_upload_retry_reuses_same_durable_job_and_never_analyzes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = FastAPI()
            app.include_router(routes.stream_router)
            client = TestClient(app)
            manager = WorkspaceManager(root / "workspaces", root / "output")
            dispatcher = RenderJobDispatcher(RenderJobStore(root / "jobs", root / "output"))
            payload = _render_payload()

            with (
                patch.object(routes, "SOURCE_MEDIA_DIR", root / "source-cache"),
                patch.object(routes, "WorkspaceManager", return_value=manager),
                patch.object(routes, "RenderJobDispatcher", return_value=dispatcher),
                patch("app.services.render_job_service.AutoDubbingPipeline", _RenderOnlyPipeline),
            ):
                responses = [
                    client.post(
                        "/api/render-script-upload",
                        data={"payload": json.dumps(payload)},
                        files={"video": ("source.mp4", b"original-video", "video/mp4")},
                    )
                    for _ in range(2)
                ]

            self.assertEqual([response.status_code for response in responses], [200, 200])
            self.assertTrue(all('"status": "success"' in response.text for response in responses))
            self.assertEqual(len(_RenderOnlyPipeline.render_calls), 1)
            expected_segments = [
                DubbingScriptSegment.model_validate(segment).model_dump()
                for segment in payload["segments"]
            ]
            call = _RenderOnlyPipeline.render_calls[0]
            self.assertEqual(call["source_bytes"], b"original-video")
            self.assertEqual(call["segments"], expected_segments)
            self.assertEqual((root / "source-cache" / "retry_source.mp4").read_bytes(), b"original-video")

    def test_upload_retry_rejects_empty_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = FastAPI()
            app.include_router(routes.stream_router)
            client = TestClient(app)
            manager = WorkspaceManager(root / "workspaces", root / "output")

            with (
                patch.object(routes, "SOURCE_MEDIA_DIR", root / "source-cache"),
                patch.object(routes, "WorkspaceManager", return_value=manager),
                patch.object(routes, "AutoDubbingPipeline", _RenderOnlyPipeline),
            ):
                response = client.post(
                    "/api/render-script-upload",
                    data={"payload": json.dumps(_render_payload())},
                    files={"video": ("source.mp4", b"", "video/mp4")},
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["detail"], "Source video upload is empty.")
            self.assertEqual(_RenderOnlyPipeline.render_calls, [])

    def test_missing_output_uses_persistent_source_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "source-cache"
            cache_root.mkdir()
            cached_source = cache_root / "cache_only_source.mp4"
            cached_source.write_bytes(b"cached-video")

            with patch.object(routes, "SOURCE_MEDIA_DIR", cache_root):
                resolved = routes._output_media_path("/media/cache_only_source.mp4")

            self.assertEqual(resolved, cached_source.resolve())


class RenderJobStatusRecoveryTests(unittest.TestCase):
    def test_status_poll_requeues_failed_job_within_retry_limit(self) -> None:
        app = FastAPI()
        app.include_router(routes.stream_router)
        client = TestClient(app)
        job_id = "8822a6e7-1b5b-4871-a3ea-51e878aa3e65"

        class Store:
            manifest = {"job_id": job_id, "status": "failed", "attempt": 1, "progress": 0}

            def read_manifest(self, _job_id):
                return dict(self.manifest)

            def output_path(self, _job_id):
                return Path("missing-output.mp4")

        class Dispatcher:
            store = Store()
            resume_calls = 0

            def resume(self, _job_id):
                self.resume_calls += 1
                self.store.manifest.update({"status": "queued", "phase": "queue", "attempt": 2})
                return RenderJobSubmission(job_id=_job_id, attempt=2, enqueued=True)

        dispatcher = Dispatcher()
        with patch.object(routes, "RenderJobDispatcher", return_value=dispatcher):
            response = client.get(f"/api/render-jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "queued")
        self.assertEqual(response.json()["attempt"], 2)
        self.assertEqual(dispatcher.resume_calls, 1)

    def test_status_poll_stops_retrying_at_limit(self) -> None:
        app = FastAPI()
        app.include_router(routes.stream_router)
        client = TestClient(app)
        job_id = "8822a6e7-1b5b-4871-a3ea-51e878aa3e65"

        class Store:
            def read_manifest(self, _job_id):
                return {"job_id": job_id, "status": "failed", "attempt": 3, "progress": 0}

            def output_path(self, _job_id):
                return Path("missing-output.mp4")

        dispatcher = type("Dispatcher", (), {"store": Store(), "resume": lambda *_: self.fail("must not retry")})()
        with patch.object(routes, "RenderJobDispatcher", return_value=dispatcher):
            response = client.get(f"/api/render-jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "failed")


class RenderScriptSourceStagingTests(unittest.TestCase):
    def test_source_is_staged_before_long_render_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = WorkspaceManager(root / "workspaces", root / "output")
            workspace = manager.create()
            source = root / "source.mp4"
            source.write_bytes(b"stable-source")
            segment = DubbingScriptSegment(
                id=1,
                start=0,
                end=1,
                original_text="source",
                translated_text="edited",
                voice_model="Truc Ly",
            )
            pipeline = AutoDubbingPipeline(
                PipelineConfig(copyright_confirmed=True, copyright_source="owned")
            )

            pipeline.dependencies.require_ffmpeg = lambda: None

            def probe_staged(staged_path: Path) -> float:
                self.assertEqual(staged_path.read_bytes(), b"stable-source")
                source.unlink()
                return 1.0

            def no_tts(*args, **kwargs):
                if False:
                    yield ""
                return []

            def render_from_staged(*, video_path: Path, output_path: Path, **kwargs) -> None:
                self.assertEqual(video_path, workspace.root / "render_source.mp4")
                self.assertEqual(video_path.read_bytes(), b"stable-source")
                output_path.write_bytes(b"rendered")

            with (
                patch.object(pipeline, "_safe_probe_duration", side_effect=probe_staged),
                patch.object(pipeline, "_run_tts_from_script", side_effect=no_tts),
                patch.object(pipeline, "_video_dimensions", return_value=(1920, 1080)),
                patch.object(pipeline, "_write_ass"),
                patch.object(pipeline, "_combine_audio_chunks"),
                patch.object(pipeline, "_render_video", side_effect=render_from_staged),
                patch.object(pipeline, "_write_srt"),
            ):
                events = list(pipeline.render_script(workspace, source, [segment]))

            self.assertFalse(source.exists())
            self.assertTrue(any('"status": "success"' in event for event in events))


class RenderJobStreamRaceTests(unittest.TestCase):
    def test_terminal_event_is_rechecked_before_stream_closes(self) -> None:
        success = {
            "status": "success",
            "phase": "complete",
            "progress": 100,
            "video_url": "/media/result.mp4",
        }

        class RacingStore:
            def __init__(self) -> None:
                self.read_count = 0

            def read_events(self, job_id, attempt):
                self.read_count += 1
                return [] if self.read_count == 1 else [success]

            def read_manifest(self, job_id):
                return {"status": "complete"}

        class Dispatcher:
            store = RacingStore()

        class ConnectedRequest:
            async def is_disconnected(self):
                return False

        response = routes._render_job_streaming_response(
            ConnectedRequest(),
            Dispatcher(),
            RenderJobSubmission(job_id="a" * 32, attempt=1, enqueued=False),
        )

        async def consume() -> str:
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(consume())
        self.assertIn('"status": "success"', body)
        self.assertGreaterEqual(Dispatcher.store.read_count, 2)


class StreamingCleanupRaceTests(unittest.TestCase):
    def test_workspace_is_not_deleted_while_blocking_worker_is_still_alive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = WorkspaceManager(root / "workspaces", root / "output")
            workspace = manager.create()
            worker_started = threading.Event()
            release_worker = threading.Event()

            class DisconnectedRequest:
                async def is_disconnected(self):
                    return True

            def runner_factory(_cancel_event):
                worker_started.set()
                release_worker.wait(timeout=2)
                self.assertTrue(workspace.root.exists())
                if False:
                    yield ""

            with patch.object(routes, "STREAM_WORKER_JOIN_TIMEOUT_SECONDS", 0.01):
                response = routes._streaming_pipeline_response(
                    DisconnectedRequest(),
                    manager,
                    workspace,
                    runner_factory,
                )

                async def consume() -> None:
                    async for _ in response.body_iterator:
                        pass

                asyncio.run(consume())

            self.assertTrue(worker_started.wait(timeout=1))
            self.assertTrue(workspace.root.exists())
            release_worker.set()
            deadline = time.time() + 2
            while workspace.root.exists() and time.time() < deadline:
                time.sleep(0.01)
            self.assertFalse(workspace.root.exists())


if __name__ == "__main__":
    unittest.main()
