from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.models.schemas import PipelineConfig, RenderScriptRequest
from app.services.render_job_service import RenderJobDispatcher, RenderJobRunner, RenderJobStore


def _payload(*, clone_path: str | None = None, text: str = "Edited line") -> RenderScriptRequest:
    return RenderScriptRequest(
        source_video_path="/media/source.mp4",
        voice_mode="clone" if clone_path else "system",
        clone_reference_audio_path=clone_path,
        voice_model="Truc Ly",
        copyright_confirmed=True,
        copyright_source="owned",
        segments=[
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "original_text": "Original line",
                "translated_text": text,
                "voice_model": "Truc Ly",
            }
        ],
    )


def _config(clone_path: Path | None = None) -> PipelineConfig:
    return PipelineConfig(
        voice_mode="clone" if clone_path else "system",
        clone_reference_audio_path=str(clone_path) if clone_path else None,
        copyright_confirmed=True,
        copyright_source="owned",
    )


class RenderJobFingerprintTests(unittest.TestCase):
    def test_same_content_with_different_paths_reuses_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_a = root / "a.mp4"
            source_b = root / "b.mp4"
            source_a.write_bytes(b"same-video")
            source_b.write_bytes(b"same-video")
            store = RenderJobStore(root / "jobs", root / "output")

            first = store.prepare(_payload(), _config(), source_a, None)
            second = store.prepare(_payload(), _config(), source_b, None)

            self.assertEqual(first, second)
            self.assertEqual(store.source_path(first).read_bytes(), b"same-video")

    def test_script_change_creates_new_job_without_copying_source_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"shared-video")
            store = RenderJobStore(root / "jobs", root / "output")

            first = store.prepare(_payload(text="First edit"), _config(), source, None)
            second = store.prepare(_payload(text="Second edit"), _config(), source, None)

            self.assertNotEqual(first, second)
            self.assertTrue(os.path.samefile(store.source_path(first), store.source_path(second)))
            source_blobs = list((store.artifact_root() / "sources").glob("*.mp4"))
            self.assertEqual(len(source_blobs), 1)

    def test_clone_content_change_creates_a_new_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            clone_a = root / "a.wav"
            clone_b = root / "b.wav"
            source.write_bytes(b"video")
            clone_a.write_bytes(b"voice-a")
            clone_b.write_bytes(b"voice-b")
            store = RenderJobStore(root / "jobs", root / "output")

            first = store.prepare(_payload(clone_path="/media/a.wav"), _config(clone_a), source, clone_a)
            second = store.prepare(_payload(clone_path="/media/b.wav"), _config(clone_b), source, clone_b)

            self.assertNotEqual(first, second)

    def test_local_api_process_restart_requeues_without_stale_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            store = RenderJobStore(root / "jobs", root / "output")
            job_id = store.prepare(_payload(), _config(), source, None)

            first_attempt, first_enqueue = store.begin_attempt(job_id)
            manifest = store.read_manifest(job_id)
            manifest["status"] = "running"
            manifest["owner_pid"] = os.getpid() + 10000
            store._atomic_write_json(store.manifest_path(job_id), manifest)
            with patch("app.services.render_job_service._pid_is_running", return_value=False):
                second_attempt, second_enqueue = store.begin_attempt(job_id)

            self.assertTrue(first_enqueue)
            self.assertTrue(second_enqueue)
            self.assertEqual(second_attempt, first_attempt + 1)

    def test_live_local_owner_is_not_requeued_when_manifest_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            store = RenderJobStore(root / "jobs", root / "output")
            job_id = store.prepare(_payload(), _config(), source, None)
            first_attempt, _ = store.begin_attempt(job_id)
            manifest = store.read_manifest(job_id)
            manifest.update({"status": "running", "updated_at": 0, "owner_pid": 12345})
            store._atomic_write_json(store.manifest_path(job_id), manifest)

            with patch("app.services.render_job_service._pid_is_running", return_value=True):
                second_attempt, second_enqueue = store.begin_attempt(job_id)

            self.assertFalse(second_enqueue)
            self.assertEqual(second_attempt, first_attempt)

    def test_dispatcher_resume_requeues_orphaned_local_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            store = RenderJobStore(root / "jobs", root / "output")
            job_id = store.prepare(_payload(), _config(), source, None)
            first_attempt, _ = store.begin_attempt(job_id)
            manifest = store.read_manifest(job_id)
            manifest.update({"status": "running", "owner_pid": 12345})
            store._atomic_write_json(store.manifest_path(job_id), manifest)
            dispatcher = RenderJobDispatcher(store)

            with (
                patch("app.services.render_job_service._pid_is_running", return_value=False),
                patch.object(dispatcher, "_submit_local") as submit_local,
            ):
                submission = dispatcher.resume(job_id)

            self.assertTrue(submission.enqueued)
            self.assertEqual(submission.attempt, first_attempt + 1)
            submit_local.assert_called_once_with(job_id, submission.attempt)

    def test_execution_lease_suppresses_duplicate_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            store = RenderJobStore(root / "jobs", root / "output")
            job_id = store.prepare(_payload(), _config(), source, None)
            attempt, _ = store.begin_attempt(job_id)

            first_token = store.acquire_execution(job_id, attempt)
            duplicate_token = store.acquire_execution(job_id, attempt)

            self.assertIsNotNone(first_token)
            self.assertIsNone(duplicate_token)
            store.release_execution(job_id, first_token or "")
            self.assertIsNotNone(store.acquire_execution(job_id, attempt))


class RenderJobManifestTests(unittest.TestCase):
    def test_atomic_manifest_write_retries_windows_sharing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            real_replace = os.replace
            calls = 0

            def flaky_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError("sharing violation")
                real_replace(source, destination)

            with (
                patch("app.services.render_job_service.os.replace", side_effect=flaky_replace),
                patch("app.services.render_job_service.time.sleep") as sleep,
            ):
                RenderJobStore._atomic_write_json(path, {"status": "running"})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "running"})
            self.assertEqual(calls, 3)
            self.assertEqual(sleep.call_count, 2)


class RenderJobRunnerTests(unittest.TestCase):
    def test_runner_persists_progress_and_completed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            store = RenderJobStore(root / "jobs", root / "output")
            job_id = store.prepare(_payload(), _config(), source, None)
            attempt, _ = store.begin_attempt(job_id)

            class SuccessfulPipeline:
                def __init__(self, _config):
                    pass

                def render_script(self, workspace, source_video_path, script_segments):
                    self.assert_source = source_video_path
                    yield "data: " + json.dumps({
                        "status": "processing",
                        "phase": "voice",
                        "progress": 60,
                        "stats": {"chunks": 1, "groups": 1},
                    }) + "\n\n"
                    output = workspace.output_dir / f"{workspace.request_id}_script_dubbed.mp4"
                    output.write_bytes(b"rendered")
                    yield "data: " + json.dumps({
                        "status": "success",
                        "phase": "complete",
                        "progress": 100,
                        "video_url": f"/media/{output.name}",
                    }) + "\n\n"

            with patch("app.services.render_job_service.AutoDubbingPipeline", SuccessfulPipeline):
                result = RenderJobRunner(store).run(job_id, attempt)

            manifest = store.read_manifest(job_id)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_groups"], 1)
            self.assertTrue(store.output_path(job_id).is_file())

    def test_failed_attempt_keeps_workspace_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            store = RenderJobStore(root / "jobs", root / "output")
            job_id = store.prepare(_payload(), _config(), source, None)
            attempt, _ = store.begin_attempt(job_id)

            class FailedPipeline:
                def __init__(self, _config):
                    pass

                def render_script(self, workspace, source_video_path, script_segments):
                    checkpoint = workspace.chunks_dir / "0000.wav"
                    checkpoint.write_bytes(b"voice-checkpoint")
                    yield "data: " + json.dumps({
                        "status": "error",
                        "phase": "error",
                        "progress": 0,
                        "error": "ffmpeg failed",
                    }) + "\n\n"

            with patch("app.services.render_job_service.AutoDubbingPipeline", FailedPipeline):
                result = RenderJobRunner(store).run(job_id, attempt)

            checkpoint = store.workspace(job_id).chunks_dir / "0000.wav"
            next_attempt, should_enqueue = store.begin_attempt(job_id)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(should_enqueue)
            self.assertEqual(next_attempt, attempt + 1)


class CeleryDispatchTests(unittest.TestCase):
    def test_celery_dispatch_uses_render_queue_and_persists_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            store = RenderJobStore(root / "jobs", root / "output")
            dispatcher = RenderJobDispatcher(store)
            fake_result = type("Result", (), {"id": "celery-task-id"})()

            with (
                patch.dict(os.environ, {"AUTODUB_QUEUE_BACKEND": "celery"}),
                patch("app.celery_app.celery_app.send_task", return_value=fake_result) as send_task,
            ):
                submission = dispatcher.submit(_payload(), _config(), source, None)

            send_task.assert_called_once_with(
                "app.tasks.run_render_job",
                args=[submission.job_id, submission.attempt],
                queue="render",
            )
            manifest = store.read_manifest(submission.job_id)
            self.assertEqual(manifest["task_id"], "celery-task-id")
            self.assertEqual(manifest["owner_pid"], 0)

    def test_celery_configuration_serializes_gpu_work(self) -> None:
        from app.celery_app import celery_app

        self.assertEqual(celery_app.conf.worker_prefetch_multiplier, 1)
        self.assertTrue(celery_app.conf.task_acks_late)
        self.assertTrue(celery_app.conf.task_reject_on_worker_lost)
        self.assertEqual(celery_app.conf.task_routes["app.tasks.run_render_job"]["queue"], "render")


if __name__ == "__main__":
    unittest.main()
