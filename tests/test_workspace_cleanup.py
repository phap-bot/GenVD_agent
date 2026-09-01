from __future__ import annotations

import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from app.utils.workspace import WorkspaceManager


class WorkspaceCleanupTests(unittest.TestCase):
    def test_evicts_only_expired_workspace_directories(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            manager = WorkspaceManager(Path(root) / "workspaces", Path(root) / "output")
            old = Path(root) / "workspaces" / "old"
            fresh = Path(root) / "workspaces" / "fresh"
            old.mkdir(parents=True)
            fresh.mkdir(parents=True)
            old_file = old / "large.tmp"
            old_file.write_bytes(b"x")
            expired_at = time.time() - 2 * 3600
            import os

            os.utime(old, (expired_at, expired_at))
            os.utime(old_file, (expired_at, expired_at))

            self.assertEqual(manager.evict_stale(3600), 1)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())

    def test_active_workspace_is_protected_from_daily_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            manager = WorkspaceManager(Path(root) / "workspaces", Path(root) / "output")
            workspace = manager.create()
            import os

            expired_at = time.time() - 2 * 3600
            os.utime(workspace.root, (expired_at, expired_at))
            self.assertEqual(manager.evict_stale(3600), 0)
            self.assertTrue(workspace.root.exists())
            manager.cleanup(workspace)

    def test_evicts_expired_output_but_keeps_fresh_and_active_files(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            manager = WorkspaceManager(Path(root) / "workspaces", Path(root) / "output")
            workspace = manager.create()
            old = manager.output_root / "old_render.mp4"
            fresh = manager.output_root / "fresh_render.mp4"
            active = manager.output_root / f"{workspace.request_id}_source.mp4"
            old.write_bytes(b"old")
            fresh.write_bytes(b"fresh")
            active.write_bytes(b"active")
            import os

            expired_at = time.time() - 13 * 3600
            os.utime(old, (expired_at, expired_at))
            os.utime(active, (expired_at, expired_at))

            self.assertEqual(manager.evict_stale_output(12 * 3600), 1)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(active.exists())
            manager.cleanup(workspace)

    def test_protects_queued_render_job_output(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            manager = WorkspaceManager(root_path / "workspaces", root_path / "output")
            job_id = "queued-job"
            render_root = root_path / "render_jobs" / job_id
            render_root.mkdir(parents=True)
            (render_root / "manifest.json").write_text(
                '{"job_id":"queued-job","status":"running"}', encoding="utf-8"
            )
            output = manager.output_root / f"{job_id}_script_dubbed.mp4"
            output.write_bytes(b"active render")
            import os

            expired_at = time.time() - 13 * 3600
            os.utime(output, (expired_at, expired_at))
            with patch.dict(os.environ, {"AUTODUB_RENDER_JOB_ROOT": str(root_path / "render_jobs")}):
                self.assertEqual(manager.evict_stale_output(12 * 3600), 0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
