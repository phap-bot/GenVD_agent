from __future__ import annotations

from typing import Any

from app.celery_app import celery_app
from app.services.render_job_service import RenderJobRunner


@celery_app.task(
    bind=True,
    name="app.tasks.run_render_job",
    acks_late=True,
    reject_on_worker_lost=True,
)
def run_render_job(self, job_id: str, attempt: int) -> dict[str, Any]:
    def publish_progress(payload: dict[str, Any]) -> None:
        self.update_state(
            state="PROGRESS",
            meta={
                "job_id": job_id,
                "attempt": attempt,
                "phase": payload.get("phase"),
                "progress": payload.get("progress"),
                "status": payload.get("status"),
            },
        )

    result = RenderJobRunner().run(job_id, attempt, state_callback=publish_progress)
    if result.get("status") == "failed":
        raise RuntimeError(f"Render job {job_id} failed; inspect its durable manifest for details.")
    return result
