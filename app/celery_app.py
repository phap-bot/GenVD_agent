from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

try:
    from celery import Celery
except ImportError as exc:
    raise RuntimeError(
        "Celery queue backend is enabled but celery is not installed. Install the project's requirements."
    ) from exc


REDIS_URL = os.environ.get("AUTODUB_REDIS_URL", "redis://127.0.0.1:6379/0")

celery_app = Celery(
    "auto_dubbing",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"],
)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 24 * 60 * 60},
    result_backend_transport_options={"visibility_timeout": 24 * 60 * 60},
    result_expires=7 * 24 * 60 * 60,
    task_routes={"app.tasks.run_render_job": {"queue": "render"}},
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Bangkok",
    enable_utc=True,
)
