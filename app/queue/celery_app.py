"""Celery application configuration for distributed background task execution."""

from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "distributed_ocr_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.queue.tasks"]
)
celery_app.conf.update(
    task_default_queue="ocr_tasks",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,       # 10 minutes max hard timeout
    task_soft_time_limit=540,  # 9 minutes soft timeout
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_routes={
        "app.queue.tasks.process_document_task": {"queue": "ocr_tasks"}
    }
)
