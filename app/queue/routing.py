"""Celery task queue names and routing configurations."""

QUEUE_OCR_TASKS = "ocr_tasks"
QUEUE_DEFAULT = "default"

TASK_PROCESS_DOCUMENT = "app.queue.tasks.process_document_task"

TASK_ROUTES = {
    TASK_PROCESS_DOCUMENT: {"queue": QUEUE_OCR_TASKS}
}
