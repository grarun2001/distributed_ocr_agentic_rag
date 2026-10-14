"""Synchronous task progress updater for Celery workers."""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import update

from app.core.logging import get_logger
from app.db.models import Document, TaskLog
from app.db.session import get_sync_db

logger = get_logger("queue_status")


def update_task_stage(
    task_id: str,
    document_id: str,
    stage: str,
    progress: float,
    message: Optional[str] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None
) -> None:
    """Record task milestone log and update document status synchronously."""
    try:
        with get_sync_db() as session:
            # 1. Add TaskLog entry
            log_entry = TaskLog(
                task_id=task_id,
                document_id=document_id,
                stage=stage,
                progress=round(progress, 1),
                message=message
            )
            session.add(log_entry)

            # 2. Update Document status if provided
            if status is not None or error_message is not None:
                values = {"updated_at": datetime.now(timezone.utc)}
                if status is not None:
                    values["status"] = status
                if error_message is not None:
                    values["error_message"] = error_message

                stmt = update(Document).where(Document.id == document_id).values(**values)
                session.execute(stmt)

            session.commit()
    except Exception as exc:
        logger.error(f"Failed updating task status for {task_id}: {exc}")
