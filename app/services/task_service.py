"""Task monitoring and status reporting service."""

from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import TaskNotFoundError
from app.db.repositories import DocumentRepository, TaskLogRepository
from app.schemas.task import TaskStatusResponse, TaskMilestoneLog


class TaskService:
    """Provides real-time task progress and audit history."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.doc_repo = DocumentRepository(session)
        self.task_repo = TaskLogRepository(session)

    async def get_task_status(self, task_id: str) -> TaskStatusResponse:
        """Fetch current task status, progress percentage, and milestone history."""
        doc = await self.doc_repo.get_by_task_id(task_id)
        if not doc:
            raise TaskNotFoundError(task_id)

        logs = await self.task_repo.get_logs_by_task_id(task_id)
        current_progress = logs[-1].progress if logs and logs[-1].progress is not None else 0.0
        current_stage = logs[-1].stage if logs else "PENDING"

        # If document already succeeded, progress is 100%
        if doc.status == "SUCCESS":
            current_progress = 100.0

        milestone_logs = [
            TaskMilestoneLog(
                stage=log.stage,
                progress=log.progress or 0.0,
                message=log.message,
                created_at=log.created_at
            )
            for log in logs
        ]

        return TaskStatusResponse(
            task_id=task_id,
            document_id=doc.id,
            status=doc.status,
            progress=current_progress,
            current_stage=current_stage,
            error_message=doc.error_message,
            logs=milestone_logs,
            updated_at=doc.updated_at
        )
