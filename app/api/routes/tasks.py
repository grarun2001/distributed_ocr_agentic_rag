"""Task monitoring endpoints for tracking asynchronous background processing."""

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_task_service
from app.schemas.task import TaskStatusResponse
from app.services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["Tasks & Processing"])


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get task status and progress",
    description="Retrieve live status, completion percentage (0-100%), and milestone log history for a background task."
)
async def get_task_status(
    task_id: str,
    service: TaskService = Depends(get_task_service)
) -> TaskStatusResponse:
    """Retrieve task lifecycle progress."""
    return await service.get_task_status(task_id)
