"""Pydantic schemas for asynchronous task status tracking."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class TaskMilestoneLog(BaseModel):
    """Single stage log within a task."""
    stage: str
    progress: float
    message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TaskStatusResponse(BaseModel):
    """Full task lifecycle status response."""
    task_id: str = Field(..., description="Unique task identifier")
    document_id: str = Field(..., description="Document identifier associated with the task")
    status: str = Field(..., description="Current status: PENDING, PROCESSING, OCR_COMPLETED, INDEXING, SUCCESS, FAILURE")
    progress: float = Field(0.0, description="Overall completion percentage (0.0 to 100.0)")
    current_stage: Optional[str] = Field(None, description="Active processing step")
    error_message: Optional[str] = Field(None, description="Detailed failure message if status is FAILURE")
    logs: List[TaskMilestoneLog] = Field(default_factory=list, description="Historical milestone logs")
    updated_at: datetime = Field(..., description="Last status update timestamp")

    model_config = {"from_attributes": True}
