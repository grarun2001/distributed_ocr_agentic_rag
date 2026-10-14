"""FastAPI dependency injection providers."""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_async_db
from app.services.upload_service import UploadService
from app.services.document_service import DocumentService
from app.services.task_service import TaskService


def get_upload_service(session: AsyncSession = Depends(get_async_db)) -> UploadService:
    """Provide UploadService instance with active async session."""
    return UploadService(session)


def get_document_service(session: AsyncSession = Depends(get_async_db)) -> DocumentService:
    """Provide DocumentService instance with active async session."""
    return DocumentService(session)


def get_task_service(session: AsyncSession = Depends(get_async_db)) -> TaskService:
    """Provide TaskService instance with active async session."""
    return TaskService(session)
