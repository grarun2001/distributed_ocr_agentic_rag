"""Asynchronous database repository layer implementing CRUD operations."""

from typing import List, Optional
from datetime import datetime, timezone
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Document, DocumentPage, TaskLog, QueryAudit


class DocumentRepository:
    """Repository handling Document database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, document_id: str, include_pages: bool = False) -> Optional[Document]:
        stmt = select(Document).where(Document.id == document_id)
        if include_pages:
            stmt = stmt.options(selectinload(Document.pages), selectinload(Document.task_logs))
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_hash(self, file_hash: str) -> Optional[Document]:
        stmt = select(Document).where(Document.file_hash == file_hash)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_task_id(self, task_id: str) -> Optional[Document]:
        stmt = select(Document).where(Document.task_id == task_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def create(self, document: Document) -> Document:
        self.session.add(document)
        await self.session.flush()
        return document

    async def update_status(
        self,
        document_id: str,
        status: str,
        error_message: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> Optional[Document]:
        values = {
            "status": status,
            "updated_at": datetime.now(timezone.utc)
        }
        if error_message is not None:
            values["error_message"] = error_message
        if task_id is not None:
            values["task_id"] = task_id

        stmt = update(Document).where(Document.id == document_id).values(**values)
        await self.session.execute(stmt)
        return await self.get_by_id(document_id)

    async def list_documents(self, limit: int = 50, offset: int = 0) -> List[Document]:
        stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, document_id: str) -> bool:
        stmt = delete(Document).where(Document.id == document_id)
        result = await self.session.execute(stmt)
        return result.rowcount > 0


class TaskLogRepository:
    """Repository handling background task progress logs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_log(
        self,
        task_id: str,
        document_id: str,
        stage: str,
        progress: float = 0.0,
        message: Optional[str] = None
    ) -> TaskLog:
        log_entry = TaskLog(
            task_id=task_id,
            document_id=document_id,
            stage=stage,
            progress=progress,
            message=message
        )
        self.session.add(log_entry)
        await self.session.flush()
        return log_entry

    async def get_logs_by_task_id(self, task_id: str) -> List[TaskLog]:
        stmt = select(TaskLog).where(TaskLog.task_id == task_id).order_by(TaskLog.created_at.asc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class DocumentPageRepository:
    """Repository handling individual page OCR artifacts."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_pages_batch(self, pages: List[DocumentPage]) -> List[DocumentPage]:
        self.session.add_all(pages)
        await self.session.flush()
        return pages

    async def get_by_document_id(self, document_id: str) -> List[DocumentPage]:
        stmt = select(DocumentPage).where(DocumentPage.document_id == document_id).order_by(DocumentPage.page_number.asc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class QueryAuditRepository:
    """Repository recording RAG query routing and latency metrics."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record_audit(self, audit: QueryAudit) -> QueryAudit:
        self.session.add(audit)
        await self.session.flush()
        return audit
