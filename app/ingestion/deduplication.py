"""SHA-256 fingerprint deduplication service."""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.constants import TaskStatus
from app.db.models import Document
from app.db.repositories import DocumentRepository


class DeduplicationService:
    """Detects previously uploaded and processed identical files via SHA-256 hash."""

    def __init__(self, session: AsyncSession) -> None:
        self.repo = DocumentRepository(session)

    async def find_existing(self, file_hash: str) -> Optional[Document]:
        """Search for an existing document with the given file hash."""
        existing = await self.repo.get_by_hash(file_hash)
        if not existing:
            return None

        # If previous attempt failed, allow reprocessing
        if existing.status == TaskStatus.FAILURE.value:
            return None

        return existing
