"""Orchestration service for file validation, hashing, storage, and database persistence."""

import uuid
from typing import List, Tuple
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import TaskStatus
from app.core.exceptions import FileValidationError
from app.core.logging import get_logger
from app.db.models import Document
from app.db.repositories import DocumentRepository, TaskLogRepository
from app.ingestion.deduplication import DeduplicationService
from app.ingestion.file_router import classify_document_category
from app.ingestion.file_validator import get_file_validator
from app.ingestion.fingerprint import compute_sha256
from app.schemas.upload import UploadResponse, BatchUploadResponse
from app.storage.artifact_store import get_artifact_store

logger = get_logger("upload_service")


class UploadService:
    """Handles end-to-end ingestion from HTTP stream to MinIO and PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.validator = get_file_validator()
        self.artifact_store = get_artifact_store()
        self.doc_repo = DocumentRepository(session)
        self.task_repo = TaskLogRepository(session)
        self.dedup_service = DeduplicationService(session)

    async def process_upload(self, file: UploadFile) -> UploadResponse:
        """Process a single uploaded file with validation, hashing, storage, and DB persistence."""
        filename = file.filename or "unknown_file"

        # 1. Read file into memory (size checked)
        contents = await file.read()
        file_size = len(contents)

        # 2. Security & format validation (header magic bytes)
        header = contents[:32] if file_size >= 32 else contents
        content_type = self.validator.validate(filename, file_size, header)

        # 3. Cryptographic SHA-256 fingerprint
        file_hash = compute_sha256(contents)

        # 4. Idempotent Deduplication Check
        existing_doc = await self.dedup_service.find_existing(file_hash)
        if existing_doc:
            logger.info(f"Duplicate file identified: '{filename}' matches doc '{existing_doc.id}'.")
            return UploadResponse(
                document_id=existing_doc.id,
                filename=existing_doc.filename,
                file_hash=existing_doc.file_hash,
                file_size=existing_doc.file_size,
                content_type=existing_doc.content_type,
                category=existing_doc.category,
                status=existing_doc.status,
                task_id=existing_doc.task_id,
                is_duplicate=True,
                created_at=existing_doc.created_at
            )

        # 5. Classify broad document category
        category = classify_document_category(filename, content_type)

        # 6. Upload raw file to MinIO Object Storage
        document_id = str(uuid.uuid4())
        minio_key = self.artifact_store.save_raw_document(
            document_id=document_id,
            filename=filename,
            data=contents,
            content_type=content_type
        )

        # 7. Create Document record in PostgreSQL
        task_id = f"task_{document_id[:8]}"
        doc = Document(
            id=document_id,
            file_hash=file_hash,
            filename=filename,
            file_size=file_size,
            content_type=content_type,
            category=category.value,
            minio_raw_path=minio_key,
            status=TaskStatus.PENDING.value,
            task_id=task_id
        )
        created_doc = await self.doc_repo.create(doc)

        # 8. Record initial task log milestone
        await self.task_repo.create_log(
            task_id=task_id,
            document_id=document_id,
            stage="INGESTION",
            progress=5.0,
            message=f"File '{filename}' successfully ingested and stored in MinIO."
        )

        await self.session.commit()

        # 9. Dispatch background processing task via Celery
        try:
            from app.queue.tasks import process_document_task
            celery_task = process_document_task.apply_async(
                args=[document_id],
                task_id=task_id
            )
            logger.info(f"Dispatched Celery task '{celery_task.id}' for doc '{document_id}'.")
        except Exception as exc:
            logger.warning(f"Failed dispatching to Celery queue: {exc}. Background worker must poll.")

        logger.info(f"Ingested document '{document_id}' (file: '{filename}', size: {file_size} bytes).")

        return UploadResponse(
            document_id=created_doc.id,
            filename=created_doc.filename,
            file_hash=created_doc.file_hash,
            file_size=created_doc.file_size,
            content_type=created_doc.content_type,
            category=created_doc.category,
            status=created_doc.status,
            task_id=created_doc.task_id,
            is_duplicate=False,
            created_at=created_doc.created_at
        )

    async def process_batch_upload(self, files: List[UploadFile]) -> BatchUploadResponse:
        """Process multiple files in a single batch."""
        responses: List[UploadResponse] = []
        dup_count = 0

        for file in files:
            res = await self.process_upload(file)
            if res.is_duplicate:
                dup_count += 1
            responses.append(res)

        return BatchUploadResponse(
            total_files=len(responses),
            duplicates_count=dup_count,
            documents=responses
        )
