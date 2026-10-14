"""Celery worker tasks for document processing and OCR execution."""

import uuid
from typing import Any, Dict
from sqlalchemy import delete, select, update

from app.core.constants import TaskStatus
from app.core.logging import get_logger
from app.db.models import Document, DocumentPage
from app.db.session import get_sync_db
from app.ocr.pipeline import get_ocr_pipeline
from app.queue.celery_app import celery_app
from app.queue.status import update_task_stage
from app.storage.artifact_store import get_artifact_store

logger = get_logger("celery_tasks")


@celery_app.task(bind=True, name="app.queue.tasks.process_document_task", max_retries=2, default_retry_delay=10)
def process_document_task(self, document_id: str) -> Dict[str, Any]:
    """Execute end-to-end OCR and page artifact generation in background worker."""
    task_id = self.request.id or f"task_{document_id[:8]}"
    logger.info(f"Task '{task_id}': Commencing OCR processing for document '{document_id}'...")

    artifact_store = get_artifact_store()
    pipeline = get_ocr_pipeline()

    try:
        # 1. Update state: PROCESSING
        update_task_stage(
            task_id=task_id,
            document_id=document_id,
            stage="PROCESSING",
            progress=15.0,
            message="Worker claimed task. Downloading file from MinIO...",
            status=TaskStatus.PROCESSING.value
        )

        # 2. Fetch document record synchronously
        with get_sync_db() as session:
            stmt = select(Document).where(Document.id == document_id)
            doc = session.execute(stmt).scalars().first()
            if not doc:
                raise ValueError(f"Document with ID '{document_id}' was not found in database.")
            raw_path = doc.minio_raw_path
            filename = doc.filename
            category = doc.category

        # 3. Retrieve raw file from MinIO
        file_bytes = artifact_store.get_raw_document(raw_path)

        # 4. Execute OCR Pipeline
        update_task_stage(
            task_id=task_id,
            document_id=document_id,
            stage="OCR_EXECUTION",
            progress=40.0,
            message=f"Extracting pages using multi-format OCR engine for '{filename}'..."
        )

        ocr_result, ocr_json_key = pipeline.process(
            document_id=document_id,
            filename=filename,
            category=category,
            file_bytes=file_bytes
        )

        # 5. Persist page-level records into PostgreSQL
        update_task_stage(
            task_id=task_id,
            document_id=document_id,
            stage="SAVING_PAGES",
            progress=75.0,
            message=f"Saving {len(ocr_result.pages)} page record(s) and structured JSON artifacts..."
        )

        with get_sync_db() as session:
            session.execute(delete(DocumentPage).where(DocumentPage.document_id == document_id))
            page_entities = [
                DocumentPage(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    page_number=p.page_number,
                    has_selectable_text=p.has_selectable_text,
                    ocr_engine=p.ocr_engine,
                    confidence=p.confidence,
                    char_count=p.char_count,
                    table_count=p.table_count,
                    image_minio_path=p.image_path,
                    structured_json_path=ocr_json_key,
                    raw_text=p.raw_text,
                    warnings=p.warnings
                )
                for p in ocr_result.pages
            ]
            session.add_all(page_entities)

            # Update Document summary fields
            values = {
                "status": TaskStatus.OCR_COMPLETED.value,
                "minio_ocr_path": ocr_json_key,
                "ocr_engine_used": ocr_result.ocr_engine_used,
                "ocr_confidence": ocr_result.average_confidence,
                "total_pages": ocr_result.total_pages,
                "ocr_text_preview": ocr_result.full_text[:10000],
                "metadata_json": ocr_result.metadata
            }
            update_stmt = update(Document).where(Document.id == document_id).values(**values)
            session.execute(update_stmt)
            session.commit()

        # 6. OCR stage completed, transition to INDEXING
        update_task_stage(
            task_id=task_id,
            document_id=document_id,
            stage="INDEXING",
            progress=85.0,
            message=f"OCR completed ({ocr_result.total_pages} page(s)). Indexing chunks in Milvus & BM25...",
            status=TaskStatus.INDEXING.value
        )

        # 7. Dual Indexing in Milvus and Persistent BM25
        from app.indexing.index_pipeline import get_indexing_pipeline
        indexing_pipeline = get_indexing_pipeline()
        total_chunks = indexing_pipeline.index_ocr_result(ocr_result)

        # 8. Finalize task: SUCCESS
        with get_sync_db() as session:
            stmt = update(Document).where(Document.id == document_id).values(
                status=TaskStatus.SUCCESS.value,
                total_chunks=total_chunks
            )
            session.execute(stmt)
            session.commit()

        update_task_stage(
            task_id=task_id,
            document_id=document_id,
            stage="SUCCESS",
            progress=100.0,
            message=f"Pipeline finished: {ocr_result.total_pages} page(s), {total_chunks} chunks indexed.",
            status=TaskStatus.SUCCESS.value
        )

        logger.info(f"Task '{task_id}' fully completed for document '{document_id}'.")
        return {
            "document_id": document_id,
            "status": TaskStatus.SUCCESS.value,
            "total_pages": ocr_result.total_pages,
            "total_chunks": total_chunks,
            "confidence": ocr_result.average_confidence,
            "ocr_json_path": ocr_json_key
        }

    except Exception as exc:
        logger.error(f"Task '{task_id}' failed for document '{document_id}': {exc}", exc_info=True)
        update_task_stage(
            task_id=task_id,
            document_id=document_id,
            stage="FAILED",
            progress=0.0,
            message=f"Processing error: {str(exc)}",
            status=TaskStatus.FAILURE.value,
            error_message=str(exc)
        )
        raise self.retry(exc=exc) if self.request.retries < self.max_retries else exc
