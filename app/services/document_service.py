"""Document management service for retrieving details, presigned URLs, and deletions."""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.db.repositories import DocumentRepository, DocumentPageRepository
from app.indexing.milvus_store import get_milvus_store
from app.schemas.document import DocumentDetailResponse, DocumentListResponse, PageSummary
from app.storage.artifact_store import get_artifact_store

settings = get_settings()
logger = get_logger("document_service")


class DocumentService:
    """Handles document queries, artifact links, and cascading cleanup."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.doc_repo = DocumentRepository(session)
        self.page_repo = DocumentPageRepository(session)
        self.artifact_store = get_artifact_store()
        self.milvus_store = get_milvus_store()

    async def get_document(self, document_id: str) -> DocumentDetailResponse:
        """Fetch full document details with presigned artifact URLs."""
        doc = await self.doc_repo.get_by_id(document_id, include_pages=True)
        if not doc:
            raise DocumentNotFoundError(document_id)

        # Generate presigned download URL for raw file
        raw_url = None
        if doc.minio_raw_path:
            try:
                raw_url = self.artifact_store.minio.get_presigned_url(
                    bucket_name=self.artifact_store.doc_bucket,
                    object_key=doc.minio_raw_path,
                    expires_seconds=3600
                )
            except Exception as exc:
                logger.warning(f"Failed to generate presigned raw URL: {exc}")

        # Generate presigned download URL for OCR JSON
        ocr_url = None
        if doc.minio_ocr_path:
            try:
                ocr_url = self.artifact_store.minio.get_presigned_url(
                    bucket_name=self.artifact_store.artifact_bucket,
                    object_key=doc.minio_ocr_path,
                    expires_seconds=3600
                )
            except Exception as exc:
                logger.warning(f"Failed to generate presigned OCR URL: {exc}")

        # Format pages
        pages = [
            PageSummary(
                page_number=p.page_number,
                ocr_engine=p.ocr_engine,
                confidence=p.confidence,
                char_count=p.char_count,
                table_count=p.table_count,
                has_selectable_text=p.has_selectable_text,
                warnings=p.warnings,
                raw_text=p.raw_text
            )
            for p in (doc.pages or [])
        ]

        full_extracted_text = "\n\n".join(p.raw_text for p in (doc.pages or []) if p.raw_text) or doc.ocr_text_preview

        return DocumentDetailResponse(
            id=doc.id,
            file_hash=doc.file_hash,
            filename=doc.filename,
            file_size=doc.file_size,
            content_type=doc.content_type,
            category=doc.category,
            status=doc.status,
            ocr_engine_used=doc.ocr_engine_used,
            ocr_confidence=doc.ocr_confidence,
            total_pages=doc.total_pages,
            total_chunks=doc.total_chunks,
            ocr_text_preview=full_extracted_text,
            error_message=doc.error_message,
            task_id=doc.task_id,
            metadata_json=doc.metadata_json,
            raw_file_url=raw_url,
            ocr_json_url=ocr_url,
            pages=pages,
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )

    async def list_documents(self, limit: int = 50, offset: int = 0) -> DocumentListResponse:
        """List documents ordered by creation time descending."""
        docs = await self.doc_repo.list_documents(limit=limit, offset=offset)
        items = []
        for d in docs:
            items.append(
                DocumentDetailResponse(
                    id=d.id,
                    file_hash=d.file_hash,
                    filename=d.filename,
                    file_size=d.file_size,
                    content_type=d.content_type,
                    category=d.category,
                    status=d.status,
                    ocr_engine_used=d.ocr_engine_used,
                    ocr_confidence=d.ocr_confidence,
                    total_pages=d.total_pages,
                    total_chunks=d.total_chunks,
                    ocr_text_preview=d.ocr_text_preview,
                    error_message=d.error_message,
                    task_id=d.task_id,
                    metadata_json=d.metadata_json,
                    raw_file_url=None,
                    ocr_json_url=None,
                    pages=[],
                    created_at=d.created_at,
                    updated_at=d.updated_at
                )
            )
        return DocumentListResponse(total=len(items), items=items)

    async def delete_document(self, document_id: str) -> bool:
        """Delete document, cascading across MinIO artifacts, Milvus vectors, and PostgreSQL."""
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)

        # 1. Clean up MinIO raw file
        if doc.minio_raw_path:
            try:
                self.artifact_store.minio.delete_object(self.artifact_store.doc_bucket, doc.minio_raw_path)
            except Exception as exc:
                logger.warning(f"Error deleting MinIO raw object for {document_id}: {exc}")

        # 2. Clean up MinIO OCR JSON
        if doc.minio_ocr_path:
            try:
                self.artifact_store.minio.delete_object(self.artifact_store.artifact_bucket, doc.minio_ocr_path)
            except Exception as exc:
                logger.warning(f"Error deleting MinIO OCR object for {document_id}: {exc}")

        # 3. Clean up Milvus dense vector collection chunks
        try:
            self.milvus_store.delete_by_document_id(document_id)
        except Exception as exc:
            logger.warning(f"Error deleting Milvus vectors for {document_id}: {exc}")

        # 4. Clean up BM25 keyword index file
        try:
            from app.indexing.bm25_store import get_bm25_store
            get_bm25_store().delete_document(document_id)
        except Exception as exc:
            logger.warning(f"Error deleting BM25 index for {document_id}: {exc}")

        # 5. Delete PostgreSQL record (cascades to pages and task logs)
        deleted = await self.doc_repo.delete(document_id)
        await self.session.commit()
        logger.info(f"Successfully deleted document '{document_id}' and all dependent artifacts.")
        return deleted
