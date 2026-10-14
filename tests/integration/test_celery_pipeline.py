"""Integration tests for Celery document processing task pipeline."""

import uuid
import pymupdf
import pytest
from sqlalchemy import select

from app.core.constants import TaskStatus
from app.db.models import Document, DocumentPage
from app.db.session import get_sync_db
from app.queue.tasks import process_document_task
from app.storage.artifact_store import get_artifact_store


def create_sample_pdf_bytes() -> bytes:
    """Generate a valid multi-page digital PDF in memory using PyMuPDF."""
    doc = pymupdf.open()
    # Page 1: Digital text and headings
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 72), "SECTION 1: ANNUAL REPORT", fontsize=18)
    page1.insert_text((50, 120), "The distributed OCR agentic system achieves high accuracy.", fontsize=12)
    page1.insert_text((50, 140), "Performance metrics show linear scalability across worker nodes.", fontsize=12)

    # Page 2: Summary text
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 72), "SECTION 2: FINANCIAL SUMMARY", fontsize=18)
    page2.insert_text((50, 120), "Total Revenue: $1,250,000. Operational Costs: $450,000.", fontsize=12)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_celery_pipeline_execution():
    """Verify synchronous execution of process_document_task on multi-page PDF."""
    artifact_store = get_artifact_store()
    test_id = str(uuid.uuid4())
    pdf_bytes = create_sample_pdf_bytes()

    # 1. Save raw PDF in MinIO
    raw_key = artifact_store.save_raw_document(test_id, "annual_report.pdf", pdf_bytes, "application/pdf")

    # 2. Insert initial Document record in PostgreSQL
    with get_sync_db() as session:
        doc = Document(
            id=test_id,
            file_hash=f"hash_{test_id[:12]}",
            filename="annual_report.pdf",
            file_size=len(pdf_bytes),
            content_type="application/pdf",
            category="PDF_DIGITAL",
            minio_raw_path=raw_key,
            status=TaskStatus.PENDING.value,
            task_id=f"task_{test_id[:8]}"
        )
        session.add(doc)
        session.commit()

    # 3. Execute Celery worker task directly
    result = process_document_task(test_id)
    assert result["status"] == TaskStatus.SUCCESS.value
    assert result["total_pages"] == 2
    assert result["confidence"] == 100.0
    assert result["total_chunks"] >= 2

    # 4. Verify Document record updated in PostgreSQL
    with get_sync_db() as session:
        stmt = select(Document).where(Document.id == test_id)
        updated_doc = session.execute(stmt).scalars().first()
        assert updated_doc.status == TaskStatus.SUCCESS.value
        assert updated_doc.total_pages == 2
        assert updated_doc.total_chunks >= 2
        assert "ANNUAL REPORT" in updated_doc.ocr_text_preview

        # Verify DocumentPage records
        stmt_pages = select(DocumentPage).where(DocumentPage.document_id == test_id).order_by(DocumentPage.page_number)
        pages = list(session.execute(stmt_pages).scalars().all())
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[0].has_selectable_text is True
        assert pages[0].ocr_engine == "NATIVE"
        assert "FINANCIAL SUMMARY" in pages[1].raw_text

        # 5. Clean up
        session.delete(updated_doc)
        session.commit()

    # Clean up MinIO artifacts
    artifact_store.minio.delete_object(artifact_store.doc_bucket, raw_key)
    artifact_store.minio.delete_object(artifact_store.artifact_bucket, result["ocr_json_path"])
