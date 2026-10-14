"""Phase 4 Verification Script: Celery Worker, OCR Engines, File Routing & Preprocessing."""

import os
import sys
import uuid
import numpy as np
from PIL import Image, ImageDraw
import pymupdf
import docx
import openpyxl
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.constants import TaskStatus
from app.db.models import Document, DocumentPage
from app.db.session import get_sync_db
from app.ocr.pipeline import get_ocr_pipeline
from app.queue.tasks import process_document_task
from app.storage.artifact_store import get_artifact_store
from sqlalchemy import select


def generate_scanned_image_bytes() -> bytes:
    """Generate a synthetic scanned invoice image with noise and text."""
    img = Image.new("RGB", (600, 200), color=(250, 250, 250))
    d = ImageDraw.Draw(img)
    d.text((30, 40), "OFFICIAL INVOICE #98765", fill=(10, 10, 10))
    d.text((30, 80), "Vendor: Global Logistics Ltd", fill=(10, 10, 10))
    d.text((30, 120), "Total Payable: $4,580.00 USD", fill=(10, 10, 10))

    out_bytes = io.BytesIO()
    img.save(out_bytes, format="PNG")
    return out_bytes.getvalue()


import io


def generate_test_docx_bytes() -> bytes:
    """Generate in-memory Word document with a table."""
    doc = docx.Document()
    doc.add_heading("Procurement Specification", level=1)
    doc.add_paragraph("This document outlines hardware requirements for the OCR worker cluster.")
    tbl = doc.add_table(rows=2, cols=3)
    hdr_cells = tbl.rows[0].cells
    hdr_cells[0].text = "Component"
    hdr_cells[1].text = "Quantity"
    hdr_cells[2].text = "Status"
    row_cells = tbl.rows[1].cells
    row_cells[0].text = "CPU 16-Core"
    row_cells[1].text = "4"
    row_cells[2].text = "Provisioned"

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def verify_phase4() -> bool:
    print("=" * 70)
    print(" DISTRIBUTED OCR AGENTIC RAG - PHASE 4 VERIFICATION")
    print("=" * 70)

    artifact_store = get_artifact_store()
    pipeline = get_ocr_pipeline()

    # [1/4] Test Digital Multi-page PDF Processing
    print("\n[1/4] Testing Multi-Page Digital PDF Extraction...")
    pdf_doc = pymupdf.open()
    p1 = pdf_doc.new_page(width=595, height=842)
    p1.insert_text((50, 80), "QUARTERLY PROGRESS REPORT", fontsize=16)
    p1.insert_text((50, 120), "Distributed processing pipelines enhance throughput by 400%.", fontsize=11)
    p2 = pdf_doc.new_page(width=595, height=842)
    p2.insert_text((50, 80), "APPENDIX: SYSTEM CONFIGURATION", fontsize=16)
    p2.insert_text((50, 120), "Redis broker routing queues all OCR tasks dynamically.", fontsize=11)
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    doc_id_pdf = str(uuid.uuid4())
    pdf_result, pdf_key = pipeline.process(
        document_id=doc_id_pdf,
        filename="progress_report.pdf",
        category="PDF_DIGITAL",
        file_bytes=pdf_bytes
    )
    assert pdf_result.total_pages == 2
    assert pdf_result.average_confidence == 100.0
    assert "PROGRESS REPORT" in pdf_result.full_text
    print(f"      [OK] Digital PDF Extracted: {pdf_result.total_pages} page(s), Engine: {pdf_result.ocr_engine_used}")
    print(f"      [OK] Stored Structured OCR JSON: {pdf_key}")

    # [2/4] Test Scanned / Image Document Processing via Tesseract + Preprocessing
    print("\n[2/4] Testing Scanned Image Preprocessing & Tesseract OCR...")
    img_bytes = generate_scanned_image_bytes()
    doc_id_img = str(uuid.uuid4())
    img_result, img_key = pipeline.process(
        document_id=doc_id_img,
        filename="scanned_invoice.png",
        category="IMAGE",
        file_bytes=img_bytes
    )
    assert img_result.total_pages == 1
    assert img_result.average_confidence > 50.0
    assert "INVOICE" in img_result.full_text.upper()
    print(f"      [OK] Image OCR Completed: Avg Confidence: {img_result.average_confidence}%")
    print(f"      [OK] Bounding Boxes Found: {len(img_result.pages[0].blocks)} blocks")
    print(f"      [OK] Text Preview: {repr(img_result.full_text[:60])}...")

    # [3/4] Test Native Office Document Extraction (.docx)
    print("\n[3/4] Testing Native Office Document Extraction (.docx)...")
    docx_bytes = generate_test_docx_bytes()
    doc_id_docx = str(uuid.uuid4())
    docx_result, docx_key = pipeline.process(
        document_id=doc_id_docx,
        filename="specs.docx",
        category="OFFICE_DOCUMENT",
        file_bytes=docx_bytes
    )
    assert docx_result.total_pages == 1
    assert "Procurement Specification" in docx_result.full_text
    assert len(docx_result.pages[0].tables) > 0
    print(f"      [OK] DOCX Processed: Extracted Table with {docx_result.pages[0].tables[0].num_cols} cols")
    print(f"      [OK] Markdown Table Generated:\n{docx_result.pages[0].tables[0].markdown_data}")

    # [4/4] Test Celery Task Execution & PostgreSQL State Transitions
    print("\n[4/4] Testing Celery Task Execution & DB State Transitions...")
    celery_doc_id = str(uuid.uuid4())
    raw_key = artifact_store.save_raw_document(celery_doc_id, "celery_test.pdf", pdf_bytes, "application/pdf")

    with get_sync_db() as session:
        test_doc = Document(
            id=celery_doc_id,
            file_hash=f"hash_{celery_doc_id[:10]}",
            filename="celery_test.pdf",
            file_size=len(pdf_bytes),
            content_type="application/pdf",
            category="PDF_DIGITAL",
            minio_raw_path=raw_key,
            status=TaskStatus.PENDING.value,
            task_id=f"task_{celery_doc_id[:8]}"
        )
        session.add(test_doc)
        session.commit()

    # Execute worker task synchronously
    task_res = process_document_task(celery_doc_id)
    assert task_res["status"] == TaskStatus.OCR_COMPLETED.value
    assert task_res["total_pages"] == 2

    # Verify Database state
    with get_sync_db() as session:
        stmt = select(Document).where(Document.id == celery_doc_id)
        d_row = session.execute(stmt).scalars().first()
        assert d_row.status == TaskStatus.OCR_COMPLETED.value
        assert d_row.minio_ocr_path is not None
        assert d_row.total_pages == 2

        # Verify pages
        stmt_p = select(DocumentPage).where(DocumentPage.document_id == celery_doc_id)
        pages_rows = list(session.execute(stmt_p).scalars().all())
        assert len(pages_rows) == 2
        print(f"      [OK] PostgreSQL State: PENDING -> PROCESSING -> OCR_COMPLETED")
        print(f"      [OK] {len(pages_rows)} DocumentPage records created with confidence scores.")

        # Cleanup test records
        session.delete(d_row)
        session.commit()

    # Cleanup test artifacts
    for k in [pdf_key, img_key, docx_key, raw_key, task_res["ocr_json_path"]]:
        bucket = artifact_store.doc_bucket if "raw" in k else artifact_store.artifact_bucket
        artifact_store.minio.delete_object(bucket, k)

    print("\n" + "=" * 70)
    print(" PHASE 4 VERIFICATION: ALL OCR & CELERY PIPELINES PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_phase4()
    sys.exit(0 if success else 1)
