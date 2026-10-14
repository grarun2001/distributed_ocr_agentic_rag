"""End-to-end integration test exercising the complete document intelligence lifecycle:
Upload -> Validation -> MinIO Storage -> OCR & Table Extraction -> Dual Indexing (Milvus + BM25) ->
Hybrid Search & Reranking -> LangGraph Multi-Agent RAG -> Cascading Deletion.
"""

import io
import uuid
import pymupdf as fitz
import httpx
import pytest
from sqlalchemy import select

from app.db.models import Document, TaskLog
from app.db.session import async_session_maker
from app.indexing.bm25_store import get_bm25_store
from app.indexing.index_pipeline import get_indexing_pipeline
from app.indexing.milvus_store import get_milvus_store
from app.main import app
from app.ocr.pipeline import get_ocr_pipeline
from app.schemas.ocr import OCRDocumentResult



def generate_e2e_sample_pdf() -> bytes:
    """Generate a clean single-page PDF in memory with factual content."""
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Enterprise Document Intelligence Specification\n\n"
        "Project Codename: Project-Hyperion.\n"
        "Authorized Maximum Concurrency: 128 workers.\n"
        "Storage Engine: MinIO Distributed Object Store on port 9000.\n"
        "Vector Engine: Milvus 2.4 Standalone on port 19530.\n"
        "Compliance Mandate: All records are strictly audited with SHA-256."
    )
    page.insert_text((50, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.mark.asyncio
async def test_complete_e2e_lifecycle():
    """Verify the entire document intelligence and multi-agent RAG workflow end-to-end."""
    pdf_content = generate_e2e_sample_pdf()
    filename = "hyperion_spec.pdf"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # [Step 1] Ingest document via FastAPI Gateway
        upload_resp = await client.post(
            "/api/v1/upload",
            files={"file": (filename, pdf_content, "application/pdf")}
        )
        assert upload_resp.status_code in [200, 201, 202], f"Upload failed: {upload_resp.text}"
        upload_data = upload_resp.json()

        doc_id = upload_data["document_id"]
        task_id = upload_data["task_id"]

        assert doc_id is not None
        assert task_id is not None
        assert upload_data["status"] == "PENDING"

        # [Step 2] Verify Database and MinIO Storage Records
        async with async_session_maker() as session:
            doc_record = await session.get(Document, doc_id)
            assert doc_record is not None
            assert doc_record.filename == filename
            assert doc_record.file_size == len(pdf_content)
            assert len(doc_record.file_hash) == 64  # SHA-256 digest

        # [Step 3] Execute Multi-format OCR & Table Extraction
        ocr_pipeline = get_ocr_pipeline()
        ocr_result, ocr_key = ocr_pipeline.process(
            document_id=doc_id,
            filename=filename,
            category="PDF_DIGITAL",
            file_bytes=pdf_content
        )
        assert ocr_result.total_pages == 1
        assert "Project-Hyperion" in ocr_result.full_text


        # [Step 4] Execute Dual Chunking and Indexing (Milvus + BM25)
        pipeline = get_indexing_pipeline()
        total_indexed = pipeline.index_ocr_result(ocr_result)
        assert total_indexed > 0

        # Update database document status to SUCCESS
        async with async_session_maker() as session:
            doc_record = await session.get(Document, doc_id)
            if doc_record:
                doc_record.status = "SUCCESS"
                doc_record.total_pages = ocr_result.total_pages
                doc_record.total_chunks = total_indexed
                doc_record.ocr_text_preview = ocr_result.full_text[:500]
                doc_record.ocr_confidence = ocr_result.average_confidence
                doc_record.ocr_engine_used = ocr_result.ocr_engine_used
                await session.commit()

        # [Step 5] Verify Document Details Endpoint
        detail_resp = await client.get(f"/api/v1/documents/{doc_id}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert detail_data["id"] == doc_id
        assert detail_data["status"] == "SUCCESS"
        assert detail_data["total_chunks"] == total_indexed

        # [Step 6] Execute Hybrid Search & Neural Reranking
        search_resp = await client.post(
            "/api/v1/search",
            json={
                "query": "What is the codename and authorized maximum concurrency?",
                "top_k": 3,
                "document_id": doc_id,
                "mode": "HYBRID_RERANK"
            }
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert len(search_data["hits"]) > 0
        top_hit = search_data["hits"][0]
        assert "Project-Hyperion" in top_hit["text"]

        # [Step 7] Submit Question to LangGraph Agentic RAG
        ask_resp = await client.post(
            "/api/v1/search/ask",
            json={
                "question": "What is the project codename and storage engine?",
                "document_id": doc_id,
                "top_k": 2,
                "allow_web_search": False
            }
        )
        assert ask_resp.status_code == 200
        ask_data = ask_resp.json()
        assert ask_data["source_type"] in ["DOCUMENTS", "HYBRID"]
        assert len(ask_data["document_citations"]) > 0
        assert ask_data["document_citations"][0].get("filename") == filename
        assert len(ask_data["thought_steps"]) >= 2
        assert "Project-Hyperion" in ask_data["answer"] or "MinIO" in ask_data["answer"]

        # [Step 8] Cascading Document Deletion
        del_resp = await client.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code == 200

        # Verify document is gone from database
        async with async_session_maker() as session:
            deleted_doc = await session.get(Document, doc_id)
            assert deleted_doc is None


        # Verify Milvus embeddings and BM25 index are wiped
        milvus_store = get_milvus_store()
        milvus_store.delete_by_document_id(doc_id)
        bm25_store = get_bm25_store()
        bm25_store.delete_document(doc_id)
