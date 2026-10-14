"""Integration tests for FastAPI file upload, deduplication, and document management."""

import pytest
import hashlib
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Verify system health endpoint returns connected for all services."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["database"] == "connected"
        assert data["services"]["redis"] == "connected"
        assert data["services"]["minio"] == "connected"
        assert data["services"]["milvus"] == "connected"


@pytest.mark.asyncio
async def test_upload_and_deduplication_flow():
    """Verify single file upload, SHA-256 calculation, and duplicate detection."""
    sample_content = b"%PDF-1.4\n1 0 obj\n<< /Title (Test PDF) >>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 >>\nstartxref\n9\n%%EOF"
    expected_hash = hashlib.sha256(sample_content).hexdigest()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # 1. First upload (New Document)
        files = {"file": ("test_sample.pdf", sample_content, "application/pdf")}
        res1 = await client.post("/api/v1/upload", files=files)
        assert res1.status_code == 201
        data1 = res1.json()
        doc_id = data1["document_id"]
        assert data1["file_hash"] == expected_hash
        assert data1["is_duplicate"] is False
        assert data1["status"] == "PENDING"
        assert data1["category"] == "PDF_DIGITAL"

        # 2. Second upload (Identical File -> Deduplication)
        files_dup = {"file": ("test_sample_copy.pdf", sample_content, "application/pdf")}
        res2 = await client.post("/api/v1/upload", files=files_dup)
        assert res2.status_code == 201
        data2 = res2.json()
        assert data2["document_id"] == doc_id
        assert data2["is_duplicate"] is True
        assert data2["file_hash"] == expected_hash

        # 3. Retrieve Document Details
        res3 = await client.get(f"/api/v1/documents/{doc_id}")
        assert res3.status_code == 200
        data3 = res3.json()
        assert data3["id"] == doc_id
        assert data3["raw_file_url"] is not None
        assert "http" in data3["raw_file_url"]

        # 4. Check Task Status
        task_id = data1["task_id"]
        res4 = await client.get(f"/api/v1/tasks/{task_id}")
        assert res4.status_code == 200
        data4 = res4.json()
        assert data4["document_id"] == doc_id
        assert len(data4["logs"]) > 0

        # 5. Clean up Document
        res5 = await client.delete(f"/api/v1/documents/{doc_id}")
        assert res5.status_code == 200
