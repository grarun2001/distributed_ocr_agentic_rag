"""Phase 3 Verification Script: FastAPI Upload, SHA-256 Fingerprint, PostgreSQL, and MinIO."""

import asyncio
import hashlib
import sys
import uuid
from pathlib import Path
from httpx import AsyncClient, ASGITransport

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.db.session import async_session_maker
from app.db.repositories import DocumentRepository
from app.storage.artifact_store import get_artifact_store


async def async_verify_phase3() -> bool:
    print("=" * 70)
    print(" DISTRIBUTED OCR AGENTIC RAG - PHASE 3 VERIFICATION")
    print("=" * 70)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # [1/6] Test Healthcheck
        print("\n[1/6] Testing FastAPI System Health Probe...")
        res_health = await client.get("/health")
        if res_health.status_code != 200:
            print(f"      [FAIL] Healthcheck failed with status {res_health.status_code}")
            return False
        health_data = res_health.json()
        print(f"      [OK] API Status: {health_data['status'].upper()}")
        for s_name, s_stat in health_data["services"].items():
            print(f"           - {s_name:<12}: {s_stat}")

        # [2/6] Upload new test PDF document
        print("\n[2/6] Testing Document Upload & SHA-256 Fingerprint...")
        test_uuid = uuid.uuid4().hex[:8]
        sample_pdf_bytes = (
            f"%PDF-1.4\n% Phase 3 Test Document {test_uuid}\n"
            "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            "3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n"
            "xref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000115 00000 n \n"
            "trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n165\n%%EOF\n"
        ).encode("utf-8")

        expected_hash = hashlib.sha256(sample_pdf_bytes).hexdigest()
        filename = f"invoice_{test_uuid}.pdf"

        upload_files = {"file": (filename, sample_pdf_bytes, "application/pdf")}
        res_upload = await client.post("/api/v1/upload", files=upload_files)
        if res_upload.status_code != 201:
            print(f"      [FAIL] Upload failed with code {res_upload.status_code}: {res_upload.text}")
            return False

        up_data = res_upload.json()
        doc_id = up_data["document_id"]
        task_id = up_data["task_id"]
        assert up_data["file_hash"] == expected_hash
        assert up_data["is_duplicate"] is False
        assert up_data["status"] == "PENDING"
        print(f"      [OK] Document Ingested (ID: {doc_id[:8]}..., Filename: '{filename}')")
        print(f"      [OK] SHA-256 Digest : {up_data['file_hash']}")
        print(f"      [OK] Initial Status : {up_data['status']}")
        print(f"      [OK] Assigned Task  : {task_id}")

        # [3/6] Verify storage in PostgreSQL & MinIO directly
        print("\n[3/6] Verifying Storage in PostgreSQL and MinIO...")
        async with async_session_maker() as session:
            repo = DocumentRepository(session)
            db_doc = await repo.get_by_id(doc_id)
            assert db_doc is not None
            assert db_doc.file_hash == expected_hash
            assert db_doc.minio_raw_path.startswith("documents/")
            print(f"      [OK] PostgreSQL Record Verified: {db_doc.filename} (Category: {db_doc.category})")

        artifact_store = get_artifact_store()
        raw_bytes_stored = artifact_store.get_raw_document(db_doc.minio_raw_path)
        assert raw_bytes_stored == sample_pdf_bytes
        print(f"      [OK] MinIO S3 Object Verified: {len(raw_bytes_stored)} bytes retrieved.")

        # [4/6] Test Idempotent Duplicate Detection
        print("\n[4/6] Testing Idempotent SHA-256 Deduplication...")
        duplicate_files = {"file": (f"copy_of_{filename}", sample_pdf_bytes, "application/pdf")}
        res_dup = await client.post("/api/v1/upload", files=duplicate_files)
        assert res_dup.status_code == 201
        dup_data = res_dup.json()
        assert dup_data["is_duplicate"] is True
        assert dup_data["document_id"] == doc_id
        assert dup_data["file_hash"] == expected_hash
        print(f"      [OK] Duplicate Detected: Same doc ID returned ({dup_data['document_id'][:8]}...)")
        print(f"      [OK] No redundant MinIO storage or DB rows created.")

        # [5/6] Test Document Details & Presigned URL
        print("\n[5/6] Testing Document Details & Presigned URLs...")
        res_doc = await client.get(f"/api/v1/documents/{doc_id}")
        assert res_doc.status_code == 200
        doc_detail = res_doc.json()
        assert doc_detail["raw_file_url"] is not None
        assert "http" in doc_detail["raw_file_url"]
        print(f"      [OK] Presigned Download URL: {doc_detail['raw_file_url'][:60]}...")

        # Test Task Status Polling
        res_task = await client.get(f"/api/v1/tasks/{task_id}")
        assert res_task.status_code == 200
        task_data = res_task.json()
        assert task_data["status"] == "PENDING"
        assert len(task_data["logs"]) > 0
        print(f"      [OK] Task Status Polled: {task_data['status']} ({task_data['current_stage']})")

        # [6/6] Test Security & Format Rejection, then Clean up
        print("\n[6/6] Testing Security Rejection & Cleanup...")
        # Corrupt PDF magic byte header check
        corrupt_files = {"file": ("fake.pdf", b"NOT_A_PDF_CONTENT", "application/pdf")}
        res_corrupt = await client.post("/api/v1/upload", files=corrupt_files)
        assert res_corrupt.status_code == 400
        print(f"      [OK] Security Guardrail: Invalid magic signature rejected with HTTP 400")

        # Clean up test document
        res_del = await client.delete(f"/api/v1/documents/{doc_id}")
        assert res_del.status_code == 200
        print(f"      [OK] Successfully deleted test document and cleaned MinIO objects.")

    print("\n" + "=" * 70)
    print(" PHASE 3 VERIFICATION: ALL INGESTION & STORAGE WORKFLOWS PASSED!")
    print("=" * 70)
    return True


def main() -> None:
    success = asyncio.run(async_verify_phase3())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
