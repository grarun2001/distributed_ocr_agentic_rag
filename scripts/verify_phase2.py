"""Phase 2 Live Integration Verification Script.

Validates end-to-end CRUD operations on PostgreSQL, Redis, MinIO, and Milvus.
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.db.session import async_session_maker
from app.db.models import Document, TaskLog, DocumentPage
from app.db.repositories import DocumentRepository, TaskLogRepository
from app.storage.minio_client import get_minio_client
from app.storage.artifact_store import get_artifact_store
from app.indexing.milvus_store import get_milvus_store
import redis

settings = get_settings()


async def verify_postgres_async() -> bool:
    print("\n[1/4] Testing PostgreSQL Async Database Operations...")
    test_id = str(uuid.uuid4())
    test_hash = f"test_hash_{uuid.uuid4().hex}"

    try:
        async with async_session_maker() as session:
            repo = DocumentRepository(session)
            doc = Document(
                id=test_id,
                file_hash=test_hash,
                filename="phase2_test_document.pdf",
                file_size=1024,
                content_type="application/pdf",
                category="PDF_DIGITAL",
                minio_raw_path=f"documents/{test_id}/raw/test.pdf",
                status="PENDING",
                task_id=f"task_{test_id[:8]}"
            )
            created_doc = await repo.create(doc)
            await session.commit()
            print(f"      [OK] Created document record (ID: {created_doc.id[:8]}...)")

            # Query back
            queried = await repo.get_by_id(test_id)
            assert queried is not None and queried.file_hash == test_hash
            print(f"      [OK] Successfully queried document by ID (Status: {queried.status})")

            # Update status
            updated = await repo.update_status(test_id, "SUCCESS")
            await session.commit()
            assert updated.status == "SUCCESS"
            print(f"      [OK] Updated document status to SUCCESS")

            # Cleanup
            deleted = await repo.delete(test_id)
            await session.commit()
            assert deleted is True
            print(f"      [OK] Cleaned up test document record")

        print("      --> SUCCESS: PostgreSQL CRUD operations verified.")
        return True
    except Exception as exc:
        print(f"      --> ERROR: PostgreSQL verification failed: {exc}")
        return False


def verify_redis() -> bool:
    print("\n[2/4] Testing Redis Connection & Broker Operations...")
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)
        ping_res = r.ping()
        assert ping_res is True
        print("      [OK] Redis PING successful")

        test_key = f"phase2_test_key_{uuid.uuid4().hex[:6]}"
        r.set(test_key, "distributed_ocr_active", ex=30)
        val = r.get(test_key)
        assert val == b"distributed_ocr_active"
        print(f"      [OK] Redis SET/GET with TTL verified (Value: {val.decode('utf-8')})")

        r.delete(test_key)
        print("      [OK] Redis key cleaned up")
        print("      --> SUCCESS: Redis message broker & cache verified.")
        return True
    except Exception as exc:
        print(f"      --> ERROR: Redis verification failed: {exc}")
        return False


def verify_minio() -> bool:
    print("\n[3/4] Testing MinIO Object Storage Operations...")
    try:
        store = get_artifact_store()
        test_doc_id = f"test_{uuid.uuid4().hex[:8]}"
        sample_bytes = b"%PDF-1.4 Mock PDF content for Phase 2 verification."

        # Upload raw document
        key = store.save_raw_document(test_doc_id, "verify.pdf", sample_bytes, "application/pdf")
        print(f"      [OK] Uploaded raw document to MinIO (Key: {key})")

        # Download raw document
        downloaded = store.get_raw_document(key)
        assert downloaded == sample_bytes
        print(f"      [OK] Downloaded and verified content integrity ({len(downloaded)} bytes)")

        # Upload structured OCR JSON
        mock_ocr = {
            "document_id": test_doc_id,
            "filename": "verify.pdf",
            "total_pages": 1,
            "pages": [{"page_number": 1, "text": "Mock OCR extracted text", "confidence": 98.5}]
        }
        ocr_key = store.save_ocr_result(test_doc_id, mock_ocr)
        print(f"      [OK] Stored OCR JSON artifact (Key: {ocr_key})")

        retrieved_ocr = store.get_ocr_result(test_doc_id)
        assert retrieved_ocr["total_pages"] == 1
        print("      [OK] Retrieved and parsed OCR JSON artifact")

        # Generate presigned URL
        url = store.minio.get_presigned_url(store.doc_bucket, key, expires_seconds=600)
        assert "http" in url
        print(f"      [OK] Generated valid presigned URL")

        # Cleanup
        store.minio.delete_object(store.doc_bucket, key)
        store.minio.delete_object(store.artifact_bucket, ocr_key)
        print("      [OK] Cleaned up MinIO test objects")
        print("      --> SUCCESS: MinIO object storage verified.")
        return True
    except Exception as exc:
        print(f"      --> ERROR: MinIO verification failed: {exc}")
        return False


def verify_milvus() -> bool:
    print("\n[4/4] Testing Milvus Vector Store Operations...")
    try:
        store = get_milvus_store()
        col = store.ensure_collection()
        print(f"      [OK] Connected to Milvus (Collection: '{col.name}')")

        test_doc_id = f"test_milvus_{uuid.uuid4().hex[:6]}"
        dummy_vector = [0.05] * settings.MILVUS_EMBEDDING_DIM

        # Insert dummy chunk
        chunks = [{
            "page_number": 1,
            "chunk_index": 0,
            "text": "This is a test vector embedding for Phase 2 validation.",
            "vector": dummy_vector
        }]
        num_inserted = store.insert_chunks(
            document_id=test_doc_id,
            file_hash="mock_hash_for_test",
            filename="test_vector.txt",
            chunks_data=chunks
        )
        assert num_inserted == 1
        print(f"      [OK] Inserted 384-dim test vector entity into Milvus")

        # Dense nearest neighbor search
        hits = store.search_dense(query_vector=dummy_vector, top_k=2)
        assert len(hits) > 0
        print(f"      [OK] Executed dense search (Top match score: {hits[0]['score']:.4f})")

        # Cleanup
        store.delete_by_document_id(test_doc_id)
        print("      [OK] Deleted test vector from collection")
        print("      --> SUCCESS: Milvus vector operations verified.")
        return True
    except Exception as exc:
        print(f"      --> ERROR: Milvus verification failed: {exc}")
        return False


async def async_main() -> int:
    print("=" * 70)
    print(" DISTRIBUTED OCR AGENTIC RAG - PHASE 2 INTEGRATION VERIFICATION")
    print("=" * 70)

    pg_ok = await verify_postgres_async()
    redis_ok = verify_redis()
    minio_ok = verify_minio()
    milvus_ok = verify_milvus()

    print("\n" + "=" * 70)
    if pg_ok and redis_ok and minio_ok and milvus_ok:
        print(" PHASE 2 VERIFICATION: ALL 4 STORAGE & QUEUE SYSTEMS OPERATIONAL!")
        print("=" * 70)
        return 0
    else:
        print(" PHASE 2 VERIFICATION: FAILURES DETECTED. CHECK LOGS ABOVE.")
        print("=" * 70)
        return 1


def main() -> None:
    code = asyncio.run(async_main())
    sys.exit(code)


if __name__ == "__main__":
    main()
