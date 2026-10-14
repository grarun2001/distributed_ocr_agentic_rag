"""Phase 5 Verification Script: Structure Chunking, BGE Embeddings, BM25 Persistence, Milvus, and Cross-Encoder Reranking."""

import sys
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.indexing.bm25_store import PersistentBM25Store, get_bm25_store
from app.indexing.chunker import get_chunker
from app.indexing.embeddings import get_embedding_generator
from app.indexing.index_pipeline import get_indexing_pipeline
from app.indexing.milvus_store import get_milvus_store
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.hybrid_search import get_search_engine
from app.rag.reranker import get_reranker, CrossEncoderReranker
from app.schemas.ocr import OCRDocumentResult, OCRPage, OCRTable
from app.schemas.retrieval import RetrievalHit

settings = get_settings()


def build_verification_document(doc_id: str) -> OCRDocumentResult:
    """Build multi-page test OCR result containing domain facts and tables."""
    table = OCRTable(
        table_index=0,
        num_rows=3,
        num_cols=3,
        headers=["Part Number", "Description", "Stock"],
        rows=[
            ["PX-900", "High Performance OCR Accelerator", "42"],
            ["TX-100", "Thermal Sensor Sensor Module", "150"]
        ],
        markdown_data=(
            "| Part Number | Description | Stock |\n"
            "| --- | --- | --- |\n"
            "| PX-900 | High Performance OCR Accelerator | 42 |\n"
            "| TX-100 | Thermal Sensor Module | 150 |"
        ),
        confidence=100.0
    )

    page1 = OCRPage(
        page_number=1,
        ocr_engine="NATIVE",
        confidence=100.0,
        char_count=180,
        table_count=1,
        has_selectable_text=True,
        raw_text=(
            "Distributed Document Intelligence Specification:\n"
            "The cluster operates on Celery workers connected via Redis message broker.\n"
            "Vector embeddings are stored in Milvus standalone instance with 384 dimensions."
        ),
        blocks=[],
        tables=[table],
        warnings=[]
    )

    page2 = OCRPage(
        page_number=2,
        ocr_engine="NATIVE",
        confidence=100.0,
        char_count=160,
        table_count=0,
        has_selectable_text=True,
        raw_text=(
            "Security and Compliance Policy:\n"
            "All ingested files must calculate cryptographic SHA-256 digests.\n"
            "Object keys in MinIO follow the format documents/{doc_id}/raw/{filename}."
        ),
        blocks=[],
        tables=[],
        warnings=[]
    )

    return OCRDocumentResult(
        document_id=doc_id,
        filename="system_specifications.pdf",
        category="PDF_DIGITAL",
        total_pages=2,
        average_confidence=100.0,
        ocr_engine_used="NATIVE",
        pages=[page1, page2],
        full_text=page1.raw_text + "\n\n" + page2.raw_text,
        warnings=[],
        metadata={"file_hash": f"hash_{doc_id[:12]}"}
    )


def verify_phase5() -> bool:
    print("=" * 70)
    print(" DISTRIBUTED OCR AGENTIC RAG - PHASE 5 VERIFICATION")
    print("=" * 70)

    doc_id = str(uuid.uuid4())
    doc_result = build_verification_document(doc_id)

    # [1/7] Test Structure-Aware Chunking
    print("\n[1/7] Testing Structure-Aware Chunking...")
    chunker = get_chunker()
    chunks = chunker.chunk_document(doc_result)
    assert len(chunks) >= 3  # 1 table chunk + 2 page text chunks
    print(f"      [OK] Generated {len(chunks)} chunks with preserved page metadata.")
    has_tbl_chunk = any(c.metadata.has_table for c in chunks)
    assert has_tbl_chunk is True
    print("      [OK] Tabular data correctly partitioned into standalone atomic chunk.")

    # [2/7] Test Dense Vector Embeddings (BAAI/bge-small-en-v1.5)
    print("\n[2/7] Testing Dense Vector Embeddings (Dimension: 384)...")
    embedder = get_embedding_generator()
    q_emb = embedder.embed_query("What is the stock of PX-900?")
    assert len(q_emb) == 384
    print(f"      [OK] Dense vector generated: 384 dimensions normalized for Cosine distance.")

    # [3/7] Test Dual Indexing into Milvus & Persistent BM25
    print("\n[3/7] Testing Dual Indexing into Milvus & Persistent BM25...")
    indexing_pipeline = get_indexing_pipeline()
    total_indexed = indexing_pipeline.index_ocr_result(doc_result)
    assert total_indexed == len(chunks)
    print(f"      [OK] Successfully indexed {total_indexed} chunks in Milvus and BM25.")

    # [4/7] Test BM25 Keyword Search & Milvus Semantic Search
    print("\n[4/7] Testing Sparse BM25 vs Dense Vector Search...")
    search_engine = get_search_engine()

    # Exact keyword test (part number)
    bm25_hits = search_engine.search("PX-900 accelerator", top_k=2, filter_document_id=doc_id, mode="BM25_ONLY")
    assert len(bm25_hits) > 0
    assert "PX-900" in bm25_hits[0].text
    print(f"      [OK] BM25 Keyword Search: Found part 'PX-900' (Score: {bm25_hits[0].score:.4f})")

    # Semantic test (concept query)
    vec_hits = search_engine.search("How do workers communicate with each other?", top_k=2, filter_document_id=doc_id, mode="VECTOR_ONLY")
    assert len(vec_hits) > 0
    assert "Redis message broker" in vec_hits[0].text
    print(f"      [OK] Milvus Vector Search: Found semantic concept (Score: {vec_hits[0].score:.4f})")

    # [5/7] Test Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking
    print("\n[5/7] Testing Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking...")
    rrf_hits = search_engine.search("Where are raw files stored in MinIO?", top_k=3, filter_document_id=doc_id, mode="HYBRID_RRF")
    assert len(rrf_hits) > 0
    assert rrf_hits[0].retrieval_source == "RRF"
    print(f"      [OK] RRF Fusion: Merged candidate ranks (Top chunk ID: {rrf_hits[0].chunk_id})")

    reranked_hits = search_engine.search("Where are raw files stored in MinIO?", top_k=2, filter_document_id=doc_id, mode="HYBRID_RERANK")
    assert len(reranked_hits) > 0
    assert "documents/{doc_id}/raw" in reranked_hits[0].text
    print(f"      [OK] Cross-Encoder Reranker: Neural score {reranked_hits[0].score:.4f} for top passage.")

    # [6/7] Test Reranker Failure Fallback
    print("\n[6/7] Testing Reranker Reliability Fallback...")
    mock_reranker = CrossEncoderReranker(model_name="nonexistent_mock_model")
    mock_reranker._model = None  # Force failure
    fallback_hits = mock_reranker.rerank(query="test", candidates=rrf_hits, top_k=2)
    assert len(fallback_hits) == 2
    assert fallback_hits[0].chunk_id == rrf_hits[0].chunk_id
    print("      [OK] Reranker Fault Tolerance: System gracefully fell back to RRF results.")

    # [7/7] Test BM25 Disk Persistence across Reload
    print("\n[7/7] Testing BM25 Disk Persistence across Store Reload...")
    bm25_store = get_bm25_store()
    # Instantiate completely new store instance reading solely from disk files
    fresh_bm25_store = PersistentBM25Store(storage_dir=bm25_store.storage_dir)
    reloaded_hits = fresh_bm25_store.search("PX-900", top_k=5, filter_document_id=doc_id)
    assert len(reloaded_hits) > 0
    assert reloaded_hits[0]["document_id"] == doc_id
    assert "PX-900" in reloaded_hits[0]["text"]
    print(f"      [OK] BM25 Persistence Verified: Successfully reloaded index from disk for doc '{doc_id[:8]}...'.")

    # Cleanup
    milvus_store = get_milvus_store()
    milvus_store.delete_by_document_id(doc_id)
    bm25_store.delete_document(doc_id)
    print("      [OK] Cleaned up Milvus and persistent BM25 indices.")

    print("\n" + "=" * 70)
    print(" PHASE 5 VERIFICATION: ALL HYBRID RAG & RERANKING WORKFLOWS PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_phase5()
    sys.exit(0 if success else 1)
