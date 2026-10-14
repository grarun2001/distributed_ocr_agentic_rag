"""Integration tests for hybrid retrieval, Milvus, persistent BM25, and neural reranking."""

import uuid
from app.indexing.bm25_store import PersistentBM25Store, get_bm25_store
from app.indexing.index_pipeline import get_indexing_pipeline
from app.indexing.milvus_store import get_milvus_store
from app.rag.hybrid_search import get_search_engine
from app.schemas.ocr import OCRDocumentResult, OCRPage


def create_test_ocr_result(doc_id: str) -> OCRDocumentResult:
    """Create a structured OCR result with distinct keyword and semantic passages."""
    p1 = OCRPage(
        page_number=1,
        ocr_engine="NATIVE",
        confidence=100.0,
        char_count=120,
        table_count=0,
        has_selectable_text=True,
        raw_text="The invoice total for Acme Corporation is $45,000 payable by December 2026.",
        blocks=[],
        tables=[],
        warnings=[]
    )
    p2 = OCRPage(
        page_number=2,
        ocr_engine="NATIVE",
        confidence=100.0,
        char_count=130,
        table_count=0,
        has_selectable_text=True,
        raw_text="Payment instructions require direct wire transfer to Bank Account 987654321.",
        blocks=[],
        tables=[],
        warnings=[]
    )
    return OCRDocumentResult(
        document_id=doc_id,
        filename="acme_invoice.pdf",
        category="PDF_DIGITAL",
        total_pages=2,
        average_confidence=100.0,
        ocr_engine_used="NATIVE",
        pages=[p1, p2],
        full_text=p1.raw_text + "\n\n" + p2.raw_text,
        warnings=[]
    )


def test_hybrid_rag_flow_and_persistence():
    """Verify dual indexing, vector + BM25 search, RRF, reranking, and disk persistence."""
    doc_id = str(uuid.uuid4())
    ocr_result = create_test_ocr_result(doc_id)

    # 1. Dual Indexing
    pipeline = get_indexing_pipeline()
    total_chunks = pipeline.index_ocr_result(ocr_result)
    assert total_chunks == 2

    search_engine = get_search_engine()

    # 2. Test BM25 Keyword Search
    bm25_hits = search_engine.search(query="Acme Corporation payable", top_k=2, mode="BM25_ONLY")
    assert len(bm25_hits) > 0
    assert "Acme Corporation" in bm25_hits[0].text
    assert bm25_hits[0].retrieval_source == "BM25"

    # 3. Test Milvus Dense Vector Search
    vector_hits = search_engine.search(query="How much money is due?", top_k=2, mode="VECTOR_ONLY")
    assert len(vector_hits) > 0
    assert vector_hits[0].retrieval_source == "DENSE"

    # 4. Test Hybrid RRF Search
    rrf_hits = search_engine.search(query="wire transfer bank account number", top_k=2, mode="HYBRID_RRF")
    assert len(rrf_hits) > 0
    assert rrf_hits[0].retrieval_source == "RRF"

    # 5. Test Cross-Encoder Reranking
    reranked_hits = search_engine.search(query="What is the account number?", top_k=2, mode="HYBRID_RERANK")
    assert len(reranked_hits) > 0
    assert "987654321" in reranked_hits[0].text

    # 6. Verify BM25 Disk Persistence: Fresh store instance reloads from disk
    bm25_store = get_bm25_store()
    fresh_bm25_store = PersistentBM25Store(storage_dir=bm25_store.storage_dir)
    reloaded_hits = fresh_bm25_store.search("987654321", top_k=1)
    assert len(reloaded_hits) > 0
    assert reloaded_hits[0]["document_id"] == doc_id
    assert "987654321" in reloaded_hits[0]["text"]

    # 7. Clean up
    milvus_store = get_milvus_store()
    milvus_store.delete_by_document_id(doc_id)
    bm25_store.delete_document(doc_id)
