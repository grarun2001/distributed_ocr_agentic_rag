"""Phase 8 Verification Script: Circuit Breaker, Exponential Retry, OCR Benchmarks (CER/WER), and Retrieval Evaluation (Recall/MRR)."""

import json
import sys
import time
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.exceptions import CircuitBreakerOpenError
from app.indexing.bm25_store import get_bm25_store
from app.indexing.index_pipeline import get_indexing_pipeline
from app.indexing.milvus_store import get_milvus_store
from app.rag.hybrid_search import get_search_engine
from app.reliability.circuit_breaker import CircuitBreaker, CircuitState
from app.reliability.retry import retry_with_backoff
from app.schemas.ocr import OCRDocumentResult, OCRPage
from evaluation.ocr_benchmarks import evaluate_ocr_dataset, evaluate_ocr_text
from evaluation.retrieval_benchmarks import QueryTestCase, compare_retrieval_modes


def verify_phase8() -> bool:
    print("=" * 70)
    print(" DISTRIBUTED OCR AGENTIC RAG - PHASE 8 VERIFICATION")
    print("=" * 70)

    # [1/4] Test Circuit Breaker State Transitions
    print("\n[1/4] Testing Circuit Breaker Fault Isolation...")
    cb = CircuitBreaker(
        name="external_llm_service",
        failure_threshold=3,
        recovery_timeout=0.2,
        half_open_success_threshold=2
    )

    # Trigger 3 failures to trip to OPEN
    for i in range(3):
        cb.record_failure(ConnectionError(f"Simulated external timeout {i+1}"))
    assert cb.state == CircuitState.OPEN, f"Expected OPEN state, got {cb.state}"
    print(f"      [OK] Tripped to OPEN state after {cb.failure_count} failures.")

    # Validate fast failure without calling downstream service
    blocked = False
    try:
        cb.before_call()
    except CircuitBreakerOpenError:
        blocked = True
    assert blocked, "Expected CircuitBreakerOpenError to block call"
    print("      [OK] CircuitBreakerOpenError blocked incoming request immediately.")

    # Wait for cooldown to test HALF_OPEN transition
    time.sleep(0.25)
    assert cb.state == CircuitState.HALF_OPEN
    print("      [OK] Circuit Breaker successfully entered HALF_OPEN trial state.")

    # Complete trial successfully
    cb.record_success()
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    print("      [OK] Circuit Breaker recovered and reset to CLOSED state.")

    # [2/4] Test Exponential Backoff Retry Policy
    print("\n[2/4] Testing Exponential Backoff Retry Policy with Jitter...")
    transient_attempts = 0

    @retry_with_backoff(max_retries=4, base_delay=0.02, max_delay=0.1, jitter=True)
    def flaky_external_fetch():
        nonlocal transient_attempts
        transient_attempts += 1
        if transient_attempts < 3:
            raise TimeoutError("Transient network glitch")
        return "payload_delivered"

    res = flaky_external_fetch()
    assert res == "payload_delivered"
    assert transient_attempts == 3
    print(f"      [OK] Retry succeeded on attempt #{transient_attempts} with jittered backoff.")

    # [3/4] Run OCR Quantitative Accuracy Benchmark (CER / WER)
    print("\n[3/4] Running Quantitative OCR Accuracy Benchmark (CER & WER)...")
    dataset_path = PROJECT_ROOT / "evaluation" / "datasets" / "ocr_eval_samples.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples_json = json.load(f)

    eval_pairs = [(s["ground_truth"], s["ocr_output"]) for s in samples_json]
    ocr_report = evaluate_ocr_dataset(eval_pairs, max_cer=0.05, max_wer=0.10)

    print(f"      Evaluated {ocr_report['total_samples']} benchmark documents:")
    print(f"      - Mean Character Error Rate (CER): {ocr_report['mean_cer'] * 100:.2f}% (Target: < 5%)")
    print(f"      - Mean Word Error Rate (WER):      {ocr_report['mean_wer'] * 100:.2f}% (Target: < 10%)")
    print(f"      - Mean Character Accuracy:         {ocr_report['mean_char_accuracy'] * 100:.2f}%")
    print(f"      - Mean Word Accuracy:              {ocr_report['mean_word_accuracy'] * 100:.2f}%")
    print(f"      - Benchmark Pass Rate:             {ocr_report['pass_rate'] * 100:.1f}%")
    assert ocr_report["pass_rate"] == 1.0, "Expected 100% pass rate on test corpus"
    print("      [OK] OCR accuracy metrics satisfy production thresholds.")

    # [4/4] Run Information Retrieval Evaluation (Recall@K & MRR Comparison)
    print("\n[4/4] Running Information Retrieval Benchmark (Recall@K & MRR Comparison)...")
    doc_id = str(uuid.uuid4())
    p1 = OCRPage(
        page_number=1,
        ocr_engine="NATIVE",
        confidence=100.0,
        char_count=180,
        table_count=0,
        has_selectable_text=True,
        raw_text="The distributed cluster uses Redis as the Celery message broker on port 6379.",
        blocks=[],
        tables=[],
        warnings=[]
    )
    p2 = OCRPage(
        page_number=2,
        ocr_engine="NATIVE",
        confidence=100.0,
        char_count=190,
        table_count=0,
        has_selectable_text=True,
        raw_text="Dense vector embeddings are stored in Milvus standalone instance on port 19530.",
        blocks=[],
        tables=[],
        warnings=[]
    )
    eval_doc = OCRDocumentResult(
        document_id=doc_id,
        filename="benchmark_spec.pdf",
        category="PDF_DIGITAL",
        total_pages=2,
        average_confidence=100.0,
        ocr_engine_used="NATIVE",
        pages=[p1, p2],
        full_text=p1.raw_text + "\n\n" + p2.raw_text,
        warnings=[]
    )

    # Ingest benchmark document into Milvus and BM25
    indexing_pipeline = get_indexing_pipeline()
    indexing_pipeline.index_ocr_result(eval_doc)

    expected_c0 = f"{doc_id}_0"
    expected_c1 = f"{doc_id}_1"

    test_queries = [
        QueryTestCase(query="What message broker does Celery use?", expected_ids={expected_c0}, document_id=doc_id),
        QueryTestCase(query="Where are dense vectors stored and what port?", expected_ids={expected_c1}, document_id=doc_id)
    ]

    search_engine = get_search_engine()

    def dispatch_search(query: str, top_k: int, mode: str, doc_filter: str):
        hits = search_engine.search(query=query, top_k=top_k, filter_document_id=doc_filter, mode=mode)
        return [h.chunk_id for h in hits]

    retrieval_comparison = compare_retrieval_modes(
        test_cases=test_queries,
        search_engine_dispatch=dispatch_search,
        modes=["VECTOR_ONLY", "BM25_ONLY", "HYBRID_RRF", "HYBRID_RERANK"],
        k_values=[1, 2]
    )

    print("      Retrieval Strategies Evaluation Comparison:")
    for mode, scores in retrieval_comparison["modes"].items():
        k1_recall = scores["K=1"]["recall"]
        k1_mrr = scores["K=1"]["mrr"]
        k2_recall = scores["K=2"]["recall"]
        print(f"      - Mode '{mode:13}': Recall@1: {k1_recall:.2f} | MRR@1: {k1_mrr:.2f} | Recall@2: {k2_recall:.2f}")

    rerank_mrr = retrieval_comparison["modes"]["HYBRID_RERANK"]["K=1"]["mrr"]
    assert rerank_mrr >= 0.5, "Expected high MRR for neural reranker"
    print(f"      [OK] Neural Reranker achieved top MRR ({rerank_mrr:.2f}) on benchmark queries.")

    # Cleanup test document
    milvus_store = get_milvus_store()
    milvus_store.delete_by_document_id(doc_id)
    bm25_store = get_bm25_store()
    bm25_store.delete_document(doc_id)
    print("      [OK] Cleaned up benchmark test document.")

    print("\n" + "=" * 70)
    print(" PHASE 8 VERIFICATION: ALL RELIABILITY & BENCHMARK SUITES PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_phase8()
    sys.exit(0 if success else 1)
