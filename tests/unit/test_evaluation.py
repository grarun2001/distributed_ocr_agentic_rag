"""Unit tests for OCR accuracy metrics (CER/WER) and Retrieval metrics (Recall/MRR)."""

import pytest
from evaluation.ocr_benchmarks import (
    calculate_cer,
    calculate_wer,
    evaluate_ocr_dataset,
    evaluate_ocr_text,
    levenshtein_distance,
)
from evaluation.retrieval_benchmarks import (
    QueryTestCase,
    calculate_metrics_for_mode,
    compare_retrieval_modes,
)


def test_levenshtein_distance():
    """Verify Levenshtein edit distance calculations."""
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("kitten", "kitten") == 0
    assert levenshtein_distance("kitten", "sitting") == 3  # k->s, e->i, +g
    assert levenshtein_distance(["hello", "world"], ["hello", "world"]) == 0
    assert levenshtein_distance(["hello", "world"], ["hello", "there", "world"]) == 1


def test_calculate_cer_and_wer_exact_match():
    """Verify 0.0 error rates on exact matching text."""
    text = "The quick brown fox jumps over the lazy dog."
    assert calculate_cer(text, text) == 0.0
    assert calculate_wer(text, text) == 0.0

    eval_res = evaluate_ocr_text(text, text)
    assert eval_res.cer == 0.0
    assert eval_res.wer == 0.0
    assert eval_res.char_accuracy == 1.0
    assert eval_res.word_accuracy == 1.0
    assert eval_res.passed is True


def test_calculate_cer_and_wer_with_errors():
    """Verify non-zero error rate detection on noisy OCR predictions."""
    ref = "Invoice total is 500 dollars"
    hyp = "Invoice total is 500 dollers"  # 1 char difference: 'a' -> 'e'

    cer = calculate_cer(ref, hyp)
    wer = calculate_wer(ref, hyp)

    assert cer > 0.0
    assert wer == 0.2  # 1 out of 5 words is wrong = 20% WER

    res = evaluate_ocr_text(ref, hyp, max_cer=0.10, max_wer=0.25)
    assert res.passed is True


def test_evaluate_ocr_dataset():
    """Verify dataset evaluation aggregates."""
    samples = [
        ("Perfect text matching.", "Perfect text matching."),
        ("Slight variation here.", "Slight variation here.")
    ]
    report = evaluate_ocr_dataset(samples)
    assert report["total_samples"] == 2
    assert report["mean_cer"] == 0.0
    assert report["mean_wer"] == 0.0
    assert report["pass_rate"] == 1.0


def test_retrieval_benchmarks_metrics():
    """Verify Recall@K, MRR, and Precision@K calculations."""
    test_cases = [
        QueryTestCase(query="q1", expected_ids={"chunk_a", "chunk_b"}),
        QueryTestCase(query="q2", expected_ids={"chunk_c"})
    ]

    # Mock search function returning known ranks
    def mock_search(query: str, k: int, doc_filter: str):
        if query == "q1":
            return ["chunk_x", "chunk_a", "chunk_y"]  # Hit at rank 2
        elif query == "q2":
            return ["chunk_c", "chunk_m", "chunk_n"]  # Hit at rank 1
        return []

    # Test K=1
    m1 = calculate_metrics_for_mode(test_cases, mock_search, mode="TEST_MODE", k=1)
    # q1 top 1 is chunk_x (miss), q2 top 1 is chunk_c (hit) -> 1/2 = 0.5 Recall@1
    assert m1.recall_at_k == 0.5
    # q1 reciprocal rank = 0, q2 reciprocal rank = 1.0 -> MRR = 0.5
    assert m1.mrr == 0.5

    # Test K=3
    m3 = calculate_metrics_for_mode(test_cases, mock_search, mode="TEST_MODE", k=3)
    # q1 hit at rank 2 (in top 3), q2 hit at rank 1 (in top 3) -> 2/2 = 1.0 Recall@3
    assert m3.recall_at_k == 1.0
    # q1 rank 2 (1/2 = 0.5), q2 rank 1 (1/1 = 1.0) -> MRR = (0.5 + 1.0)/2 = 0.75
    assert m3.mrr == 0.75


def test_compare_retrieval_modes():
    """Verify multi-mode comparison dispatch."""
    test_cases = [
        QueryTestCase(query="find documents", expected_ids={"doc_1"})
    ]

    def mock_dispatch(query: str, k: int, mode: str, doc_filter: str):
        if mode == "HYBRID_RERANK":
            return ["doc_1", "doc_2"]  # Rank 1 hit
        return ["doc_2", "doc_1"]      # Rank 2 hit

    comp = compare_retrieval_modes(
        test_cases=test_cases,
        search_engine_dispatch=mock_dispatch,
        modes=["VECTOR_ONLY", "HYBRID_RERANK"],
        k_values=[1, 2]
    )

    assert "VECTOR_ONLY" in comp["modes"]
    assert "HYBRID_RERANK" in comp["modes"]
    # At K=1, VECTOR_ONLY missed (doc_2), while HYBRID_RERANK hit (doc_1)
    assert comp["modes"]["HYBRID_RERANK"]["K=1"]["recall"] == 1.0
    assert comp["modes"]["VECTOR_ONLY"]["K=1"]["recall"] == 0.0
