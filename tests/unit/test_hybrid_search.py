"""Unit tests for Reciprocal Rank Fusion and Cross-Encoder reranker fallback."""

from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import CrossEncoderReranker
from app.schemas.retrieval import RetrievalHit


def test_reciprocal_rank_fusion_logic():
    """Verify RRF formula merges ranks correctly from dense and sparse results."""
    dense_hits = [
        {"document_id": "doc1", "filename": "a.pdf", "page_number": 1, "chunk_index": 0, "text": "Common chunk", "score": 0.9},
        {"document_id": "doc1", "filename": "a.pdf", "page_number": 1, "chunk_index": 1, "text": "Dense unique", "score": 0.8},
    ]
    sparse_hits = [
        {"document_id": "doc1", "filename": "a.pdf", "page_number": 1, "chunk_index": 0, "text": "Common chunk", "score": 10.5},
        {"document_id": "doc1", "filename": "a.pdf", "page_number": 1, "chunk_index": 2, "text": "Sparse unique", "score": 8.0},
    ]

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_k=3)

    assert len(fused) == 3
    # Common chunk ranked #1 in both lists must have the highest combined RRF score
    assert fused[0].chunk_id == "doc1_0"
    assert fused[0].text == "Common chunk"
    assert fused[0].retrieval_source == "RRF"
    assert fused[0].score == 1.0  # normalized max score


def test_reranker_fallback_on_failure():
    """Verify that if the Cross-Encoder fails, the pipeline safely falls back to RRF candidates."""
    reranker = CrossEncoderReranker(model_name="nonexistent_mock_model_123")
    # Simulate failed model load
    reranker._model = None

    candidates = [
        RetrievalHit(
            chunk_id="doc1_0",
            document_id="doc1",
            filename="a.pdf",
            page_number=1,
            chunk_index=0,
            text="High priority candidate",
            score=0.95,
            retrieval_source="RRF"
        ),
        RetrievalHit(
            chunk_id="doc1_1",
            document_id="doc1",
            filename="a.pdf",
            page_number=1,
            chunk_index=1,
            text="Secondary candidate",
            score=0.75,
            retrieval_source="RRF"
        )
    ]

    # Must return candidates without raising an exception
    results = reranker.rerank(query="test query", candidates=candidates, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "doc1_0"
    assert results[0].score == 0.95
