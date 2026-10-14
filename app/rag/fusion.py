"""Reciprocal Rank Fusion (RRF) for merging dense and sparse retrieval results."""

from typing import Any, Dict, List
from app.core.constants import DEFAULT_RRF_K
from app.schemas.retrieval import RetrievalHit


def reciprocal_rank_fusion(
    dense_hits: List[Dict[str, Any]],
    sparse_hits: List[Dict[str, Any]],
    k: int = DEFAULT_RRF_K,
    top_k: int = 10
) -> List[RetrievalHit]:
    """
    Merge and rank candidate hits from dense Milvus and sparse BM25 search.
    Formula: RRF_score(d) = sum(1 / (k + rank_m(d)))
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # 1. Process Dense Vector Hits (1-indexed ranks)
    for rank, hit in enumerate(dense_hits, start=1):
        chunk_id = f"{hit['document_id']}_{hit['chunk_index']}"
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
        chunk_map[chunk_id] = hit

    # 2. Process Sparse BM25 Hits (1-indexed ranks)
    for rank, hit in enumerate(sparse_hits, start=1):
        chunk_id = f"{hit['document_id']}_{hit['chunk_index']}"
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = hit

    # 3. Sort by aggregated RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    top_chunk_ids = sorted_chunk_ids[:top_k]

    # 4. Construct standardized RetrievalHit objects
    fused_hits: List[RetrievalHit] = []
    max_rrf = rrf_scores[top_chunk_ids[0]] if top_chunk_ids else 1.0

    for cid in top_chunk_ids:
        raw_hit = chunk_map[cid]
        # Normalize relative to max RRF score
        norm_score = round(rrf_scores[cid] / max_rrf, 4) if max_rrf > 0 else 0.0
        fused_hits.append(
            RetrievalHit(
                chunk_id=cid,
                document_id=raw_hit["document_id"],
                filename=raw_hit["filename"],
                page_number=int(raw_hit["page_number"]),
                chunk_index=int(raw_hit["chunk_index"]),
                text=raw_hit["text"],
                score=norm_score,
                retrieval_source="RRF"
            )
        )

    return fused_hits
