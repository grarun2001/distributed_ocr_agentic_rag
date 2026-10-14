"""Hybrid search engine combining Milvus dense retrieval, persistent BM25, and neural reranking."""

from typing import List, Optional
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.indexing.bm25_store import get_bm25_store
from app.indexing.embeddings import get_embedding_generator
from app.indexing.milvus_store import get_milvus_store
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.reranker import get_reranker
from app.schemas.retrieval import RetrievalHit

logger = get_logger("hybrid_search")


class HybridSearchEngine:
    """Coordinates multi-modal vector and keyword retrieval strategies."""

    def __init__(self) -> None:
        self.embedder = get_embedding_generator()
        self.milvus_store = get_milvus_store()
        self.bm25_store = get_bm25_store()
        self.reranker = get_reranker()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_document_id: Optional[str] = None,
        mode: str = "HYBRID_RERANK"
    ) -> List[RetrievalHit]:
        """
        Execute search according to mode:
        VECTOR_ONLY, BM25_ONLY, HYBRID_RRF, HYBRID_RERANK
        """
        try:
            logger.info(f"Executing search: query='{query}', mode='{mode}', top_k={top_k}")

            # 1. Mode: BM25 Only
            if mode == "BM25_ONLY":
                sparse_hits = self.bm25_store.search(query, top_k=top_k, filter_document_id=filter_document_id)
                return [
                    RetrievalHit(
                        chunk_id=h["chunk_id"],
                        document_id=h["document_id"],
                        filename=h["filename"],
                        page_number=h["page_number"],
                        chunk_index=h["chunk_index"],
                        text=h["text"],
                        score=h["score"],
                        retrieval_source="BM25"
                    )
                    for h in sparse_hits
                ]

            # 2. Mode: Vector Only
            if mode == "VECTOR_ONLY":
                q_vec = self.embedder.embed_query(query)
                expr = f'document_id == "{filter_document_id}"' if filter_document_id else None
                dense_hits = self.milvus_store.search_dense(q_vec, top_k=top_k, filter_expr=expr)
                return [
                    RetrievalHit(
                        chunk_id=f"{h['document_id']}_{h['chunk_index']}",
                        document_id=h["document_id"],
                        filename=h["filename"],
                        page_number=h["page_number"],
                        chunk_index=h["chunk_index"],
                        text=h["text"],
                        score=round(h["score"], 4),
                        retrieval_source="DENSE"
                    )
                    for h in dense_hits
                ]

            # 3. Hybrid Modes (HYBRID_RRF and HYBRID_RERANK)
            # A. Retrieve top 20 candidates from Milvus
            q_vec = self.embedder.embed_query(query)
            expr = f'document_id == "{filter_document_id}"' if filter_document_id else None
            dense_hits = self.milvus_store.search_dense(q_vec, top_k=20, filter_expr=expr)

            # B. Retrieve top 20 candidates from Persistent BM25
            sparse_hits = self.bm25_store.search(query, top_k=20, filter_document_id=filter_document_id)

            # C. Reciprocal Rank Fusion (k=60)
            fused_candidates = reciprocal_rank_fusion(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                k=60,
                top_k=max(10, top_k * 2)
            )

            if mode == "HYBRID_RRF":
                return fused_candidates[:top_k]

            # D. Neural Cross-Encoder Reranking
            reranked_hits = self.reranker.rerank(
                query=query,
                candidates=fused_candidates,
                top_k=top_k
            )
            return reranked_hits

        except Exception as exc:
            logger.error(f"Hybrid search failed: {exc}")
            raise RetrievalError(f"Search engine error: {exc}")


_search_engine_instance: Optional[HybridSearchEngine] = None


def get_search_engine() -> HybridSearchEngine:
    """Retrieve singleton HybridSearchEngine instance."""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = HybridSearchEngine()
    return _search_engine_instance
