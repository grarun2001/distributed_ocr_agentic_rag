"""Neural Cross-Encoder reranker with graceful fallback to RRF rankings."""

from typing import List, Optional
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.retrieval import RetrievalHit

settings = get_settings()
logger = get_logger("reranker")


class CrossEncoderReranker:
    """Performs deep cross-attention scoring on candidate passages with fault tolerance."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model = None

    def _load_model(self) -> CrossEncoder:
        """Lazily load the CrossEncoder model on CPU."""
        if self._model is None:
            try:
                logger.info(f"Loading Cross-Encoder model '{self.model_name}' on CPU...")
                self._model = CrossEncoder(self.model_name, device="cpu")
                logger.info("Cross-Encoder reranker loaded successfully.")
            except Exception as exc:
                logger.warning(f"Failed loading Cross-Encoder '{self.model_name}': {exc}. Fallbacks will be active.")
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalHit],
        top_k: int = 5
    ) -> List[RetrievalHit]:
        """
        Rerank candidate passages using neural cross-attention.
        Reliability Contract: If model fails or is unavailable, returns candidate list directly.
        """
        if not candidates:
            return []

        model = self._load_model()
        if model is None:
            logger.warning("Cross-Encoder unavailable. Returning un-reranked RRF candidates.")
            return candidates[:top_k]

        try:
            # Prepare query-passage pairs
            pairs = [[query, c.text] for c in candidates]
            scores = model.predict(pairs)

            # Assign new scores and sort
            reranked_hits: List[RetrievalHit] = []
            for idx, hit in enumerate(candidates):
                raw_s = float(scores[idx])
                # Sigmoid normalization if logit
                norm_score = 1.0 / (1.0 + (2.71828 ** (-raw_s)))
                hit_copy = hit.model_copy(
                    update={
                        "score": round(norm_score, 4),
                        "retrieval_source": "RERANKED"
                    }
                )
                reranked_hits.append(hit_copy)

            reranked_hits.sort(key=lambda h: h.score, reverse=True)
            return reranked_hits[:top_k]

        except Exception as exc:
            # RELIABILITY FALLBACK: Gracefully fall back to RRF rankings without crashing
            logger.warning(f"Cross-Encoder reranking failed ({exc}). Falling back gracefully to RRF hits.")
            return candidates[:top_k]


_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Retrieve singleton CrossEncoderReranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
