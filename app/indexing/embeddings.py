"""Dense vector embedding generator using Sentence-Transformers (CPU-optimized)."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.exceptions import IndexingError
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger("embeddings")


class EmbeddingGenerator:
    """Computes dense vector representations with dimension matching Milvus schema."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.expected_dim = settings.MILVUS_EMBEDDING_DIM
        self._model = None

    def _load_model(self) -> SentenceTransformer:
        """Lazily load SentenceTransformer model onto CPU."""
        if self._model is None:
            try:
                logger.info(f"Loading embedding model '{self.model_name}' on CPU...")
                self._model = SentenceTransformer(self.model_name, device="cpu")
                # Validate dimension matches Milvus configuration
                sample_emb = self._model.encode("test", normalize_embeddings=True)
                actual_dim = len(sample_emb)
                if actual_dim != self.expected_dim:
                    raise IndexingError(
                        f"Embedding dimension mismatch: Model '{self.model_name}' outputs {actual_dim}-dim, "
                        f"but Milvus configuration requires {self.expected_dim}-dim."
                    )
                logger.info(f"Embedding model loaded successfully (Dimension: {actual_dim}).")
            except Exception as exc:
                logger.error(f"Failed loading embedding model '{self.model_name}': {exc}")
                raise IndexingError(f"Embedding initialization error: {exc}")
        return self._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute normalized dense embeddings for a batch of text passages."""
        if not texts:
            return []
        model = self._load_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as exc:
            logger.error(f"Error encoding batch of {len(texts)} texts: {exc}")
            raise IndexingError(f"Embedding inference failed: {exc}")

    def embed_query(self, query: str) -> List[float]:
        """Compute normalized dense embedding for a single search question."""
        model = self._load_model()
        try:
            emb = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
            return emb.tolist()
        except Exception as exc:
            logger.error(f"Error encoding query '{query}': {exc}")
            raise IndexingError(f"Query embedding inference failed: {exc}")


_embedding_generator_instance: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Retrieve singleton EmbeddingGenerator instance."""
    global _embedding_generator_instance
    if _embedding_generator_instance is None:
        _embedding_generator_instance = EmbeddingGenerator()
    return _embedding_generator_instance
