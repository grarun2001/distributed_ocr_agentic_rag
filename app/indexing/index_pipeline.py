"""Dual indexing pipeline inserting into Milvus vector store and persistent BM25 index."""

from typing import List
from app.core.exceptions import IndexingError
from app.core.logging import get_logger
from app.indexing.bm25_store import get_bm25_store
from app.indexing.chunker import get_chunker
from app.indexing.embeddings import get_embedding_generator
from app.indexing.milvus_store import get_milvus_store
from app.schemas.ocr import OCRDocumentResult
from app.schemas.retrieval import DocumentChunkItem

logger = get_logger("index_pipeline")


class IndexingPipeline:
    """Orchestrates chunking, embedding generation, dense Milvus indexing, and sparse BM25 indexing."""

    def __init__(self) -> None:
        self.chunker = get_chunker()
        self.embedder = get_embedding_generator()
        self.milvus_store = get_milvus_store()
        self.bm25_store = get_bm25_store()

    def index_ocr_result(self, ocr_result: OCRDocumentResult) -> int:
        """Process structured OCR result and index into both Milvus and persistent BM25."""
        document_id = ocr_result.document_id
        filename = ocr_result.filename

        logger.info(f"Starting dual indexing for document '{document_id}' ({filename})...")

        # 1. Structure-aware passage chunking
        chunks: List[DocumentChunkItem] = self.chunker.chunk_document(ocr_result)
        if not chunks:
            logger.warning(f"No chunks generated for document '{document_id}'.")
            return 0

        logger.info(f"Generated {len(chunks)} chunk(s) for document '{document_id}'.")

        try:
            # 2. Dense vector embeddings
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_texts(texts)

            # 3. Format entities for Milvus insertion
            milvus_entities = []
            for idx, c in enumerate(chunks):
                milvus_entities.append({
                    "page_number": c.metadata.page_number,
                    "chunk_index": c.metadata.chunk_index,
                    "text": c.text,
                    "vector": embeddings[idx]
                })

            # Purge existing chunks for document_id if previously indexed
            try:
                self.milvus_store.delete_by_document_id(document_id)
            except Exception as del_err:
                logger.debug(f"Milvus pre-cleanup note for doc {document_id}: {del_err}")

            # 4. Insert into Milvus
            inserted_milvus_count = self.milvus_store.insert_chunks(
                document_id=document_id,
                file_hash=ocr_result.metadata.get("file_hash", f"hash_{document_id[:8]}"),
                filename=filename,
                chunks_data=milvus_entities
            )

            # 5. Insert into Persistent BM25 Store
            inserted_bm25_count = self.bm25_store.index_document(
                document_id=document_id,
                chunks=chunks
            )

            logger.info(
                f"Successfully dual-indexed document '{document_id}': "
                f"{inserted_milvus_count} vectors in Milvus, {inserted_bm25_count} passages in BM25."
            )
            return len(chunks)

        except Exception as exc:
            logger.error(f"Dual indexing failed for document '{document_id}': {exc}")
            raise IndexingError(f"Indexing pipeline failure: {exc}")


_index_pipeline_instance = IndexingPipeline()


def get_indexing_pipeline() -> IndexingPipeline:
    """Retrieve singleton IndexingPipeline instance."""
    return _index_pipeline_instance
