"""Milvus vector database connection, schema definition, and search manager."""

from typing import Any, Dict, List, Optional
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility
)

from app.core.config import get_settings
from app.core.exceptions import IndexingError, RetrievalError
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger("milvus_store")


class MilvusStore:
    """Manages Milvus vector collection, indexing, and dense vector searches."""

    def __init__(self) -> None:
        self.host = settings.MILVUS_HOST
        self.port = str(settings.MILVUS_PORT)
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dim = settings.MILVUS_EMBEDDING_DIM
        self._connected = False

    def connect(self) -> None:
        """Establish connection to Milvus standalone server."""
        if not self._connected:
            try:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    user=settings.MILVUS_USER or None,
                    password=settings.MILVUS_PASSWORD or None
                )
                self._connected = True
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            except Exception as exc:
                logger.error(f"Failed connecting to Milvus: {exc}")
                raise IndexingError(f"Milvus connection error: {exc}")

    def ensure_collection(self) -> Collection:
        """Create Milvus collection and IVF_FLAT/HNSW index if not present."""
        self.connect()
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.load()
                return collection

            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            schema = CollectionSchema(
                fields=fields,
                description="Distributed OCR Document Chunks with Dense Embeddings"
            )

            collection = Collection(name=self.collection_name, schema=schema)

            # Create vector index (IVF_FLAT with COSINE similarity)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            collection.load()
            logger.info(f"Created and loaded Milvus collection '{self.collection_name}'.")
            return collection

        except Exception as exc:
            logger.error(f"Error ensuring Milvus collection '{self.collection_name}': {exc}")
            raise IndexingError(f"Milvus collection error: {exc}")

    def insert_chunks(
        self,
        document_id: str,
        file_hash: str,
        filename: str,
        chunks_data: List[Dict[str, Any]]
    ) -> int:
        """Insert dense embeddings and chunk metadata into Milvus."""
        collection = self.ensure_collection()
        if not chunks_data:
            return 0

        try:
            entities = [
                [document_id] * len(chunks_data),
                [file_hash] * len(chunks_data),
                [filename] * len(chunks_data),
                [chunk["page_number"] for chunk in chunks_data],
                [chunk["chunk_index"] for chunk in chunks_data],
                [chunk["text"][:4090] for chunk in chunks_data],
                [chunk["vector"] for chunk in chunks_data]
            ]
            mr = collection.insert(entities)
            collection.flush()
            logger.info(f"Inserted {len(chunks_data)} chunks for doc '{document_id}' into Milvus.")
            return len(mr.primary_keys)
        except Exception as exc:
            logger.error(f"Failed inserting chunks into Milvus: {exc}")
            raise IndexingError(f"Milvus insertion failed: {exc}")

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform dense vector nearest-neighbor search with Cosine metric."""
        collection = self.ensure_collection()
        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["id", "document_id", "filename", "page_number", "chunk_index", "text"]
            )

            hits_list = []
            for hit in results[0]:
                hits_list.append({
                    "id": hit.id,
                    "score": float(hit.score),
                    "document_id": hit.entity.get("document_id"),
                    "filename": hit.entity.get("filename"),
                    "page_number": int(hit.entity.get("page_number")),
                    "chunk_index": int(hit.entity.get("chunk_index")),
                    "text": hit.entity.get("text")
                })
            return hits_list
        except Exception as exc:
            logger.error(f"Milvus vector search failed: {exc}")
            raise RetrievalError(f"Milvus search error: {exc}")

    def delete_by_document_id(self, document_id: str) -> None:
        """Purge all chunks associated with a document ID."""
        collection = self.ensure_collection()
        try:
            expr = f'document_id == "{document_id}"'
            collection.delete(expr)
            collection.flush()
            logger.info(f"Deleted vector chunks for document '{document_id}'.")
        except Exception as exc:
            logger.error(f"Failed deleting vectors for doc '{document_id}': {exc}")
            raise IndexingError(f"Milvus delete error: {exc}")


_milvus_store_instance: Optional[MilvusStore] = None


def get_milvus_store() -> MilvusStore:
    """Retrieve singleton instance of MilvusStore."""
    global _milvus_store_instance
    if _milvus_store_instance is None:
        _milvus_store_instance = MilvusStore()
    return _milvus_store_instance
