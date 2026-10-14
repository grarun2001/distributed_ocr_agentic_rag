"""Pydantic schemas for text chunking, hybrid search queries, and retrieval results."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata context preserved for each indexed chunk."""
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original filename")
    page_number: int = Field(..., description="Page number where chunk originated")
    chunk_index: int = Field(..., description="0-indexed sequence position within document")
    char_count: int = Field(..., description="Character length of chunk")
    has_table: bool = Field(False, description="Flag if chunk contains tabular data")
    source_type: str = Field("document", description="Source classification: document, table, code")


class DocumentChunkItem(BaseModel):
    """Normalized chunk item ready for dual indexing."""
    chunk_id: str = Field(..., description="Unique deterministic chunk ID: {document_id}_{chunk_index}")
    text: str = Field(..., description="Extracted content text")
    metadata: ChunkMetadata = Field(..., description="Contextual chunk metadata")


class RetrievalHit(BaseModel):
    """A single ranked retrieval match."""
    chunk_id: str = Field(..., description="Matched chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    filename: str = Field(..., description="Filename")
    page_number: int = Field(..., description="Page number")
    chunk_index: int = Field(..., description="Chunk index")
    text: str = Field(..., description="Chunk content text")
    score: float = Field(..., description="Normalized fusion or reranker relevance score")
    retrieval_source: str = Field(..., description="Origin: DENSE, BM25, RRF, RERANKED")

    model_config = {"from_attributes": True}


class HybridSearchQuery(BaseModel):
    """Request payload for hybrid dense + sparse retrieval."""
    query: str = Field(..., min_length=1, description="Search question or keyword query")
    top_k: int = Field(5, ge=1, le=20, description="Number of final reranked hits to return")
    document_id: Optional[str] = Field(None, description="Optional filter scoping search to a specific document")
    mode: str = Field("HYBRID_RERANK", description="Retrieval mode: VECTOR_ONLY, BM25_ONLY, HYBRID_RRF, HYBRID_RERANK")


class HybridSearchResponse(BaseModel):
    """Response containing fused and reranked hits."""
    query: str
    total_hits: int
    mode: str
    hits: List[RetrievalHit]
