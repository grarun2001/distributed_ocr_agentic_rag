"""Pydantic schemas for document metadata, pages, and listing responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PageSummary(BaseModel):
    """Page-level extraction summary."""
    page_number: int
    ocr_engine: str
    confidence: Optional[float] = None
    char_count: int
    table_count: int
    has_selectable_text: bool
    warnings: Optional[List[str]] = None
    raw_text: Optional[str] = None

    model_config = {"from_attributes": True}


class DocumentDetailResponse(BaseModel):
    """Comprehensive document information with secure download URLs."""
    id: str = Field(..., description="Document UUID")
    file_hash: str = Field(..., description="SHA-256 fingerprint")
    filename: str = Field(..., description="File name")
    file_size: int = Field(..., description="Size in bytes")
    content_type: str = Field(..., description="MIME type")
    category: str = Field(..., description="Document category")
    status: str = Field(..., description="Current status")
    ocr_engine_used: Optional[str] = None
    ocr_confidence: Optional[float] = None
    total_pages: int
    total_chunks: int
    ocr_text_preview: Optional[str] = None
    error_message: Optional[str] = None
    task_id: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    raw_file_url: Optional[str] = Field(None, description="Presigned URL to download raw document")
    ocr_json_url: Optional[str] = Field(None, description="Presigned URL to download full OCR JSON artifact")
    pages: List[PageSummary] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated collection of documents."""
    total: int = Field(..., description="Total documents matching criteria")
    items: List[DocumentDetailResponse] = Field(..., description="Page items")
