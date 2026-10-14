"""Pydantic schemas for file upload requests and responses."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Metadata response for a single uploaded document."""
    document_id: str = Field(..., description="Unique UUID of the stored document")
    filename: str = Field(..., description="Original filename uploaded by client")
    file_hash: str = Field(..., description="SHA-256 cryptographic digest of document bytes")
    file_size: int = Field(..., description="Size of file in bytes")
    content_type: str = Field(..., description="Detected MIME content type")
    category: str = Field(..., description="Broad document category (e.g. PDF_DIGITAL, IMAGE)")
    status: str = Field(..., description="Current processing lifecycle status")
    task_id: Optional[str] = Field(None, description="Background task tracking identifier")
    is_duplicate: bool = Field(False, description="Flag indicating if identical file hash already exists")
    created_at: datetime = Field(..., description="Timestamp when record was created")

    model_config = {"from_attributes": True}


class BatchUploadResponse(BaseModel):
    """Response containing outcome of multi-file batch upload."""
    total_files: int = Field(..., description="Total count of files processed in upload batch")
    duplicates_count: int = Field(..., description="Count of duplicate files identified")
    documents: List[UploadResponse] = Field(..., description="List of created or matched document records")
