"""Pydantic schemas for structured OCR outputs, bounding boxes, and tables."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OCRBoundingBox(BaseModel):
    """Bounding box coordinates: [x, y, width, height]."""
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Box width in pixels")
    height: int = Field(..., description="Box height in pixels")


class OCRBlock(BaseModel):
    """Structured text block (paragraph, line, heading) with spatial location."""
    text: str = Field(..., description="Extracted text content")
    confidence: float = Field(..., description="OCR confidence score (0.0 - 100.0)")
    bbox: OCRBoundingBox = Field(..., description="Bounding box on original or rendered page")
    block_type: str = Field("paragraph", description="Type: heading, paragraph, table, list_item")


class OCRTable(BaseModel):
    """Extracted table representation."""
    table_index: int = Field(..., description="0-indexed table position on page")
    num_rows: int = Field(..., description="Row count")
    num_cols: int = Field(..., description="Column count")
    headers: List[str] = Field(default_factory=list, description="Extracted column headers")
    rows: List[List[str]] = Field(default_factory=list, description="Matrix of cell values")
    csv_data: str = Field("", description="CSV-formatted representation")
    markdown_data: str = Field("", description="Markdown table representation")
    confidence: float = Field(100.0, description="Extraction confidence")


class OCRPage(BaseModel):
    """Full extraction metadata for a single document page."""
    page_number: int = Field(..., description="1-indexed page number")
    ocr_engine: str = Field(..., description="Engine used: NATIVE, TESSERACT, PADDLEOCR, HYBRID")
    confidence: float = Field(..., description="Average page-level confidence score (0.0 - 100.0)")
    char_count: int = Field(..., description="Total characters extracted on this page")
    table_count: int = Field(0, description="Number of detected tables")
    has_selectable_text: bool = Field(False, description="True if digital selectable text was present")
    raw_text: str = Field(..., description="Combined page text content")
    blocks: List[OCRBlock] = Field(default_factory=list, description="Granular bounding box blocks")
    tables: List[OCRTable] = Field(default_factory=list, description="Extracted tabular structures")
    warnings: List[str] = Field(default_factory=list, description="Quality warnings e.g. LOW_CONFIDENCE, SKEWED")
    image_path: Optional[str] = Field(None, description="MinIO key to rendered/preprocessed page image")


class OCRDocumentResult(BaseModel):
    """Complete document-level structured OCR and extraction artifact."""
    document_id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Original file name")
    category: str = Field(..., description="Document category")
    total_pages: int = Field(..., description="Total pages in document")
    average_confidence: float = Field(..., description="Weighted average confidence across all pages")
    ocr_engine_used: str = Field(..., description="Primary or dominant engine used")
    pages: List[OCRPage] = Field(..., description="List of per-page extraction records")
    full_text: str = Field(..., description="Combined full document text")
    warnings: List[str] = Field(default_factory=list, description="Document-level quality warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")
