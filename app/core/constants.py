"""Application constants, enumerations, and system definitions."""

from enum import Enum
from typing import Dict, Set


class TaskStatus(str, Enum):
    """Document processing task lifecycle states."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    OCR_COMPLETED = "OCR_COMPLETED"
    INDEXING = "INDEXING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class DocumentCategory(str, Enum):
    """Broad category of processed document."""
    PDF_DIGITAL = "PDF_DIGITAL"
    PDF_SCANNED = "PDF_SCANNED"
    PDF_MIXED = "PDF_MIXED"
    IMAGE = "IMAGE"
    OFFICE_DOCUMENT = "OFFICE_DOCUMENT"
    SPREADSHEET = "SPREADSHEET"
    PRESENTATION = "PRESENTATION"
    STRUCTURED_TEXT = "STRUCTURED_TEXT"
    UNKNOWN = "UNKNOWN"


class OCREngineType(str, Enum):
    """Available OCR and extraction engine types."""
    NATIVE = "NATIVE"
    TESSERACT = "TESSERACT"
    PADDLEOCR = "PADDLEOCR"
    HYBRID = "HYBRID"


class RetrievalMode(str, Enum):
    """RAG retrieval strategies."""
    VECTOR_ONLY = "VECTOR_ONLY"
    BM25_ONLY = "BM25_ONLY"
    HYBRID_RRF = "HYBRID_RRF"
    HYBRID_RERANK = "HYBRID_RERANK"


class AgentRole(str, Enum):
    """Agentic AI workflow node identities."""
    ORCHESTRATOR = "orchestrator"
    RESEARCH_EVIDENCE = "research_evidence_agent"
    WEB_SEARCH = "web_search_agent"
    ANSWER_VERIFICATION = "answer_verification_agent"


# File Extension to MIME Type Mapping
MIME_TYPE_MAPPING: Dict[str, str] = {
    # PDF
    "pdf": "application/pdf",
    # Images
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "bmp": "image/bmp",
    "webp": "image/webp",
    # Office Documents
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
    "rtf": "application/rtf",
    "md": "text/markdown",
    "html": "text/html",
    # Spreadsheets
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "csv": "text/csv",
    # Presentations
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint",
    # Structured Data
    "xml": "application/xml",
}

# Supported File Extensions
SUPPORTED_EXTENSIONS: Set[str] = set(MIME_TYPE_MAPPING.keys())

# Magic byte signatures (file headers) for validation
MAGIC_SIGNATURES: Dict[str, bytes] = {
    "pdf": b"%PDF",
    "png": b"\x89PNG\r\n\x1a\n",
    "jpg": b"\xff\xd8\xff",
    "jpeg": b"\xff\xd8\xff",
    "tiff_le": b"II*\x00",
    "tiff_be": b"MM\x00*",
    "bmp": b"BM",
    "webp": b"RIFF",
    "zip_office": b"PK\x03\x04",  # docx, xlsx, pptx are ZIP containers
}

# Default System Thresholds
DEFAULT_OCR_CONFIDENCE_THRESHOLD: float = 60.0
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
DEFAULT_RRF_K: int = 60
DEFAULT_TOP_K_RETRIEVAL: int = 10
DEFAULT_TOP_K_RERANK: int = 5
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
