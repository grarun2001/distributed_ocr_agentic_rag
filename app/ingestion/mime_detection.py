"""MIME type detection combining extension mapping and magic byte signatures."""

from pathlib import Path
from typing import Optional
from app.core.constants import MIME_TYPE_MAPPING, MAGIC_SIGNATURES


def detect_mime_type(filename: str, header_bytes: Optional[bytes] = None) -> str:
    """Detect canonical MIME type based on file extension and header signature bytes."""
    ext = Path(filename).suffix.lstrip(".").lower()

    # Check magic byte signatures if header is provided
    if header_bytes and len(header_bytes) >= 4:
        if header_bytes.startswith(MAGIC_SIGNATURES["pdf"]):
            return "application/pdf"
        elif header_bytes.startswith(MAGIC_SIGNATURES["png"]):
            return "image/png"
        elif header_bytes.startswith(MAGIC_SIGNATURES["jpg"]) or header_bytes.startswith(MAGIC_SIGNATURES["jpeg"]):
            return "image/jpeg"
        elif header_bytes.startswith(MAGIC_SIGNATURES["bmp"]):
            return "image/bmp"
        elif len(header_bytes) >= 12 and header_bytes.startswith(MAGIC_SIGNATURES["webp"]) and header_bytes[8:12] == b"WEBP":
            return "image/webp"
        elif header_bytes.startswith(MAGIC_SIGNATURES["tiff_le"]) or header_bytes.startswith(MAGIC_SIGNATURES["tiff_be"]):
            return "image/tiff"
        elif header_bytes.startswith(MAGIC_SIGNATURES["zip_office"]):
            # ZIP container could be docx, xlsx, pptx
            if ext in ["docx", "xlsx", "pptx"]:
                return MIME_TYPE_MAPPING.get(ext, "application/octet-stream")

    # Fallback to extension mapping
    return MIME_TYPE_MAPPING.get(ext, "application/octet-stream")
