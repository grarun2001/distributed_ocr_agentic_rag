"""Document routing classifier determining category and extraction paths."""

from pathlib import Path
from app.core.constants import DocumentCategory


def classify_document_category(filename: str, content_type: str) -> DocumentCategory:
    """Classify incoming file into broad processing category."""
    ext = Path(filename).suffix.lstrip(".").lower()

    if ext == "pdf" or content_type == "application/pdf":
        return DocumentCategory.PDF_DIGITAL

    if ext in ["png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"] or content_type.startswith("image/"):
        return DocumentCategory.IMAGE

    if ext in ["docx", "doc", "txt", "rtf", "md", "html"]:
        return DocumentCategory.OFFICE_DOCUMENT

    if ext in ["xlsx", "xls", "csv"]:
        return DocumentCategory.SPREADSHEET

    if ext in ["pptx", "ppt"]:
        return DocumentCategory.PRESENTATION

    if ext in ["xml", "json"]:
        return DocumentCategory.STRUCTURED_TEXT

    return DocumentCategory.UNKNOWN
