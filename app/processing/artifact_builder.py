"""Constructs structured document-level OCR artifacts and JSON representations."""

from typing import List, Set
from app.ocr.confidence import calculate_average_confidence
from app.processing.normalization import normalize_text
from app.processing.structure_preserver import preserve_document_structure
from app.schemas.ocr import OCRDocumentResult, OCRPage


def build_ocr_document_result(
    document_id: str,
    filename: str,
    category: str,
    pages: List[OCRPage]
) -> OCRDocumentResult:
    """Build standardized OCRDocumentResult from individual processed pages."""
    normalized_pages: List[OCRPage] = []
    page_confidences: List[float] = []
    doc_warnings_set: Set[str] = set()
    full_text_parts: List[str] = []
    dominant_engines: Set[str] = set()

    for p in pages:
        clean_text = normalize_text(p.raw_text)
        structured_text = preserve_document_structure(clean_text)

        p_copy = p.model_copy(update={"raw_text": structured_text, "char_count": len(structured_text)})
        normalized_pages.append(p_copy)

        page_confidences.append(p.confidence)
        dominant_engines.add(p.ocr_engine)
        for w in p.warnings:
            doc_warnings_set.add(w)

        if structured_text:
            full_text_parts.append(f"--- Page {p.page_number} ---\n{structured_text}")

    avg_conf = calculate_average_confidence(page_confidences)
    engine_summary = ", ".join(sorted(dominant_engines)) if dominant_engines else "UNKNOWN"
    combined_full_text = "\n\n".join(full_text_parts)

    return OCRDocumentResult(
        document_id=document_id,
        filename=filename,
        category=category,
        total_pages=len(normalized_pages),
        average_confidence=avg_conf,
        ocr_engine_used=engine_summary,
        pages=normalized_pages,
        full_text=combined_full_text,
        warnings=sorted(list(doc_warnings_set)),
        metadata={
            "engines_used": list(dominant_engines),
            "total_characters": len(combined_full_text),
            "total_tables": sum(p.table_count for p in normalized_pages)
        }
    )
