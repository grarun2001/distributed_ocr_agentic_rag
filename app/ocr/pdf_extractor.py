"""PyMuPDF extractor supporting digital text layers, scanned page rendering, and mixed PDFs."""

from typing import List, Tuple
import cv2
import numpy as np
import pymupdf

from app.core.exceptions import OCRError
from app.core.logging import get_logger
from app.ocr.engine_factory import get_ocr_engine
from app.ocr.image_preprocessor import get_image_preprocessor
from app.ocr.table_extractor import get_table_extractor
from app.schemas.ocr import OCRBlock, OCRBoundingBox, OCRPage, OCRTable
from app.storage.artifact_store import get_artifact_store

logger = get_logger("pdf_extractor")


class PDFExtractor:
    """Extracts digital and scanned pages with per-page dynamic routing."""

    def __init__(self) -> None:
        self.preprocessor = get_image_preprocessor()
        self.table_extractor = get_table_extractor()
        self.artifact_store = get_artifact_store()

    def process_pdf_bytes(
        self,
        document_id: str,
        pdf_bytes: bytes,
        preferred_engine: str = "hybrid"
    ) -> List[OCRPage]:
        """Process all pages of a PDF, dynamically routing digital vs scanned pages."""
        pages_result: List[OCRPage] = []
        doc = None

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            logger.info(f"Processing PDF '{document_id}' with {total_pages} page(s)...")

            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = doc[page_idx]

                # 1. Inspect text layer in natural reading order
                extracted_text = page.get_text("text", sort=True).strip()
                char_count = len(extracted_text)

                # Heuristic: page is digital if char_count >= 50 and text is not corrupted CID font garbage
                cid_count = extracted_text.count("(cid:")
                is_readable_digital = char_count >= 50 and cid_count < 10

                if is_readable_digital:
                    # Native digital extraction with geometric layout sorting
                    blocks_raw = page.get_text("blocks", sort=True)
                    blocks: List[OCRBlock] = []
                    for b in blocks_raw:
                        x0, y0, x1, y1, text, block_no, block_type = b
                        clean_t = str(text).strip()
                        if clean_t:
                            b_type = "heading" if clean_t.startswith("#") else ("table" if clean_t.startswith("|") else "paragraph")
                            blocks.append(
                                OCRBlock(
                                    text=clean_t,
                                    confidence=100.0,
                                    bbox=OCRBoundingBox(
                                        x=int(x0),
                                        y=int(y0),
                                        width=max(1, int(x1 - x0)),
                                        height=max(1, int(y1 - y0))
                                    ),
                                    block_type=b_type
                                )
                            )

                    # Extract digital vector tables
                    tables = self.table_extractor.extract_from_pdf_page(page, page_num)

                    # Also extract any markdown tables detected in text
                    md_tables = TableExtractor.extract_tables_from_markdown(extracted_text, start_index=len(tables))
                    for m_tbl in md_tables:
                        if not any(t.headers == m_tbl.headers for t in tables):
                            tables.append(m_tbl)

                    # Save CSV artifact for each extracted table
                    for tbl in tables:
                        if tbl.csv_data:
                            try:
                                self.artifact_store.save_table_csv(document_id, page_num, tbl.table_index, tbl.csv_data)
                            except Exception:
                                pass

                    pages_result.append(
                        OCRPage(
                            page_number=page_num,
                            ocr_engine="NATIVE",
                            confidence=100.0,
                            char_count=char_count,
                            table_count=len(tables),
                            has_selectable_text=True,
                            raw_text=extracted_text,
                            blocks=blocks,
                            tables=tables,
                            warnings=[],
                            image_path=None
                        )
                    )
                    logger.info(f"Page {page_num}/{total_pages} processed natively (Digital).")

                else:
                    # Scanned or image page: render at crisp 300 DPI
                    logger.info(f"Page {page_num}/{total_pages} is scanned/image-only. Rendering to 300 DPI...")
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    cv_img = self.preprocessor.from_bytes(img_bytes)

                    # Enhance for vision while preserving full color channels
                    vision_img, warnings, metrics = self.preprocessor.preprocess_for_vision(cv_img)

                    # Store rendered page image in MinIO
                    page_bytes = self.preprocessor.to_bytes(vision_img, ".png")
                    image_key = self.artifact_store.save_page_image(document_id, page_num, page_bytes)

                    # Run OCR Engine
                    engine = get_ocr_engine(preferred_engine)
                    text, avg_conf, blocks = engine.extract_text_and_blocks(vision_img, page_num)

                    if avg_conf < 60.0:
                        warnings.append("LOW_OCR_CONFIDENCE")

                    # Extract tables: morphological grid detection + Markdown table transcription parsing
                    tables = self.table_extractor.extract_from_image(vision_img)
                    md_tables = TableExtractor.extract_tables_from_markdown(text, start_index=len(tables))
                    for m_tbl in md_tables:
                        if not any(t.headers == m_tbl.headers for t in tables):
                            tables.append(m_tbl)

                    # Save CSV artifact for each extracted table
                    for tbl in tables:
                        if tbl.csv_data:
                            try:
                                self.artifact_store.save_table_csv(document_id, page_num, tbl.table_index, tbl.csv_data)
                            except Exception:
                                pass

                    pages_result.append(
                        OCRPage(
                            page_number=page_num,
                            ocr_engine=engine.engine_name,
                            confidence=avg_conf,
                            char_count=len(text),
                            table_count=len(tables),
                            has_selectable_text=False,
                            raw_text=text,
                            blocks=blocks,
                            tables=tables,
                            warnings=warnings,
                            image_path=image_key
                        )
                    )
                    logger.info(f"Page {page_num}/{total_pages} OCR completed ({engine.engine_name}, Conf: {avg_conf}%, Tables: {len(tables)}).")

            return pages_result

        except Exception as exc:
            logger.error(f"Failed processing PDF '{document_id}': {exc}")
            raise OCRError(f"PDF extraction error for document '{document_id}': {exc}")
        finally:
            if doc is not None:
                doc.close()


_pdf_extractor_instance = PDFExtractor()


def get_pdf_extractor() -> PDFExtractor:
    """Retrieve singleton instance of PDFExtractor."""
    return _pdf_extractor_instance
