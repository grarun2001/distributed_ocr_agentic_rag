"""End-to-end OCR and extraction pipeline."""

from typing import Tuple
from app.core.exceptions import OCRError
from app.core.logging import get_logger
from app.ingestion.document_loader import get_document_loader
from app.processing.artifact_builder import build_ocr_document_result
from app.schemas.ocr import OCRDocumentResult
from app.storage.artifact_store import get_artifact_store

logger = get_logger("ocr_pipeline")


class OCRPipeline:
    """Unified document processing orchestrator."""

    def __init__(self) -> None:
        self.loader = get_document_loader()
        self.artifact_store = get_artifact_store()

    def process(
        self,
        document_id: str,
        filename: str,
        category: str,
        file_bytes: bytes,
        preferred_engine: str = "hybrid"
    ) -> Tuple[OCRDocumentResult, str]:
        """
        Run complete extraction, artifact creation, and S3 JSON persistence.
        Returns:
            Tuple of (OCRDocumentResult, minio_ocr_json_key)
        """
        try:
            logger.info(f"Starting OCR extraction pipeline for document '{document_id}' ({filename})...")

            # 1. Multi-format dynamic extraction
            pages = self.loader.load_and_extract(
                document_id=document_id,
                filename=filename,
                file_bytes=file_bytes,
                preferred_engine=preferred_engine
            )

            # 2. Build structured artifact
            result = build_ocr_document_result(
                document_id=document_id,
                filename=filename,
                category=category,
                pages=pages
            )

            # 3. Store full OCR JSON in MinIO
            ocr_key = self.artifact_store.save_ocr_result(document_id, result.model_dump())
            logger.info(f"Saved full structured OCR JSON to MinIO: '{ocr_key}' (Pages: {result.total_pages}).")

            return result, ocr_key

        except Exception as exc:
            logger.error(f"OCR Pipeline processing failed for document '{document_id}': {exc}")
            raise OCRError(f"OCR Pipeline error on document '{document_id}': {exc}")


_ocr_pipeline_instance = OCRPipeline()


def get_ocr_pipeline() -> OCRPipeline:
    """Retrieve singleton instance of OCRPipeline."""
    return _ocr_pipeline_instance
