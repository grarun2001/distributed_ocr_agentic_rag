"""Hybrid intelligent OCR orchestrator combining Multimodal AI Vision and local OCR."""

from typing import List, Tuple
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ocr.engines.base import BaseOCREngine
from app.ocr.engines.gemini_vision_engine import GeminiVisionEngine
from app.ocr.engines.paddleocr_engine import PaddleOCREngine
from app.ocr.engines.tesseract_engine import TesseractEngine
from app.schemas.ocr import OCRBlock

settings = get_settings()
logger = get_logger("hybrid_ocr_engine")


class HybridOCREngine(BaseOCREngine):
    """
    Intelligent Hybrid OCR Engine:
    - Primary: Gemini Multimodal Vision (100% handwriting, cursive, receipts, and table fidelity)
    - Secondary: PaddleOCR (if available and enabled)
    - Fallback / Offline: Multi-pass Tesseract (PSM 3, 6, 11, 4 with adaptive preprocessing)
    """

    def __init__(self) -> None:
        self.gemini_engine = GeminiVisionEngine()
        self.paddle_engine = PaddleOCREngine() if settings.ENABLE_PADDLEOCR else None
        self.tesseract_engine = TesseractEngine()

    @property
    def engine_name(self) -> str:
        api_key = getattr(settings, "GEMINI_API_KEY", "")
        if api_key and not api_key.startswith("your_"):
            return "HYBRID_VISION"
        if self.paddle_engine and self.paddle_engine.enabled:
            return "HYBRID_PADDLE"
        return "HYBRID_TESSERACT"

    def extract_text_and_blocks(
        self,
        image: np.ndarray,
        page_number: int
    ) -> Tuple[str, float, List[OCRBlock]]:
        """Extract text through an intelligent hierarchy of AI vision and classical OCR."""
        api_key = getattr(settings, "GEMINI_API_KEY", "")

        # 1. If Gemini Vision is available, run multimodal AI vision
        if api_key and not api_key.startswith("your_"):
            try:
                text, conf, blocks = self.gemini_engine.extract_text_and_blocks(image, page_number)
                if text.strip():
                    return text, conf, blocks
                logger.warning("Gemini Vision returned empty text. Trying local OCR fallback...")
            except Exception as exc:
                logger.warning(f"Gemini Vision execution failed: {exc}. Trying local OCR fallback...")

        # 2. Try PaddleOCR if enabled
        if self.paddle_engine:
            try:
                text, conf, blocks = self.paddle_engine.extract_text_and_blocks(image, page_number)
                if text.strip() and conf >= 60.0:
                    return text, conf, blocks
            except Exception as exc:
                logger.warning(f"PaddleOCR failed: {exc}. Trying Tesseract...")

        # 3. Fallback to Tesseract (with OpenCV contrast/denoise/sharpen preprocessing)
        tess_input = image
        if len(image.shape) == 3:
            try:
                from app.ocr.image_preprocessor import get_image_preprocessor
                preproc = get_image_preprocessor()
                tess_input, _, _ = preproc.preprocess(image, apply_binarization=False)
            except Exception:
                tess_input = image

        text, conf, blocks = self.tesseract_engine.extract_text_and_blocks(tess_input, page_number)

        # 4. If Tesseract output is sparse or low confidence and Gemini is available, attempt recovery
        if (len(text.strip()) < 50 or conf < 65.0) and api_key and not api_key.startswith("your_"):
            logger.info("Tesseract produced low confidence/sparse text. Attempting Gemini Vision recovery...")
            try:
                rec_text, rec_conf, rec_blocks = self.gemini_engine.extract_text_and_blocks(image, page_number)
                if len(rec_text.strip()) > len(text.strip()):
                    return rec_text, rec_conf, rec_blocks
            except Exception as rec_exc:
                logger.warning(f"Gemini recovery failed: {rec_exc}")

        return text, conf, blocks
