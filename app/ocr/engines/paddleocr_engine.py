"""PaddleOCR engine implementation with angle classification and layout awareness."""

from typing import List, Optional, Tuple
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ocr.confidence import calculate_average_confidence, normalize_confidence
from app.ocr.engines.base import BaseOCREngine
from app.ocr.engines.tesseract_engine import TesseractEngine
from app.schemas.ocr import OCRBlock, OCRBoundingBox

settings = get_settings()
logger = get_logger("paddleocr_engine")


class PaddleOCREngine(BaseOCREngine):
    """Advanced OCR engine with layout analysis and angle classification."""

    def __init__(self) -> None:
        self._paddle_ocr = None
        self._fallback_engine = TesseractEngine()
        self.enabled = settings.ENABLE_PADDLEOCR

        if self.enabled:
            try:
                from paddleocr import PaddleOCR
                self._paddle_ocr = PaddleOCR(
                    use_angle_cls=settings.PADDLEOCR_USE_ANGLE_CLS,
                    lang=settings.PADDLEOCR_LANG,
                    show_log=False
                )
                logger.info("PaddleOCR engine initialized successfully.")
            except Exception as exc:
                logger.warning(f"PaddleOCR initialization failed: {exc}. Will use Tesseract fallback.")
                self._paddle_ocr = None

    @property
    def engine_name(self) -> str:
        return "PADDLEOCR" if self._paddle_ocr is not None else "TESSERACT_FALLBACK"

    def extract_text_and_blocks(
        self,
        image: np.ndarray,
        page_number: int
    ) -> Tuple[str, float, List[OCRBlock]]:
        """Extract text with PaddleOCR or fallback to Tesseract."""
        if self._paddle_ocr is None:
            return self._fallback_engine.extract_text_and_blocks(image, page_number)

        try:
            results = self._paddle_ocr.ocr(image, cls=settings.PADDLEOCR_USE_ANGLE_CLS)
            if not results or not results[0]:
                return "", 0.0, []

            blocks: List[OCRBlock] = []
            confidences: List[float] = []

            for line in results[0]:
                poly = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_content = str(line[1][0]).strip()
                raw_conf = float(line[1][1])

                if not text_content:
                    continue

                conf = normalize_confidence(raw_conf, scale_max=1.0)
                confidences.append(conf)

                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                blocks.append(
                    OCRBlock(
                        text=text_content,
                        confidence=conf,
                        bbox=OCRBoundingBox(
                            x=x_min,
                            y=y_min,
                            width=max(1, x_max - x_min),
                            height=max(1, y_max - y_min)
                        ),
                        block_type="paragraph"
                    )
                )

            full_text = "\n".join(b.text for b in blocks)
            avg_conf = calculate_average_confidence(confidences)
            return full_text, avg_conf, blocks

        except Exception as exc:
            logger.warning(f"PaddleOCR execution error on page {page_number}: {exc}. Invoking fallback.")
            return self._fallback_engine.extract_text_and_blocks(image, page_number)
