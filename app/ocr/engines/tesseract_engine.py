"""Tesseract OCR engine implementation with spatial bounding box extraction."""

import os
from typing import List, Tuple
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from app.core.config import get_settings
from app.core.exceptions import OCRError
from app.core.logging import get_logger
from app.ocr.confidence import calculate_average_confidence, normalize_confidence
from app.ocr.engines.base import BaseOCREngine
from app.schemas.ocr import OCRBlock, OCRBoundingBox

settings = get_settings()
logger = get_logger("tesseract_engine")


class TesseractEngine(BaseOCREngine):
    """Baseline OCR engine utilizing Tesseract 5.5."""

    def __init__(self) -> None:
        if os.path.exists(settings.TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        self.languages = settings.TESSERACT_LANGUAGES

    @property
    def engine_name(self) -> str:
        return "TESSERACT"

    def _extract_with_config(
        self,
        rgb_img: np.ndarray,
        config: str = ""
    ) -> Tuple[str, float, List[OCRBlock]]:
        """Extract text and bounding blocks using specific Tesseract CLI configuration."""
        data = pytesseract.image_to_data(
            rgb_img,
            lang=self.languages,
            config=config,
            output_type=Output.DICT
        )

        blocks: List[OCRBlock] = []
        word_confidences: List[float] = []
        current_block_words: List[str] = []
        current_block_conf: List[float] = []
        current_bbox = [0, 0, 0, 0]  # x1, y1, x2, y2
        prev_block_num = -1

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text_word = data["text"][i].strip()
            conf = float(data["conf"][i])

            if not text_word or conf < 0:
                continue

            norm_conf = normalize_confidence(conf)
            word_confidences.append(norm_conf)

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])
            block_num = int(data["block_num"][i])

            if prev_block_num != block_num and current_block_words:
                blk_text = " ".join(current_block_words)
                avg_c = calculate_average_confidence(current_block_conf)
                bw = max(1, current_bbox[2] - current_bbox[0])
                bh = max(1, current_bbox[3] - current_bbox[1])
                blocks.append(
                    OCRBlock(
                        text=blk_text,
                        confidence=avg_c,
                        bbox=OCRBoundingBox(x=current_bbox[0], y=current_bbox[1], width=bw, height=bh),
                        block_type="paragraph"
                    )
                )
                current_block_words = []
                current_block_conf = []
                current_bbox = [x, y, x + w, y + h]
            else:
                if not current_block_words:
                    current_bbox = [x, y, x + w, y + h]
                else:
                    current_bbox[0] = min(current_bbox[0], x)
                    current_bbox[1] = min(current_bbox[1], y)
                    current_bbox[2] = max(current_bbox[2], x + w)
                    current_bbox[3] = max(current_bbox[3], y + h)

            current_block_words.append(text_word)
            current_block_conf.append(norm_conf)
            prev_block_num = block_num

        if current_block_words:
            blk_text = " ".join(current_block_words)
            avg_c = calculate_average_confidence(current_block_conf)
            bw = max(1, current_bbox[2] - current_bbox[0])
            bh = max(1, current_bbox[3] - current_bbox[1])
            blocks.append(
                OCRBlock(
                    text=blk_text,
                    confidence=avg_c,
                    bbox=OCRBoundingBox(x=current_bbox[0], y=current_bbox[1], width=bw, height=bh),
                    block_type="paragraph"
                )
            )

        full_text = "\n\n".join(b.text for b in blocks)
        mean_conf = calculate_average_confidence(word_confidences)
        return full_text, mean_conf, blocks

    def extract_text_and_blocks(
        self,
        image: np.ndarray,
        page_number: int
    ) -> Tuple[str, float, List[OCRBlock]]:
        """Extract text with automatic adaptive PSM modes (auto-detects single block, sparse, or full document)."""
        try:
            # Ensure RGB image for Tesseract
            if len(image.shape) == 2:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Pass 1: Standard Automatic Page Segmentation (--psm 3)
            best_text, best_conf, best_blocks = self._extract_with_config(rgb_img, config="--psm 3")

            # Adaptive Pass: If text is sparse (< 50 chars) or low confidence, auto-test PSM 6 (single block) & PSM 11 (sparse text)
            if len(best_text.strip()) < 50 or best_conf < 55.0:
                for alt_psm in ["--psm 6", "--psm 11", "--psm 4"]:
                    try:
                        alt_text, alt_conf, alt_blocks = self._extract_with_config(rgb_img, config=alt_psm)
                        # Score: characters * normalized confidence
                        current_score = len(best_text.strip()) * (best_conf / 100.0)
                        alt_score = len(alt_text.strip()) * (alt_conf / 100.0)
                        if alt_score > current_score and len(alt_text.strip()) > len(best_text.strip()):
                            logger.info(f"Adaptive PSM switched to {alt_psm} (Yield: {len(alt_text)} chars, Conf: {alt_conf:.1f}%).")
                            best_text, best_conf, best_blocks = alt_text, alt_conf, alt_blocks
                    except Exception:
                        pass

            return best_text, best_conf, best_blocks

        except Exception as exc:
            logger.error(f"Tesseract OCR failed on page {page_number}: {exc}")
            raise OCRError(f"Tesseract engine failure on page {page_number}: {exc}")
