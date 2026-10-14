"""Multimodal AI Vision OCR engine using Google Gemini 2.5 Flash."""

import base64
import os
import re
from typing import List, Optional, Tuple
import cv2
import httpx
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ocr.engines.base import BaseOCREngine
from app.ocr.engines.tesseract_engine import TesseractEngine
from app.schemas.ocr import OCRBlock, OCRBoundingBox

settings = get_settings()
logger = get_logger("gemini_vision_engine")


class GeminiVisionEngine(BaseOCREngine):
    """
    High-accuracy multimodal OCR engine capable of recognizing:
    - Cursive and print handwriting
    - Low-resolution and mobile camera captures
    - Complex receipts, invoices, and nested tables
    - Multi-lingual and mathematical documents
    """

    def __init__(self) -> None:
        self._fallback_engine = TesseractEngine()

    @property
    def engine_name(self) -> str:
        return "GEMINI_VISION"

    def _get_api_config(self) -> Tuple[str, str]:
        current_settings = get_settings()
        api_key = getattr(current_settings, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        model_name = getattr(current_settings, "GEMINI_MODEL_NAME", "gemini-2.5-flash") or "gemini-2.5-flash"
        return api_key, model_name

    def extract_text_and_blocks(
        self,
        image: np.ndarray,
        page_number: int
    ) -> Tuple[str, float, List[OCRBlock]]:
        """Extract complete text and blocks from an image using Gemini Vision."""
        api_key, model_name = self._get_api_config()

        if not api_key or api_key.startswith("your_"):
            logger.warning("No valid GEMINI_API_KEY found. Falling back to Tesseract.")
            return self._fallback_engine.extract_text_and_blocks(image, page_number)

        try:
            # 1. Encode image to JPEG bytes
            success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                logger.warning("Failed to encode image to JPEG buffer. Falling back to Tesseract.")
                return self._fallback_engine.extract_text_and_blocks(image, page_number)

            img_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
            h, w = image.shape[:2]

            # 2. Call Gemini Vision API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            prompt = (
                "You are an advanced, high-precision OCR and document analysis engine.\n"
                "Your objective is to transcribe 100% of the content from this document image with complete verbatim accuracy.\n\n"
                "CRITICAL EXTRACTION GUIDELINES:\n"
                "1. COMPLETENESS: Extract every printed word, handwritten note, cursive script, number, code, symbol, currency sign, date, identifier, header, and footer. Do NOT omit, truncate, or summarize anything.\n"
                "2. READING ORDER: Maintain natural reading order (top-to-bottom, left-to-right). For multi-column layouts (newsletters, academic papers, multi-column resumes), process column-by-column rather than weaving horizontally across columns.\n"
                "3. STRUCTURE & HEADINGS: Use standard Markdown formatting for document structure (# Main Title, ## Section Header, ### Subsection).\n"
                "4. TABLES & TABULAR DATA: Format all tables, receipts, invoices, line items, and columnar data into clean GitHub Markdown tables (| Column 1 | Column 2 | with separator | --- | --- |). Keep all cell values aligned verbatim.\n"
                "5. LISTS: Format bullet points with '- ' and numbered lists with '1. ', '2. '.\n"
                "6. FORMS & KEY-VALUE PAIRS: For form fields and metadata (e.g. Invoice #, Date, Patient Name, Total), format cleanly as '**Key:** Value'.\n"
                "7. SPECIAL ELEMENTS: Transcribe stamps as '[Stamp: ...]', signatures as '[Signature: ...]', and checkboxes as '[x]' or '[ ]'.\n"
                "8. VERBATIM FIDELITY: Never alter numbers, spellings, or mathematical symbols. Output ONLY the transcribed document in clean Markdown. Never include conversational preambles (e.g. 'Here is the transcription:'), meta comments, or wrap the whole response in a single ```markdown code fence."
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_b64
                                }
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 8192
                }
            }

            resp = None
            for attempt in range(2):
                try:
                    resp = httpx.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=httpx.Timeout(45.0, connect=10.0)
                    )
                    if resp.status_code == 200:
                        break
                except Exception as net_err:
                    if attempt == 0:
                        import time
                        time.sleep(1.0)
                        continue
                    raise net_err

            if resp is None or resp.status_code != 200:
                err_detail = resp.text[:200] if resp else "No response"
                logger.warning(
                    f"Gemini Vision API returned status {resp.status_code if resp else 'None'}: {err_detail}. "
                    "Falling back to Tesseract."
                )
                return self._fallback_engine.extract_text_and_blocks(image, page_number)

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini Vision returned empty candidates. Falling back to Tesseract.")
                return self._fallback_engine.extract_text_and_blocks(image, page_number)

            raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

            # Strip markdown code block wrapper if model wrapped entire output in ```markdown ... ```
            cleaned_text = re.sub(r"^```(?:markdown)?\s*\n", "", raw_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r"\n```\s*$", "", cleaned_text).strip()

            # Strip any introductory conversational chatter
            cleaned_text = re.sub(
                r"^(?:Here(?:'s| is) (?:the )?(?:verbatim )?(?:extracted )?(?:transcription|text)[^\n]*\n+|Transcribed Text:[^\n]*\n+)",
                "",
                cleaned_text,
                flags=re.IGNORECASE
            ).strip()

            if not cleaned_text:
                logger.warning("Gemini Vision returned empty text after cleanup. Falling back to Tesseract.")
                return self._fallback_engine.extract_text_and_blocks(image, page_number)

            # 3. Create structured OCRBlocks from paragraphs/sections
            blocks: List[OCRBlock] = []
            paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [line.strip() for line in cleaned_text.split("\n") if line.strip()]

            total_paras = max(1, len(paragraphs))
            slice_h = max(10, h // total_paras)

            for idx, para in enumerate(paragraphs):
                y_start = min(h - 10, idx * slice_h)
                box_h = min(h - y_start, slice_h)

                if para.startswith("#"):
                    b_type = "heading"
                elif para.startswith("|"):
                    b_type = "table"
                elif para.startswith(("- ", "* ", "1. ", "2. ", "3. ")):
                    b_type = "list"
                else:
                    b_type = "paragraph"

                blocks.append(
                    OCRBlock(
                        text=para,
                        confidence=99.0,
                        bbox=OCRBoundingBox(
                            x=0,
                            y=y_start,
                            width=w,
                            height=box_h
                        ),
                        block_type=b_type
                    )
                )

            logger.info(
                f"Gemini Vision transcribed page {page_number} successfully "
                f"({len(cleaned_text)} chars, {len(blocks)} blocks)."
            )
            return cleaned_text, 99.0, blocks

        except Exception as exc:
            logger.warning(f"Gemini Vision error on page {page_number}: {exc}. Falling back to Tesseract.")
            return self._fallback_engine.extract_text_and_blocks(image, page_number)
