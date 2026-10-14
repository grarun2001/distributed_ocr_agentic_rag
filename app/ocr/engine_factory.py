"""Factory and registry for OCR engines."""

from typing import Dict
from app.core.config import get_settings
from app.ocr.engines.base import BaseOCREngine
from app.ocr.engines.gemini_vision_engine import GeminiVisionEngine
from app.ocr.engines.hybrid_engine import HybridOCREngine
from app.ocr.engines.paddleocr_engine import PaddleOCREngine
from app.ocr.engines.tesseract_engine import TesseractEngine

settings = get_settings()

_engine_registry: Dict[str, BaseOCREngine] = {}


def get_ocr_engine(engine_name: str = "default") -> BaseOCREngine:
    """Retrieve or lazily initialize an OCR engine instance."""
    normalized_name = engine_name.lower()
    if normalized_name in ["default", "hybrid", "auto"]:
        normalized_name = "hybrid"

    if normalized_name in _engine_registry:
        return _engine_registry[normalized_name]

    if normalized_name in ["gemini", "vision", "gemini_vision"]:
        engine = GeminiVisionEngine()
    elif normalized_name == "hybrid":
        engine = HybridOCREngine()
    elif normalized_name == "paddleocr":
        engine = PaddleOCREngine()
    elif normalized_name == "tesseract":
        engine = TesseractEngine()
    else:
        # Default fallback to Hybrid
        engine = HybridOCREngine()

    _engine_registry[normalized_name] = engine
    return engine
