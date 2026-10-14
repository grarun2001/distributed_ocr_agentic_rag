"""Abstract base class for OCR engines."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from app.schemas.ocr import OCRBlock


class BaseOCREngine(ABC):
    """Abstract interface for all optical character recognition engines."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Identifier name of the OCR engine."""
        pass

    @abstractmethod
    def extract_text_and_blocks(
        self,
        image: np.ndarray,
        page_number: int
    ) -> Tuple[str, float, List[OCRBlock]]:
        """
        Extract text and structured spatial blocks from an image array.
        Returns:
            Tuple of (combined_text, average_confidence, list_of_ocr_blocks)
        """
        pass
