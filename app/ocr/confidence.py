"""Confidence score normalization and threshold assessment."""

from typing import List
from app.core.config import get_settings

settings = get_settings()


def normalize_confidence(score: float, scale_max: float = 100.0) -> float:
    """Normalize score to 0.0 - 100.0 scale."""
    if scale_max <= 1.0:
        score = score * 100.0
    return max(0.0, min(100.0, round(float(score), 2)))


def calculate_average_confidence(confidences: List[float]) -> float:
    """Compute mean confidence over list of scores, excluding negative/invalid scores."""
    valid_scores = [s for s in confidences if s >= 0.0]
    if not valid_scores:
        return 0.0
    return round(sum(valid_scores) / len(valid_scores), 2)


def is_confidence_acceptable(confidence: float) -> bool:
    """Check if confidence score satisfies the system threshold."""
    return confidence >= settings.OCR_CONFIDENCE_THRESHOLD
