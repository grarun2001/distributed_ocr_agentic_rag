"""Quality assessment heuristics for images and OCR results."""

from typing import List, Tuple
import cv2
import numpy as np


def compute_blur_score(image: np.ndarray) -> float:
    """Compute Laplacian variance to detect image blurriness (higher is sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_contrast_score(image: np.ndarray) -> float:
    """Compute standard deviation of pixel intensities (higher is better contrast)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(np.std(gray))


def detect_skew_angle(image: np.ndarray) -> float:
    """Detect dominant text line skew angle in degrees using minAreaRect."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find non-zero points
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    (w, h) = rect[1]
    if w < h:
        angle = angle - 90 if angle > 0 else angle + 90

    while angle < -45:
        angle += 90
    while angle > 45:
        angle -= 90

    # Only correct subtle real tilts (e.g. between 0.5 and 15 degrees).
    # If the detected angle is greater than 15 degrees, it's document aspect ratio, not line skew.
    if abs(angle) > 15.0 or abs(angle) < 0.5:
        return 0.0

    return float(round(angle, 2))


def assess_image_quality(image: np.ndarray) -> Tuple[List[str], dict]:
    """Run quality checks and return list of warnings and metrics dictionary."""
    warnings: List[str] = []
    blur_score = compute_blur_score(image)
    contrast_score = compute_contrast_score(image)
    skew_angle = detect_skew_angle(image)

    # Thresholds
    if blur_score < 100.0:
        warnings.append("BLURRY_IMAGE")
    if contrast_score < 30.0:
        warnings.append("LOW_IMAGE_CONTRAST")
    if abs(skew_angle) > 2.0:
        warnings.append("HIGH_SKEW_DETECTED")

    metrics = {
        "blur_score": round(blur_score, 2),
        "contrast_score": round(contrast_score, 2),
        "skew_angle": round(skew_angle, 2)
    }
    return warnings, metrics
