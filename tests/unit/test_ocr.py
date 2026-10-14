"""Unit tests for OCR text processing, image preprocessing, and table extractors."""

import numpy as np
from PIL import Image, ImageDraw

from app.ocr.confidence import calculate_average_confidence, normalize_confidence
from app.ocr.engine_factory import get_ocr_engine
from app.ocr.image_preprocessor import get_image_preprocessor
from app.ocr.quality_checks import compute_blur_score, compute_contrast_score
from app.ocr.table_extractor import TableExtractor
from app.processing.normalization import normalize_text
from app.processing.structure_preserver import preserve_document_structure


def test_text_normalization():
    """Verify de-hyphenation and Unicode normalization."""
    raw = "This is a distri-\nbuted sys-\ntem with   excessive   spaces."
    cleaned = normalize_text(raw)
    assert "distributed system" in cleaned
    assert "excessive   spaces" not in cleaned


def test_structure_preserver():
    """Verify markdown heading and bullet preservation."""
    raw = "SECTION 1: OVERVIEW\n* Item A\n* Item B\nRegular text paragraph."
    preserved = preserve_document_structure(raw)
    assert "### SECTION 1: OVERVIEW" in preserved
    assert "- Item A" in preserved
    assert "- Item B" in preserved


def test_table_markdown_formatter():
    """Verify table Markdown formatting."""
    headers = ["Name", "Qty", "Price"]
    rows = [["Widget", "10", "$5.00"], ["Gadget", "2", "$20.00"]]
    md = TableExtractor.format_markdown_table(headers, rows)
    assert "| Name | Qty | Price |" in md
    assert "| Widget | 10 | $5.00 |" in md


def test_image_preprocessor_pipeline():
    """Verify OpenCV preprocessor runs cleanly on synthetic image array."""
    preprocessor = get_image_preprocessor()
    # Create black-and-white synthetic image
    test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    processed, warnings, metrics = preprocessor.preprocess(test_img)

    assert processed is not None
    assert len(processed.shape) == 2  # Grayscale / binarized
    assert "blur_score" in metrics
    assert "contrast_score" in metrics


def test_tesseract_ocr_synthetic():
    """Verify Tesseract engine extracts text and bounding boxes from generated image."""
    engine = get_ocr_engine("tesseract")

    # Generate synthetic image with clear text
    pil_img = Image.new("RGB", (300, 80), color=(255, 255, 255))
    d = ImageDraw.Draw(pil_img)
    d.text((20, 30), "INVOICE TOTAL", fill=(0, 0, 0))
    np_img = np.array(pil_img)

    text, conf, blocks = engine.extract_text_and_blocks(np_img, page_number=1)
    assert "INVOICE" in text.upper()
    assert conf > 50.0
    assert len(blocks) > 0
    assert blocks[0].bbox.width > 0
