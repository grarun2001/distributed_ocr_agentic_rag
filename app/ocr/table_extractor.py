"""Table detection and extraction for digital PDFs, spreadsheets, and scanned documents."""

import csv
import io
import json
import os
import re
from typing import Any, Dict, List, Optional
import cv2
import httpx
import numpy as np
import pymupdf

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.ocr import OCRTable

logger = get_logger("table_extractor")


class TableExtractor:
    """Extracts tables and outputs CSV and Markdown representations."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @classmethod
    def extract_tables_from_markdown(cls, text: str, start_index: int = 0) -> List[OCRTable]:
        """Parse GitHub markdown formatted tables from text into structured OCRTable objects."""
        if not text or "|" not in text:
            return []

        lines = [line.strip() for line in text.split("\n")]
        tables: List[OCRTable] = []
        table_lines: List[str] = []

        def process_block(blk_lines: List[str]):
            if len(blk_lines) < 3:
                return
            # Second line must be a separator line: | --- | --- | or |--|:--:|
            sep_line = blk_lines[1]
            if not re.match(r"^\|?(\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?$", sep_line):
                return

            def parse_line(line_str: str) -> List[str]:
                parts = line_str.split("|")
                if parts and parts[0].strip() == "":
                    parts = parts[1:]
                if parts and parts[-1].strip() == "":
                    parts = parts[:-1]
                return [p.strip() for p in parts]

            headers = parse_line(blk_lines[0])
            rows = [parse_line(l) for l in blk_lines[2:]]
            valid_rows = [r for r in rows if any(r)]
            if not headers or not valid_rows:
                return

            col_count = len(headers)
            normalized_rows = []
            for r in valid_rows:
                if len(r) < col_count:
                    r = r + [""] * (col_count - len(r))
                elif len(r) > col_count:
                    r = r[:col_count]
                normalized_rows.append(r)

            idx = start_index + len(tables)
            md_str = cls.format_markdown_table(headers, normalized_rows)
            csv_str = cls.format_csv_table(headers, normalized_rows)

            tables.append(
                OCRTable(
                    table_index=idx,
                    num_rows=len(normalized_rows) + 1,
                    num_cols=col_count,
                    headers=headers,
                    rows=normalized_rows,
                    csv_data=csv_str,
                    markdown_data=md_str,
                    confidence=99.0
                )
            )

        in_table = False
        for line in lines:
            if "|" in line:
                table_lines.append(line)
                in_table = True
            else:
                if in_table and table_lines:
                    process_block(table_lines)
                    table_lines = []
                    in_table = False

        if table_lines:
            process_block(table_lines)

        return tables


    @staticmethod
    def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
        """Format table as a standard GitHub Markdown table."""
        if not headers and not rows:
            return ""

        effective_headers = headers if headers else [f"Col {i+1}" for i in range(len(rows[0]))]
        header_line = "| " + " | ".join(effective_headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(effective_headers)) + " |"
        row_lines = ["| " + " | ".join(r) + " |" for r in rows]

        return "\n".join([header_line, sep_line] + row_lines)

    @staticmethod
    def format_csv_table(headers: List[str], rows: List[List[str]]) -> str:
        """Format table as CSV string."""
        out = io.StringIO()
        writer = csv.writer(out)
        if headers:
            writer.writerow(headers)
        for r in rows:
            writer.writerow(r)
        return out.getvalue().strip()

    def has_table_grid_structure(self, cv_img: np.ndarray) -> bool:
        """Fast morphological check to detect whether image contains table grid cells."""
        if cv_img is None or cv_img.size == 0:
            return False

        try:
            if len(cv_img.shape) == 3:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_img

            h, w = gray.shape[:2]
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Proportional kernels for horizontal and vertical lines
            h_len = max(20, min(50, w // 25))
            v_len = max(20, min(50, h // 25))

            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
            h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
            v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

            grid = cv2.add(h_lines, v_lines)
            contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter candidate cells
            cells = [
                c for c in contours
                if 40 < cv2.boundingRect(c)[2] < (w * 0.95) and 15 < cv2.boundingRect(c)[3] < (h * 0.5)
            ]

            return len(cells) >= 4
        except Exception as exc:
            logger.debug(f"Error checking table grid structure: {exc}")
            return False

    def extract_tables_with_vision(self, image_bytes: bytes) -> List[OCRTable]:
        """Extract structured tables using multimodal vision model."""
        api_key = getattr(self.settings, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        model_name = getattr(self.settings, "GEMINI_MODEL_NAME", "gemini-2.5-flash") or "gemini-2.5-flash"

        if not api_key or api_key.startswith("your_"):
            logger.warning("No valid GEMINI_API_KEY configured for vision table extraction.")
            return []

        import base64
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "Analyze this document page image. Detect any structured tables present.\n"
            "Extract each table with its headers and rows verbatim.\n"
            "Return JSON ONLY with this schema:\n"
            "{\n"
            '  "has_tables": true,\n'
            '  "tables": [\n'
            "    {\n"
            '      "headers": ["Col 1", "Col 2"],\n'
            '      "rows": [\n'
            '        ["Cell 1", "Cell 2"]\n'
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            'If no tables are present on this page, return {"has_tables": false, "tables": []}.'
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }

        resp = None
        for attempt in range(3):
            try:
                resp = httpx.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=35.0)
                if resp.status_code == 200:
                    break
                elif resp.status_code in [429, 503]:
                    logger.warning(f"Vision table API returned {resp.status_code} (attempt {attempt+1}/3). Retrying in {(attempt+1)*2}s...")
                    import time
                    time.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    logger.warning(f"Vision table API returned status {resp.status_code}: {resp.text[:200]}")
                    return []
            except Exception as net_err:
                if attempt < 2:
                    import time
                    time.sleep(2.0 * (attempt + 1))
                    continue
                logger.error(f"Vision table extraction network error: {net_err}")
                return []

        if resp is None or resp.status_code != 200:
            return []

        try:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return []

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return []

            result_json = json.loads(parts[0].get("text", "{}"))
            raw_tables = result_json.get("tables", [])

            tables_found: List[OCRTable] = []
            for idx, tbl in enumerate(raw_tables):
                headers = [str(h or "").strip() for h in tbl.get("headers", [])]
                rows = [[str(cell or "").strip() for cell in r] for r in tbl.get("rows", [])]

                if not headers and not rows:
                    continue

                md = self.format_markdown_table(headers, rows)
                csv_str = self.format_csv_table(headers, rows)

                tables_found.append(
                    OCRTable(
                        table_index=idx,
                        num_rows=len(rows) + (1 if headers else 0),
                        num_cols=len(headers) if headers else (len(rows[0]) if rows else 0),
                        headers=headers,
                        rows=rows,
                        csv_data=csv_str,
                        markdown_data=md,
                        confidence=98.0
                    )
                )

            logger.info(f"Vision table extraction found {len(tables_found)} table(s).")
            return tables_found

        except Exception as exc:
            logger.error(f"Vision table extraction failed: {exc}")
            return []

    def extract_from_pdf_page(self, page: pymupdf.Page, page_num: int) -> List[OCRTable]:
        """Extract structured tables natively from a digital PDF page using PyMuPDF or Vision fallback."""
        tables_found: List[OCRTable] = []

        # 1. Native PyMuPDF vector table extraction
        try:
            tabs = page.find_tables()
            for idx, tab in enumerate(tabs):
                df = tab.extract()
                if not df or len(df) < 1:
                    continue

                headers = [str(col or "").strip() for col in df[0]]
                rows = [[str(cell or "").strip() for cell in r] for r in df[1:]]

                # Filter out accidental non-table text blocks
                if len(rows) >= 1 or len(headers) >= 2:
                    md = self.format_markdown_table(headers, rows)
                    csv_str = self.format_csv_table(headers, rows)

                    tables_found.append(
                        OCRTable(
                            table_index=idx,
                            num_rows=len(rows) + (1 if headers else 0),
                            num_cols=len(headers) if headers else (len(rows[0]) if rows else 0),
                            headers=headers,
                            rows=rows,
                            csv_data=csv_str,
                            markdown_data=md,
                            confidence=100.0
                        )
                    )
        except Exception as exc:
            logger.debug(f"PyMuPDF native find_tables error on page {page_num}: {exc}")

        if tables_found:
            return tables_found

        # 2. Vision table extraction for scanned or image-based tables
        try:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if self.has_table_grid_structure(cv_img):
                logger.info(f"Page {page_num}: Table grid structure detected via morphology. Extracting tables with AI Vision...")
                tables_found = self.extract_tables_with_vision(img_bytes)
        except Exception as exc:
            logger.warning(f"Error during fallback table extraction on page {page_num}: {exc}")

        return tables_found

    def extract_from_image(self, image: np.ndarray) -> List[OCRTable]:
        """Extract structured tables from an image array."""
        if image is None or image.size == 0:
            return []

        try:
            if self.has_table_grid_structure(image):
                logger.info("Table grid structure detected in image. Extracting with AI Vision...")
                success, buf = cv2.imencode(".png", image)
                if success:
                    return self.extract_tables_with_vision(buf.tobytes())
        except Exception as exc:
            logger.warning(f"Error extracting tables from image: {exc}")

        return []

    def detect_table_regions_in_image(self, gray_image: np.ndarray) -> List[OCRTable]:
        """Backward compatible alias for table extraction on image."""
        return self.extract_from_image(gray_image)


_table_extractor_instance = TableExtractor()


def get_table_extractor() -> TableExtractor:
    """Retrieve singleton instance of TableExtractor."""
    return _table_extractor_instance
