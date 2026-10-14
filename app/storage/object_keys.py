"""Object storage key naming conventions and path generators."""

from app.core.security import sanitize_filename


def get_raw_document_key(document_id: str, filename: str) -> str:
    """Generate object key for the original uploaded file."""
    safe_name = sanitize_filename(filename)
    return f"documents/{document_id}/raw/{safe_name}"


def get_ocr_json_key(document_id: str) -> str:
    """Generate object key for the full structured OCR JSON document artifact."""
    return f"documents/{document_id}/ocr/full_result.json"


def get_page_image_key(document_id: str, page_number: int) -> str:
    """Generate object key for a rendered/preprocessed page image."""
    return f"documents/{document_id}/pages/page_{page_number:04d}.png"


def get_page_json_key(document_id: str, page_number: int) -> str:
    """Generate object key for page-level structured OCR JSON."""
    return f"documents/{document_id}/pages/page_{page_number:04d}.json"


def get_table_csv_key(document_id: str, page_number: int, table_index: int) -> str:
    """Generate object key for extracted table CSV artifact."""
    return f"documents/{document_id}/tables/page_{page_number:04d}_table_{table_index:02d}.csv"
