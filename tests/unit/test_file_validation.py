"""Unit tests for file validation and security checks."""

import pytest
from app.core.exceptions import FileValidationError
from app.ingestion.file_validator import get_file_validator


def test_file_validator_valid_pdf():
    """Verify validator accepts valid PDF header and size."""
    validator = get_file_validator()
    mime = validator.validate("report.pdf", 1024, b"%PDF-1.5 test")
    assert mime == "application/pdf"


def test_file_validator_reject_empty():
    """Verify validator rejects 0 byte empty files."""
    validator = get_file_validator()
    with pytest.raises(FileValidationError, match="is empty"):
        validator.validate("empty.pdf", 0, b"")


def test_file_validator_reject_unsupported_extension():
    """Verify validator blocks unsupported extensions (e.g. .exe, .sh)."""
    validator = get_file_validator()
    with pytest.raises(FileValidationError, match="not supported"):
        validator.validate("malware.exe", 1024, b"MZ\x90\x00")


def test_file_validator_reject_corrupt_magic_bytes():
    """Verify validator blocks file with extension .pdf but bad magic header."""
    validator = get_file_validator()
    with pytest.raises(FileValidationError, match="Invalid PDF header"):
        validator.validate("spoofed.pdf", 512, b"GIF89a fake header")
